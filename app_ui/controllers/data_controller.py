# app_ui/controllers/data_controller.py

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from sc_core.data_management.sc_format import build_sc_format

from app_ui.state import keys
from app_ui.state.invalidation import maybe_invalidate_on_data_config_change
from app_ui.state.workflow import recompute_workflow_from_state
from app_ui.state.invalidation import invalidate_from_data_change


@dataclass
class DataConfigValidation:
    ok: bool
    errors: List[str]

def standardize_merge_keys(
    df: pd.DataFrame,
    *,
    city_col: str,
    date_col: str,
) -> pd.DataFrame:
    """
    Return a copy of df where the selected city/date columns are renamed
    to the canonical merge keys: 'ville' and 'date'.
    """
    if city_col not in df.columns:
        raise ValueError(f"Column '{city_col}' not found in dataframe.")
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe.")

    out = df.copy()

    rename_map = {}
    if city_col != "ville":
        rename_map[city_col] = "ville"
    if date_col != "date":
        rename_map[date_col] = "date"

    out = out.rename(columns=rename_map)
    return out

def _normalize_year_list(values: Sequence[Any]) -> List[int]:
    cleaned: List[int] = []
    for v in values:
        if v is None:
            continue
        try:
            cleaned.append(int(v))
        except Exception:
            continue
    return sorted(set(cleaned))


def _safe_sorted_unique_strings(values: Sequence[Any]) -> List[str]:
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s:
            cleaned.append(s)
    return sorted(set(cleaned))

def _looks_like_single_column_parse(df: pd.DataFrame) -> bool:
    """
    Heuristic: detect when a CSV has probably been parsed with the wrong separator.

    Typical symptom:
    - only one column
    - header still contains separators like ';' or ','
    - values also still contain separators
    """
    if df is None or df.empty:
        return False

    if df.shape[1] != 1:
        return False

    col_name = str(df.columns[0])

    suspicious_header = any(sep in col_name for sep in [",", ";", "\t", "|"])

    sample_values = df.iloc[:20, 0].astype(str).tolist()
    suspicious_values = any(
        any(sep in val for sep in [",", ";", "\t", "|"])
        for val in sample_values
    )

    return suspicious_header or suspicious_values


def _score_parsed_dataframe(df: pd.DataFrame) -> tuple[int, int]:
    """
    Scoring heuristic to choose the best parse candidate.

    Higher is better.
    Priority:
    - more than one column
    - more rows
    """
    if df is None:
        return (-10**9, -10**9)

    n_cols = int(df.shape[1])
    n_rows = int(df.shape[0])

    # Penalize suspicious one-column parses very heavily
    if _looks_like_single_column_parse(df):
        return (-10**6, n_rows)

    return (n_cols, n_rows)

def load_uploaded_dataframe(uploaded_file) -> pd.DataFrame:
    """
    Read a user-uploaded file into a DataFrame.

    Supported:
    - csv
    - xlsx / xls / xlsm

    The CSV loader is robust to:
    - multiple separators
    - multiple encodings
    - malformed rows
    - wrong single-column parses caused by bad separator detection
    """
    if uploaded_file is None:
        raise ValueError("No uploaded file provided.")

    filename = str(uploaded_file.name).lower()

    raw = uploaded_file.read()
    if not raw:
        raise ValueError("Uploaded file is empty.")

    if filename.endswith((".xlsx", ".xls", ".xlsm")):
        try:
            return pd.read_excel(BytesIO(raw))
        except Exception as exc:
            raise ValueError(f"Unable to read Excel file: {exc}") from exc

    if not filename.endswith(".csv"):
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")

    candidate_encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    candidate_separators = [",", ";", "\t", "|"]

    errors = []
    candidates: list[tuple[str, str, str, pd.DataFrame]] = []

    for encoding in candidate_encodings:
        for sep in candidate_separators:
            # Pass 1: strict / fast
            try:
                df = pd.read_csv(
                    BytesIO(raw),
                    sep=sep,
                    encoding=encoding,
                    engine="c",
                )
                candidates.append(("strict", encoding, sep, df))
            except Exception as exc:
                errors.append(f"[strict] encoding={encoding}, sep={repr(sep)} -> {exc}")

            # Pass 2: python engine
            try:
                df = pd.read_csv(
                    BytesIO(raw),
                    sep=sep,
                    encoding=encoding,
                    engine="python",
                )
                candidates.append(("python", encoding, sep, df))
            except Exception as exc:
                errors.append(f"[python] encoding={encoding}, sep={repr(sep)} -> {exc}")

            # Pass 3: tolerant mode
            try:
                df = pd.read_csv(
                    BytesIO(raw),
                    sep=sep,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                )
                candidates.append(("skip-bad-lines", encoding, sep, df))
            except Exception as exc:
                errors.append(f"[skip-bad-lines] encoding={encoding}, sep={repr(sep)} -> {exc}")

    if not candidates:
        error_preview = "\n".join(errors[:8])
        raise ValueError(
            "Unable to read CSV file after multiple attempts.\n"
            "Possible causes: malformed quotes, inconsistent separators, corrupted rows, or encoding issue.\n\n"
            f"First attempts:\n{error_preview}"
        )

    # Keep only non-empty candidates
    candidates = [c for c in candidates if c[3] is not None and not c[3].empty]

    if not candidates:
        raise ValueError("CSV file could be opened, but every parsing attempt returned an empty dataframe.")

    # Pick the best parse using our heuristic
    best_mode, best_encoding, best_sep, best_df = max(
        candidates,
        key=lambda x: _score_parsed_dataframe(x[3]),
    )

    # Final guard: reject suspicious one-column parse
    if _looks_like_single_column_parse(best_df):
        tried = ", ".join([f"{enc}/{repr(sep)}" for _, enc, sep, _ in candidates[:10]])
        raise ValueError(
            "The file appears to have been parsed into a single suspicious column.\n"
            "This usually means the separator is wrong or the file is malformed.\n"
            f"Best attempt used encoding={best_encoding}, sep={repr(best_sep)}, mode={best_mode}.\n"
            f"Attempts tried: {tried}"
        )

    return best_df

def load_multiple_uploaded_dataframes(uploaded_files) -> Dict[str, pd.DataFrame]:
    """
    Load multiple uploaded files into a dict {filename: dataframe}.
    """
    if not uploaded_files:
        raise ValueError("No uploaded files provided.")

    loaded: Dict[str, pd.DataFrame] = {}
    for file in uploaded_files:
        df = load_uploaded_dataframe(file)
        loaded[file.name] = df

    return loaded


def merge_uploaded_dataframes(
    dfs_by_name: Dict[str, pd.DataFrame],
    *,
    how: str = "outer",
) -> pd.DataFrame:
    """
    Merge multiple already-standardized dataframes on canonical keys ('ville', 'date').

    Non-key duplicated columns are renamed with a file prefix.
    """
    if not dfs_by_name:
        raise ValueError("No dataframes to merge.")

    how = str(how).lower().strip()
    if how not in {"outer", "inner", "left", "right"}:
        raise ValueError("how must be one of {'outer', 'inner', 'left', 'right'}.")

    prepared = []
    seen_non_key_cols = set()

    for name, df in dfs_by_name.items():
        if "ville" not in df.columns:
            raise ValueError(f"Column 'ville' not found in standardized file '{name}'.")
        if "date" not in df.columns:
            raise ValueError(f"Column 'date' not found in standardized file '{name}'.")

        local_df = df.copy()
        non_keys = [c for c in local_df.columns if c not in {"ville", "date"}]

        rename_map = {}
        for c in non_keys:
            if c in seen_non_key_cols:
                safe_name = name.rsplit(".", 1)[0].replace(" ", "_")
                rename_map[c] = f"{safe_name}__{c}"
            else:
                seen_non_key_cols.add(c)

        if rename_map:
            local_df = local_df.rename(columns=rename_map)

        prepared.append(local_df)

    merged = prepared[0]
    for nxt in prepared[1:]:
        merged = pd.merge(
            merged,
            nxt,
            on=["ville", "date"],
            how=how,
        )

    return merged

def infer_candidate_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Infer candidate columns for city/date/outcome/covariates.
    """
    all_cols = list(df.columns)

    text_like = []
    numeric_like = []
    datetime_like = []

    for col in all_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            numeric_like.append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            datetime_like.append(col)
            continue

        # soft datetime detection from strings
        try:
            sample = s.dropna().astype(str).head(50)
            parsed = pd.to_datetime(
                sample,
                errors="coerce",
                format="mixed",
                dayfirst=True,
            )
            if parsed.notna().mean() >= 0.7:
                datetime_like.append(col)
                continue
        except Exception:
            pass

        text_like.append(col)

    return {
        "all": all_cols,
        "city_candidates": text_like if text_like else all_cols,
        "date_candidates": datetime_like + [c for c in all_cols if c not in datetime_like],
        "y_candidates": numeric_like if numeric_like else all_cols,
    }


def extract_available_units_and_years(
    df: pd.DataFrame,
    city_col: str,
    date_col: str,
) -> Tuple[List[str], List[int]]:
    """
    Build sorted lists of unique units and years from selected columns.
    Robust to:
    - datetime columns
    - integer year columns
    - string year columns
    - mixed date strings
    """
    if city_col not in df.columns:
        raise ValueError(f"Column '{city_col}' not found in dataframe.")
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in dataframe.")

    units = _safe_sorted_unique_strings(df[city_col].tolist())

    s = df[date_col]

    # Case 1: already datetime
    if pd.api.types.is_datetime64_any_dtype(s):
        years = _normalize_year_list(s.dt.year.dropna().tolist())
        return units, years

    # Case 2: numeric year directly
    s_num = pd.to_numeric(s, errors="coerce")
    valid_num = s_num.dropna()

    if not valid_num.empty:
        # if values look like years
        if ((valid_num >= 1000) & (valid_num <= 3000)).mean() >= 0.8:
            years = _normalize_year_list(valid_num.astype(int).tolist())
            return units, years

    # Case 3: parse mixed date strings
    s_dt = pd.to_datetime(
        s.astype(str),
        errors="coerce",
        format="mixed",
        dayfirst=True,
    )
    valid_dt = s_dt.dropna()

    if not valid_dt.empty:
        years = _normalize_year_list(valid_dt.dt.year.tolist())
        return units, years

    raise ValueError(
        f"Unable to infer years from column '{date_col}'. "
        "Check that the selected time column contains years or valid date strings."
    )


def build_data_config(
    *,
    city_col: str,
    date_col: str,
    y_col: str,
    treated_city: str,
    donor_cities: Sequence[str],
    treatment_year: int,
    covariate_cols: Sequence[str],
    y_feature_years: Sequence[int],
    cov_feature_years: Sequence[int],
    normalize_X: bool,
    normalize_method: str,
    allow_empty_features: bool = False,
) -> Dict[str, Any]:
    """
    Canonical immutable-like config dict stored in session_state.
    """
    return {
        "city_col": str(city_col),
        "date_col": str(date_col),
        "y_col": str(y_col),
        "treated_city": str(treated_city),
        "donor_cities": list(_safe_sorted_unique_strings(donor_cities)),
        "treatment_year": int(treatment_year),
        "covariate_cols": list(dict.fromkeys([str(c) for c in covariate_cols])),
        "y_feature_years": _normalize_year_list(y_feature_years),
        "cov_feature_years": _normalize_year_list(cov_feature_years),
        "normalize_X": bool(normalize_X),
        "normalize_method": str(normalize_method),
        "allow_empty_features": bool(allow_empty_features),
    }


def validate_data_config(
    df: Optional[pd.DataFrame],
    config: Dict[str, Any],
) -> DataConfigValidation:
    """
    Validate UI selections before calling build_sc_format.
    """
    errors: List[str] = []

    if df is None:
        errors.append("No dataframe is loaded.")

    required_keys = [
        "city_col",
        "date_col",
        "y_col",
        "treated_city",
        "donor_cities",
        "treatment_year",
        "covariate_cols",
        "y_feature_years",
        "cov_feature_years",
        "normalize_X",
        "normalize_method",
        "allow_empty_features",
    ]
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing config key: {key}")

    if errors:
        return DataConfigValidation(ok=False, errors=errors)

    if config["city_col"] not in df.columns:
        errors.append(f"city_col='{config['city_col']}' not found.")
    if config["date_col"] not in df.columns:
        errors.append(f"date_col='{config['date_col']}' not found.")
    if config["y_col"] not in df.columns:
        errors.append(f"y_col='{config['y_col']}' not found.")

    donors = list(config["donor_cities"])
    treated_city = str(config["treated_city"]).strip()

    if not treated_city:
        errors.append("Treated unit must be selected.")
    if len(donors) == 0:
        errors.append("At least one donor unit must be selected.")
    if treated_city in donors:
        errors.append("Treated unit cannot also appear in donor pool.")

    if len(config["y_feature_years"]) == 0 and not config["allow_empty_features"]:
        errors.append("At least one outcome feature year must be selected.")

    normalize_method = config["normalize_method"]
    if normalize_method not in {"robust", "zscore"}:
        errors.append("normalize_method must be one of {'robust', 'zscore'}.")

    missing_covs = [c for c in config["covariate_cols"] if c not in df.columns]
    if missing_covs:
        errors.append(f"Missing covariates in dataframe: {missing_covs}")

    return DataConfigValidation(ok=(len(errors) == 0), errors=errors)


def build_sc_format_from_config(
    df: pd.DataFrame,
    config: Dict[str, Any],
):
    """Delegate SCFormat construction to the backend, capturing any imputation warnings.

    Imputation warnings raised by build_sc_format (via Python's warnings module)
    are surfaced here as Streamlit warnings so the user sees them in the UI.
    """
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always", UserWarning)

        sc_fmt = build_sc_format(
            df=df,
            y_col=config["y_col"],
            treated_city=config["treated_city"],
            donor_cities=config["donor_cities"],
            treatment_year=int(config["treatment_year"]),
            covariate_cols=config["covariate_cols"],
            y_feature_years=config["y_feature_years"],
            cov_feature_years=config["cov_feature_years"],
            city_col=config["city_col"],
            date_col=config["date_col"],
            normalize_X=config["normalize_X"],
            normalize_method=config["normalize_method"],
            allow_empty_features=config["allow_empty_features"],
        )

    # Surface any warnings captured from sc_core to the UI
    for w in caught:
        if issubclass(w.category, UserWarning):
            st.warning(f"⚠️ {w.message}")

    return sc_fmt


def persist_loaded_dataframe(df: pd.DataFrame) -> None:
    """
    Persist a newly merged raw dataframe.

    This is a structural data change and must invalidate every downstream
    object derived from the previous dataset.
    """
    invalidate_from_data_change()

    st.session_state[keys.DF_RAW] = df
    st.session_state[keys.DATA_SUMMARY] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
    }
    recompute_workflow_from_state()


def persist_built_sc_format(config: Dict[str, Any], sc_format: Any) -> None:
    """
    Persist config + built sc_format.

    If the data configuration changed, all downstream estimation artifacts
    must be invalidated before storing the new SCFormat.
    """
    maybe_invalidate_on_data_config_change(config)
    st.session_state[keys.DATA_CONFIG] = config
    st.session_state[keys.SC_FORMAT] = sc_format
    st.session_state[keys.EXPLORATION_SUMMARY] = None
    recompute_workflow_from_state()


def build_sc_format_summary(sc_format: Any) -> Dict[str, Any]:
    """
    Small summary block for UI cards.
    """
    summary: Dict[str, Any] = {}

    years = getattr(sc_format, "years", None)
    y1 = getattr(sc_format, "Y1", None)
    y0 = getattr(sc_format, "Y0", None)
    x1 = getattr(sc_format, "X1", None)
    x0 = getattr(sc_format, "X0", None)
    group_names = getattr(sc_format, "group_names", None)
    pre_mask = getattr(sc_format, "pre_mask", None)
    post_mask = getattr(sc_format, "post_mask", None)

    try:
        summary["n_periods"] = int(len(years)) if years is not None else None
    except Exception:
        summary["n_periods"] = None

    try:
        summary["n_donors"] = int(y0.shape[1]) if y0 is not None else None
    except Exception:
        summary["n_donors"] = None

    try:
        summary["n_features"] = int(x0.shape[0]) if x0 is not None else None
    except Exception:
        summary["n_features"] = None

    try:
        summary["n_pre_periods"] = int(pre_mask.sum()) if pre_mask is not None else None
    except Exception:
        summary["n_pre_periods"] = None

    try:
        summary["n_post_periods"] = int(post_mask.sum()) if post_mask is not None else None
    except Exception:
        summary["n_post_periods"] = None

    summary["group_names"] = list(group_names) if group_names is not None else []

    # lightweight shape diagnostics
    try:
        summary["Y1_shape"] = tuple(y1.shape) if hasattr(y1, "shape") else None
    except Exception:
        summary["Y1_shape"] = None

    try:
        summary["Y0_shape"] = tuple(y0.shape) if hasattr(y0, "shape") else None
    except Exception:
        summary["Y0_shape"] = None

    try:
        summary["X1_shape"] = tuple(x1.shape) if hasattr(x1, "shape") else None
    except Exception:
        summary["X1_shape"] = None

    try:
        summary["X0_shape"] = tuple(x0.shape) if hasattr(x0, "shape") else None
    except Exception:
        summary["X0_shape"] = None

    return summary