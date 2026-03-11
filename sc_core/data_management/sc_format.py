"""
sc_core.data_management.sc_format
==================================
Central data contract for SC-APP.

SCFormat is the single pivot object that carries all prepared data
between the UI layer and the scientific core.  This module is
**Streamlit-free**: it never imports streamlit.  Imputation warnings
are raised via the standard `warnings` module so callers (data_controller)
can catch and surface them in the UI.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core data contract
# ---------------------------------------------------------------------------

@dataclass
class SCFormat:
    """
    Pivot object carrying all prepared data for a Synthetic Control estimation.

    Attributes
    ----------
    Metadata / provenance
        city_col, date_col, y_col, covariate_cols,
        y_feature_years, cov_feature_years

    SCM identity / time structure
        treated, donors, years, T0, pre_mask, post_mask

    Outcome / covariate matrices
        Y1  : (T,)      treated outcome
        Y0  : (T, J)    donor outcomes
        X1  : (K,)      treated feature vector
        X0  : (K, J)    donor feature matrix

    Feature design metadata
        row_var, group_names, feature_names, X_long
    """

    # Metadata
    city_col: str
    date_col: str
    y_col: str
    covariate_cols: List[str]
    y_feature_years: List[int]
    cov_feature_years: List[int]

    # SCM identity / time
    treated: str
    donors: List[str]
    years: Any          # np.ndarray[int]
    T0: int
    pre_mask: Any       # np.ndarray[bool]
    post_mask: Any      # np.ndarray[bool]

    # Matrices
    Y1: Any             # np.ndarray (T,)
    Y0: Any             # np.ndarray (T, J)
    X1: Any             # np.ndarray (K,)
    X0: Any             # np.ndarray (K, J)

    # Feature design
    row_var: List[str]
    group_names: List[str]
    feature_names: List[str]
    X_long: Any         # pd.DataFrame


# ---------------------------------------------------------------------------
# Frozen design object (train/val sub-selection)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class XDesign:
    """Frozen sub-selection of feature matrices for train/val splits."""
    X1: np.ndarray
    X0: np.ndarray
    X_long: pd.DataFrame
    feature_names: List[str]
    row_var: List[str]
    group_names: List[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_year_series(s: pd.Series) -> pd.Series:
    """Convert a date column to integer calendar years (Int64).

    Accepts integer-like values and datetime-like strings.
    """
    y = pd.to_numeric(s, errors="coerce")
    if y.isna().mean() > 0.5:
        dt = pd.to_datetime(s, errors="coerce")
        y = dt.dt.year
    return y.astype("Int64")


def _is_dynamic_cov(
    d: pd.DataFrame,
    city_col: str,
    date_col: str,
    col: str,
) -> bool:
    """Return True if the covariate has >= 2 distinct time observations for any city."""
    tmp = d[[city_col, date_col, col]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp[np.isfinite(tmp[col])].copy()

    if tmp.empty:
        return False

    ny = tmp.groupby(city_col, sort=False)[date_col].nunique(dropna=True)
    return bool((ny >= 2).any())


def _normalize_x_matrix(
    X_mat: np.ndarray,
    normalize_X: bool,
    normalize_method: str,
) -> np.ndarray:
    """Row-wise normalisation of the feature matrix.

    normalize_method: 'robust' (median/IQR) or 'zscore' (mean/std).
    Rows that are all-missing or have zero scale are left unchanged.
    """
    if (not normalize_X) or X_mat.size == 0:
        return X_mat

    Xn = X_mat.copy()
    row_has_any = np.any(np.isfinite(Xn), axis=1)

    center = np.zeros(Xn.shape[0], dtype=float)
    scale = np.ones(Xn.shape[0], dtype=float)

    if np.any(row_has_any):
        X_valid = Xn[row_has_any, :]

        if normalize_method == "robust":
            center_v = np.nanmedian(X_valid, axis=1)
            q75 = np.nanpercentile(X_valid, 75, axis=1)
            q25 = np.nanpercentile(X_valid, 25, axis=1)
            scale_v = q75 - q25
        elif normalize_method == "zscore":
            center_v = np.nanmean(X_valid, axis=1)
            scale_v = np.nanstd(X_valid, axis=1)
        else:
            raise ValueError("normalize_method must be 'robust' or 'zscore'.")

        scale_v = np.where(~np.isfinite(scale_v) | (scale_v <= 1e-12), 1.0, scale_v)
        center_v = np.where(~np.isfinite(center_v), 0.0, center_v)

        center[row_has_any] = center_v
        scale[row_has_any] = scale_v

    scale = np.where(~np.isfinite(scale) | (scale <= 1e-12), 1.0, scale)
    center = np.where(~np.isfinite(center), 0.0, center)

    return (Xn - center.reshape(-1, 1)) / scale.reshape(-1, 1)


def _impute_missing(X: np.ndarray, strategy: str = "zero") -> np.ndarray:
    """Replace NaN values in a 2-D array.

    Parameters
    ----------
    X        : array with possible NaNs
    strategy : 'zero', 'mean', or 'median'  (row-wise for mean/median)

    Returns
    -------
    np.ndarray — NaN-free copy
    """
    X = np.asarray(X, dtype=float)
    X_out = X.copy()

    if strategy == "zero":
        X_out[np.isnan(X_out)] = 0.0
        return X_out

    if strategy in ("mean", "median"):
        row_stat = (
            np.nanmean(X_out, axis=1)
            if strategy == "mean"
            else np.nanmedian(X_out, axis=1)
        )
        row_stat = np.where(np.isnan(row_stat), 0.0, row_stat)
        idx = np.where(np.isnan(X_out))
        X_out[idx] = row_stat[idx[0]]
        return X_out

    raise ValueError("strategy must be 'zero', 'mean', or 'median'.")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_sc_format(
    df: pd.DataFrame,
    y_col: str,
    treated_city: str,
    donor_cities: Sequence[str],
    treatment_year: int,
    covariate_cols: Optional[Sequence[str]] = None,
    y_feature_years: Optional[Sequence[int]] = None,
    cov_feature_years: Optional[Sequence[int]] = None,
    city_col: str = "ville",
    date_col: str = "date",
    normalize_X: bool = True,
    normalize_method: str = "robust",
    allow_empty_features: bool = False,
    strategy: str = "zero",
) -> SCFormat:
    """Build a validated SCFormat from a raw panel DataFrame.

    Parameters
    ----------
    df               : raw panel (long format)
    y_col            : outcome column name
    treated_city     : name of the treated unit
    donor_cities     : sequence of donor unit names
    treatment_year   : first post-treatment period (T0)
    covariate_cols   : covariate column names (optional, default None → no covariates)
    y_feature_years  : pre-treatment years used as outcome features (default: all pre-years)
    cov_feature_years: pre-treatment years for covariate features (default: all pre-years)
    city_col         : panel unit column name  (default 'ville')
    date_col         : panel time column name  (default 'date')
    normalize_X      : row-wise normalisation of X matrices
    normalize_method : 'robust' (median/IQR) or 'zscore' (mean/std)
    allow_empty_features: if False, raise when no features could be built
    strategy         : imputation strategy for NaN in X ('zero', 'mean', 'median')

    Returns
    -------
    SCFormat

    Raises
    ------
    ValueError  : on any invalid configuration
    UserWarning : if NaN imputation was applied
    """
    # ---- Input guards ----------------------------------------
    d = df.copy()

    if city_col not in d.columns:
        raise ValueError(f"city_col='{city_col}' not found in dataframe.")
    if date_col not in d.columns:
        raise ValueError(f"date_col='{date_col}' not found in dataframe.")
    if y_col not in d.columns:
        raise ValueError(f"y_col='{y_col}' not found in dataframe.")

    treated_city = str(treated_city).strip()
    donors = [str(c).strip() for c in donor_cities if str(c).strip()]

    if not donors:
        raise ValueError("At least one donor city must be provided.")
    if treated_city in donors:
        raise ValueError("treated_city cannot also appear in donor_cities.")

    T0 = int(treatment_year)
    # FIX C3: guard against covariate_cols=None
    covariate_cols_clean: List[str] = [str(c) for c in (covariate_cols or [])]

    d[city_col] = d[city_col].astype(str).str.strip()
    d[date_col] = _to_year_series(d[date_col])

    keep_cols = [city_col, date_col, y_col] + covariate_cols_clean
    keep_cols = [c for c in keep_cols if c in d.columns]
    d = d[keep_cols].copy()
    d = d[d[city_col].isin([treated_city] + donors)].copy()

    if d.empty:
        raise ValueError("No rows left after filtering treated and donor cities.")

    # ---- Outcome wide matrix ---------------------------------
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    wideY = (
        d.pivot_table(
            index=date_col, columns=city_col, values=y_col, aggfunc="mean"
        )
        .sort_index()
    )

    try:
        wideY.index = wideY.index.astype(int)
    except Exception:
        try:
            wideY.index = pd.Index(
                pd.to_numeric(wideY.index, errors="coerce").astype("Int64")
            )
        except Exception:
            pass

    wideY = wideY.loc[~pd.isna(wideY.index)].copy()

    if treated_city not in wideY.columns:
        raise ValueError(
            f"Treated city '{treated_city}' not found in outcome wide table."
        )
    for c in donors:
        if c not in wideY.columns:
            raise ValueError(f"Donor city '{c}' not found in outcome wide table.")

    years = wideY.index.to_numpy()
    if len(years) == 0:
        raise ValueError("No time periods found after pivoting outcome data.")

    Y1 = wideY[treated_city].to_numpy(dtype=float)
    Y0 = wideY[donors].to_numpy(dtype=float)

    pre_mask = years < T0
    post_mask = years >= T0

    if not np.any(pre_mask):
        raise ValueError("No pre-treatment periods found before treatment_year.")
    if not np.any(post_mask):
        raise ValueError("No post-treatment periods found at or after treatment_year.")

    # ---- Outcome feature years --------------------------------
    pre_years_all = years[pre_mask]

    if y_feature_years is None:
        y_feat_years: List[int] = [int(y) for y in pre_years_all.tolist()]
    else:
        y_feat_years = [int(x) for x in y_feature_years if x is not None]

    # ---- Pre-treatment frame for covariates ------------------
    pre_df = d[d[date_col] < T0].copy()

    if cov_feature_years is None:
        cov_years: List[int] = sorted(
            [
                int(x)
                for x in pd.unique(pre_df[date_col].dropna()).tolist()
                if x is not None
            ]
        )
    else:
        # FIX C2: compute cov_years locally; never use the original parameter in return
        cov_years = sorted([int(x) for x in cov_feature_years if x is not None])

    # ---- Build X_long ----------------------------------------
    rows: List[Dict[str, Any]] = []

    def _append_row(var_name: str, year: Optional[int], series_row: pd.Series) -> None:
        rec: Dict[str, Any] = {
            "var_name": str(var_name),
            "date": (np.nan if year is None else int(year)),
        }
        rec[treated_city] = float(series_row.get(treated_city, np.nan))
        for c in donors:
            rec[c] = float(series_row.get(c, np.nan))
        rows.append(rec)

    # 1) Outcome lags
    for yy in y_feat_years:
        if yy in wideY.index:
            _append_row("y", int(yy), wideY.loc[int(yy)])

    # 2) Covariates
    if covariate_cols_clean:
        for col in covariate_cols_clean:
            if col not in pre_df.columns:
                continue

            pre_df[col] = pd.to_numeric(pre_df[col], errors="coerce")
            is_dyn = _is_dynamic_cov(pre_df, city_col, date_col, col)

            if is_dyn:
                wideX = (
                    pre_df.pivot_table(
                        index=date_col, columns=city_col, values=col, aggfunc="mean"
                    )
                    .sort_index()
                )
                wideX = wideX.loc[~pd.isna(wideX.index)].copy()
                try:
                    wideX.index = pd.Index(
                        pd.to_numeric(wideX.index, errors="coerce").astype("Int64")
                    ).astype(int)
                except Exception:
                    pass

                for yy in cov_years:
                    if yy in wideX.index:
                        _append_row(str(col), int(yy), wideX.loc[int(yy)])
            else:
                s = (
                    pre_df[[city_col, col]]
                    .dropna(subset=[col])
                    .groupby(city_col, sort=False)[col]
                    .first()
                )
                s_row = pd.Series(
                    {u: s.get(u, np.nan) for u in [treated_city] + donors}
                )
                _append_row(str(col), None, s_row)

    # ---- Assemble and sort X_long ----------------------------
    X_long = pd.DataFrame(rows)
    if X_long.empty:
        X_long = pd.DataFrame(columns=["var_name", "date", treated_city] + donors)

    if not X_long.empty:
        def _sort_block(block: pd.DataFrame) -> pd.DataFrame:
            if block.empty:
                return block
            dt = pd.to_numeric(block["date"], errors="coerce")
            block = block.assign(
                _isna=dt.isna().astype(int),
                _dt=dt.fillna(-(10 ** 18)).astype(float),
            )
            return (
                block.sort_values(
                    by=["var_name", "_isna", "_dt"],
                    ascending=[True, True, False],
                    kind="mergesort",
                )
                .drop(columns=["_isna", "_dt"])
            )

        y_block = _sort_block(X_long[X_long["var_name"] == "y"].copy())
        rest = _sort_block(X_long[X_long["var_name"] != "y"].copy())
        X_long = pd.concat([y_block, rest], axis=0, ignore_index=True)

    if X_long.empty and not allow_empty_features:
        raise ValueError(
            "No predictor features were constructed. "
            "Check covariates and feature years."
        )

    # ---- Solver matrices X1, X0 ------------------------------
    unit_cols = [treated_city] + donors
    X_mat = (
        X_long[unit_cols].to_numpy(dtype=float)
        if len(X_long)
        else np.zeros((0, len(unit_cols)), dtype=float)
    )

    # Drop entirely-missing rows
    if X_mat.size > 0:
        valid_row_mask = ~np.all(~np.isfinite(X_mat), axis=1)
        if not np.all(valid_row_mask):
            X_long = X_long.loc[valid_row_mask].reset_index(drop=True)
            X_mat = X_mat[valid_row_mask, :]

    # Impute — FIX C1: no Streamlit import; use warnings.warn instead
    if X_mat.size > 0 and np.isnan(X_mat).any():
        n_nan = int(np.isnan(X_mat).sum())
        warnings.warn(
            f"[build_sc_format] {n_nan} missing value(s) imputed in feature matrix X "
            f"using strategy='{strategy}'. "
            "Verify input data and selected feature years.",
            UserWarning,
            stacklevel=2,
        )
        X_mat = _impute_missing(X_mat, strategy=strategy)

    # Row-wise normalisation
    X_mat = _normalize_x_matrix(
        X_mat=X_mat,
        normalize_X=normalize_X,
        normalize_method=normalize_method,
    )

    if X_mat.shape[0]:
        X1 = X_mat[:, 0].astype(float)
        X0 = X_mat[:, 1:].astype(float)
    else:
        X1 = np.asarray([], dtype=float)
        X0 = np.zeros((0, len(donors)), dtype=float)

    # ---- Feature names / groups ------------------------------
    feature_names: List[str] = []
    row_var: List[str] = []

    for _, r in X_long.iterrows():
        var = str(r["var_name"])
        row_var.append(var)
        feature_names.append(
            f"{var}@static" if pd.isna(r["date"]) else f"{var}@{int(float(r['date']))}"
        )

    group_names = list(dict.fromkeys(row_var))

    # ---- Return (all variables explicitly localised) ---------
    return SCFormat(
        city_col=str(city_col),
        date_col=str(date_col),
        y_col=str(y_col),
        covariate_cols=covariate_cols_clean,            # FIX C3: always a list
        y_feature_years=[int(y) for y in y_feat_years],
        cov_feature_years=[int(y) for y in cov_years],  # FIX C2: local cov_years
        years=np.asarray(years, dtype=int) if len(years) else np.asarray([], dtype=int),
        donors=donors,
        treated=treated_city,
        T0=T0,
        Y1=Y1,
        Y0=Y0,
        pre_mask=np.asarray(pre_mask, dtype=bool),
        post_mask=np.asarray(post_mask, dtype=bool),
        X1=X1,
        X0=X0,
        feature_names=feature_names,
        X_long=X_long,
        row_var=row_var,
        group_names=group_names,
    )


# ---------------------------------------------------------------------------
# Sub-design builder (train/val splits)
# ---------------------------------------------------------------------------

def build_x_design_from_scformat(
    sc_format: SCFormat,
    *,
    feature_years: Sequence[int],
    include_static: bool = True,
    normalize_X: bool = True,
    normalize_method: str = "robust",
) -> XDesign:
    """Build a sub-selection of X matrices from SCFormat.X_long.

    Keeps rows whose date is in *feature_years*, and optionally static rows
    (date is NaN).  Useful for train/validation splits.
    """
    X_long = sc_format.X_long.copy()

    if X_long.empty:
        return XDesign(
            X1=np.asarray([], dtype=float),
            X0=np.zeros((0, len(sc_format.donors)), dtype=float),
            X_long=X_long,
            feature_names=[],
            row_var=[],
            group_names=[],
        )

    years_set = {int(y) for y in feature_years}
    dt = pd.to_numeric(X_long["date"], errors="coerce")
    mask_dynamic = dt.isin(list(years_set))
    mask_static = dt.isna()

    keep_mask = (mask_dynamic | mask_static) if include_static else mask_dynamic
    X_sub = X_long.loc[keep_mask].copy().reset_index(drop=True)

    unit_cols = [sc_format.treated] + list(sc_format.donors)
    X_mat = (
        X_sub[unit_cols].to_numpy(dtype=float)
        if len(X_sub)
        else np.zeros((0, len(unit_cols)), dtype=float)
    )
    X_mat = _normalize_x_matrix(
        X_mat, normalize_X=normalize_X, normalize_method=normalize_method
    )

    if X_mat.shape[0]:
        X1 = X_mat[:, 0].astype(float)
        X0 = X_mat[:, 1:].astype(float)
    else:
        X1 = np.asarray([], dtype=float)
        X0 = np.zeros((0, len(sc_format.donors)), dtype=float)

    feature_names: List[str] = []
    row_var: List[str] = []

    for _, r in X_sub.iterrows():
        var = str(r["var_name"])
        row_var.append(var)
        feature_names.append(
            f"{var}@static" if pd.isna(r["date"]) else f"{var}@{int(float(r['date']))}"
        )

    group_names = list(dict.fromkeys(row_var))

    return XDesign(
        X1=X1,
        X0=X0,
        X_long=X_sub,
        feature_names=feature_names,
        row_var=row_var,
        group_names=group_names,
    )
