from __future__ import annotations

from typing import Dict, Any, Sequence, Optional

import numpy as np
import pandas as pd


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
    s = s.astype(str).str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def is_dynamic_variable(
    df: pd.DataFrame,
    *,
    city_col: str,
    date_col: str,
    value_col: str,
) -> bool:
    """
    Une variable est dynamique si au moins une ville possède
    au moins 2 dates distinctes observées.
    """
    tmp = df[[city_col, date_col, value_col]].copy()
    tmp[value_col] = _coerce_numeric_series(tmp[value_col])

    tmp = tmp[np.isfinite(tmp[value_col])]

    if tmp.empty:
        return False

    ny = tmp.groupby(city_col)[date_col].nunique(dropna=True)
    return bool((ny >= 2).any())


def build_dynamic_payload(
    df: pd.DataFrame,
    *,
    variable: str,
    treated_unit: str,
    control_units: Sequence[str],
    date_col: str = "date",
    city_col: str = "ville",
    max_missing_by_city: int = 2,
    intervention_time: Optional[int] = None,
    show_envelope: bool = True,
) -> Dict[str, Any]:

    units = [treated_unit] + [u for u in control_units if u != treated_unit]

    tmp = df.loc[df[city_col].isin(units), [city_col, date_col, variable]].copy()
    tmp[variable] = _coerce_numeric_series(tmp[variable])

    wide = (
        tmp.pivot_table(index=date_col, columns=city_col, values=variable, aggfunc="mean")
        .sort_index()
    )

    to_drop = [
        c for c in wide.columns
        if wide[c].isna().sum() >= max_missing_by_city
    ]

    wide = wide.drop(columns=to_drop, errors="ignore")

    controls = [c for c in control_units if c in wide.columns and c != treated_unit]

    return dict(
        variable_type="dynamic",
        df_wide=wide,
        variable_name=variable,
        treated_unit=treated_unit,
        control_units=controls,
        intervention_time=intervention_time,
        show_envelope=bool(show_envelope),
    )


def build_static_payload(
    df: pd.DataFrame,
    *,
    variable: str,
    treated_unit: str,
    control_units: Sequence[str],
    city_col: str = "ville",
) -> Dict[str, Any]:

    units = [treated_unit] + [u for u in control_units if u != treated_unit]

    tmp = df.loc[df[city_col].isin(units), [city_col, variable]].copy()
    tmp[variable] = _coerce_numeric_series(tmp[variable])

    tmp = tmp.dropna()

    tmp = (
        tmp.groupby(city_col)[variable]
        .first()
        .reset_index()
        .rename(columns={variable: "value"})
    )

    return dict(
        variable_type="static",
        df_static=tmp,
        variable_name=variable,
        treated_unit=treated_unit,
    )


def build_exploration_payload(
    df: pd.DataFrame,
    *,
    variable: str,
    treated_unit: str,
    control_units: Sequence[str],
    city_col: str = "ville",
    date_col: str = "date",
    max_missing_by_city: int = 2,
    intervention_time: Optional[int] = None,
    show_envelope: bool = True,
):

    dynamic = is_dynamic_variable(
        df,
        city_col=city_col,
        date_col=date_col,
        value_col=variable,
    )

    if dynamic:
        return build_dynamic_payload(
            df,
            variable=variable,
            treated_unit=treated_unit,
            control_units=control_units,
            city_col=city_col,
            date_col=date_col,
            max_missing_by_city=max_missing_by_city,
            intervention_time=intervention_time,
            show_envelope=show_envelope,
        )

    return build_static_payload(
        df,
        variable=variable,
        treated_unit=treated_unit,
        control_units=control_units,
        city_col=city_col,
    )

def build_exploration_comment_context(payload: dict) -> dict:
    variable_type = payload.get("variable_type")
    variable_name = payload.get("variable_name")
    treated_unit = payload.get("treated_unit")

    if variable_type == "dynamic":
        df_wide = payload["df_wide"]
        controls = payload.get("control_units", [])

        treated_series = df_wide[treated_unit].dropna()
        control_df = df_wide[controls].copy() if controls else pd.DataFrame(index=df_wide.index)

        context = {
            "variable": variable_name,
            "variable_type": "dynamic",
            "treated_unit": treated_unit,
            "control_units": controls,
            "n_periods": int(df_wide.shape[0]),
            "n_controls": int(len(controls)),
            "treated_non_null_count": int(treated_series.shape[0]),
            "treated_first_value": float(treated_series.iloc[0]) if not treated_series.empty else None,
            "treated_last_value": float(treated_series.iloc[-1]) if not treated_series.empty else None,
            "intervention_time": payload.get("intervention_time"),
            "show_envelope": bool(payload.get("show_envelope", True)),
        }

        if not control_df.empty:
            last_row = control_df.iloc[-1].dropna()
            context["control_last_min"] = float(last_row.min()) if not last_row.empty else None
            context["control_last_max"] = float(last_row.max()) if not last_row.empty else None
        else:
            context["control_last_min"] = None
            context["control_last_max"] = None

        return context

    df_static = payload["df_static"]
    value_col = payload.get("value_col", "value")
    unit_col = payload.get("unit_col", "ville")

    treated_values = df_static.loc[df_static[unit_col].astype(str) == str(treated_unit), value_col]

    return {
        "variable": variable_name,
        "variable_type": "static",
        "treated_unit": treated_unit,
        "n_units": int(df_static.shape[0]),
        "treated_value": float(treated_values.iloc[0]) if not treated_values.empty else None,
        "max_value": float(df_static[value_col].max()) if not df_static.empty else None,
        "min_value": float(df_static[value_col].min()) if not df_static.empty else None,
    }