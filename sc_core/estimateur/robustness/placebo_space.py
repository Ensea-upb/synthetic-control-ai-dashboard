from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Callable

import numpy as np
import pandas as pd

from .prepare import select_valid_predictor_vars
from ..core.types import EstimationResult
from ..validation.scoring import gap
from .rmspe import compute_rmspe_metrics
from .prepare import (
    build_robustness_panel_from_scformat,
    get_pre_post_periods_from_scformat,
    get_backend_outcome_and_predictors,
)


@dataclass
class PlaceboSpaceResult:
    treated_unit: str
    placebo_units: List[str]
    gaps_dict: Dict[str, np.ndarray]
    rmspe_info: Dict[str, Dict[str, Any]]
    ratio_series: pd.Series


def rebuild_df_for_treated_unit(
    df: pd.DataFrame,
    *,
    treated_col: str,
) -> pd.DataFrame:
    """
    Rebuild dataframe so that `treated_col` becomes the placebo treated unit,
    and all remaining unit columns become donor columns.

    Input expected:
    | vars | annee | ville_traite | donor1 | donor2 | ... |

    Important:
    - 'ville_traite' is the current treated column in the canonical backend panel.
    - When we pick a placebo unit, we must EXCLUDE the previous 'ville_traite'
      from the donor columns, otherwise renaming creates duplicate columns.
    """
    required = {"vars", "annee", "ville_traite"}
    if not required.issubset(df.columns):
        raise ValueError("Input dataframe must contain vars, annee, ville_traite.")

    unit_cols = [c for c in df.columns if c not in {"vars", "annee"}]
    if treated_col not in unit_cols:
        raise ValueError(f"treated_col='{treated_col}' not found among unit columns.")

    donor_cols = [c for c in unit_cols if c not in {treated_col, "ville_traite"}]

    out = df[["vars", "annee", treated_col] + donor_cols].copy()
    out = out.rename(columns={treated_col: "ville_traite"})
    return out


def run_placebo_space(
    *,
    df: pd.DataFrame,
    outcome_var: str,
    predictor_vars,
    pre_periods,
    post_periods,
    estimator_fn: Callable[..., EstimationResult],
    estimator_kwargs: Dict[str, Any] | None = None,
    original_treated_name: str = "ville_traite",
) -> PlaceboSpaceResult:
    """
    Run placebo-space robustness on a canonical backend dataframe.

    Parameters
    ----------
    df :
        Canonical robustness panel with columns:
        ['vars', 'annee', 'ville_traite', donor_1, donor_2, ...]
    outcome_var :
        Backend outcome variable name, typically 'y'.
    predictor_vars :
        Predictor names admissible on the full pre-treatment period.
    pre_periods, post_periods :
        Period split.
    estimator_fn :
        Estimator over dataframe returning EstimationResult.
    """
    estimator_kwargs = estimator_kwargs or {}

    unit_cols = [c for c in df.columns if c not in {"vars", "annee"}]
    if original_treated_name not in unit_cols:
        raise ValueError(
            f"original_treated_name='{original_treated_name}' not found in unit columns."
        )

    placebo_units = [u for u in unit_cols if u != original_treated_name]

    all_periods = list(pre_periods) + list(post_periods)
    df_all = df.loc[df["annee"].isin(all_periods)].copy()

    gaps_dict: Dict[str, np.ndarray] = {}
    rmspe_info: Dict[str, Dict[str, Any]] = {}

    pre_len = len(pre_periods)
    total_len = len(all_periods)
    pre_idx = np.arange(0, pre_len, dtype=int)
    post_idx = np.arange(pre_len, total_len, dtype=int)

    for placebo_unit in placebo_units:
        df_placebo = rebuild_df_for_treated_unit(df_all, treated_col=placebo_unit)
        # Si predictor_vars is None → mode outcome-only, pas de filtre
        if predictor_vars is None:
            predictor_vars_used = None
        else:
            predictor_vars_used = select_valid_predictor_vars(
                df_placebo,
                predictor_vars=predictor_vars,
                pre_periods=pre_periods,
            )

        res = estimator_fn(
            df=df_placebo,
            outcome_var=outcome_var,
            predictor_vars=predictor_vars_used,
            pre_periods=pre_periods,
            **estimator_kwargs,
        )

        df_y = df_placebo.loc[df_placebo["vars"].astype(str) == str(outcome_var)].copy()
        df_y = df_y.sort_values("annee")

        y_true = df_y["ville_traite"].to_numpy(dtype=float)
        y_synth = df_y[res.donor_names].to_numpy(dtype=float) @ res.w

        rmspe_res = compute_rmspe_metrics(
            y_true=y_true,
            y_synth=y_synth,
            pre_idx=pre_idx,
            post_idx=post_idx,
        )

        gaps_dict[placebo_unit] = rmspe_res.gaps
        rmspe_info[placebo_unit] = {
            "pre_rmspe": rmspe_res.pre_rmspe,
            "post_rmspe": rmspe_res.post_rmspe,
            "ratio": rmspe_res.ratio,
            "result": res,
        }

    ratio_series = pd.Series(
        {unit: info["ratio"] for unit, info in rmspe_info.items()},
        name="rmspe_ratio",
    ).sort_values(ascending=False)

    return PlaceboSpaceResult(
        treated_unit=original_treated_name,
        placebo_units=placebo_units,
        gaps_dict=gaps_dict,
        rmspe_info=rmspe_info,
        ratio_series=ratio_series,
    )


def run_placebo_space_from_scformat(
    *,
    sc_format,
    estimator_fn,
    estimator_kwargs: Dict[str, Any] | None = None,
) -> PlaceboSpaceResult:
    """
    Backend-native SCFormat wrapper.

    It reconstructs the canonical robustness panel, derives backend outcome/predictors,
    and runs placebo-space using the canonical treated column 'ville_traite'.
    """
    estimator_kwargs = estimator_kwargs or {}

    panel_df = build_robustness_panel_from_scformat(sc_format)
    outcome_var, predictor_vars = get_backend_outcome_and_predictors(sc_format, panel_df)
    pre_periods, post_periods, _ = get_pre_post_periods_from_scformat(sc_format)

    return run_placebo_space(
        df=panel_df,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        pre_periods=pre_periods,
        post_periods=post_periods,
        estimator_fn=estimator_fn,
        estimator_kwargs=estimator_kwargs,
        original_treated_name="ville_traite",
    )