from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Callable, List

import numpy as np
import pandas as pd

from .prepare import select_valid_predictor_vars
from ..core.types import EstimationResult
from ..validation.scoring import gap
from .prepare import (
    build_robustness_panel_from_scformat,
    get_pre_post_periods_from_scformat,
    get_backend_outcome_and_predictors,
)


@dataclass
class BackdatingRun:
    pseudo_t0: Any
    pre_periods_used: List[Any]
    result: EstimationResult
    y_true: np.ndarray
    y_synth: np.ndarray
    gaps: np.ndarray


@dataclass
class BackdatingResult:
    runs: List[BackdatingRun]


def run_backdating(
    *,
    df: pd.DataFrame,
    outcome_var: str,
    predictor_vars,
    all_periods,
    pseudo_t0_list,
    estimator_fn: Callable[..., EstimationResult],
    estimator_kwargs: Dict[str, Any] | None = None,
) -> BackdatingResult:
    estimator_kwargs = estimator_kwargs or {}
    all_periods = list(all_periods)

    runs: List[BackdatingRun] = []

    for pseudo_t0 in pseudo_t0_list:
        pre_periods_used = [p for p in all_periods if p < pseudo_t0]
        if len(pre_periods_used) < 2:
            continue
        if predictor_vars is None:
            predictor_vars_used = None
        else:
            predictor_vars_used = select_valid_predictor_vars(
                df,
                predictor_vars=predictor_vars,
                pre_periods=pre_periods_used,
            )

        res = estimator_fn(
            df=df,
            outcome_var=outcome_var,
            predictor_vars=predictor_vars_used,
            pre_periods=pre_periods_used,
            **estimator_kwargs,
        )

        df_y = df.loc[df["vars"].astype(str) == str(outcome_var)].copy()
        df_y = df_y.loc[df_y["annee"].isin(all_periods)].sort_values("annee")

        y_true = df_y["ville_traite"].to_numpy(dtype=float)
        y_synth = df_y[res.donor_names].to_numpy(dtype=float) @ res.w
        g = gap(y_true, y_synth)

        runs.append(
            BackdatingRun(
                pseudo_t0=pseudo_t0,
                pre_periods_used=pre_periods_used,
                result=res,
                y_true=y_true,
                y_synth=y_synth,
                gaps=g,
            )
        )

    return BackdatingResult(runs=runs)


def run_backdating_from_scformat(
    *,
    sc_format,
    estimator_fn,
    estimator_kwargs: Dict[str, Any] | None = None,
    pseudo_t0_list=None,
) -> BackdatingResult:
    """
    Wrapper backend natif SCFormat.

    Il reconstruit le panel robustesse canonique, dérive l'outcome backend,
    filtre les prédicteurs admissibles et choisit des pseudo dates si besoin.
    """
    estimator_kwargs = estimator_kwargs or {}

    panel_df = build_robustness_panel_from_scformat(sc_format)
    outcome_var, predictor_vars = get_backend_outcome_and_predictors(sc_format, panel_df)
    _, _, all_periods = get_pre_post_periods_from_scformat(sc_format)

    if pseudo_t0_list is None:
        # FIX C4: restrict pseudo-T0 to pre-treatment years only.
        # We exclude the first pre-year (need >= 2 periods before pseudo_t0)
        # and years >= T0 (those are post-treatment, not valid pseudo-T0s).
        pre_years, _, _ = get_pre_post_periods_from_scformat(sc_format)
        # Drop the first pre-year: pseudo_t0 must have >= 2 periods before it
        pseudo_t0_list = pre_years[1:] if len(pre_years) >= 2 else []
    else:
        pseudo_t0_list = list(pseudo_t0_list)

    return run_backdating(
        df=panel_df,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        all_periods=all_periods,
        pseudo_t0_list=pseudo_t0_list,
        estimator_fn=estimator_fn,
        estimator_kwargs=estimator_kwargs,
    )