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
class LeaveOneOutResult:
    base_result: EstimationResult
    dropped_donors: List[str]
    results_by_donor: Dict[str, EstimationResult]
    gaps_by_donor: Dict[str, np.ndarray]


def drop_donor_from_df(
    df: pd.DataFrame,
    donor_name: str,
) -> pd.DataFrame:
    if donor_name not in df.columns:
        raise ValueError(f"donor_name='{donor_name}' not found in dataframe columns.")

    return df.drop(columns=[donor_name]).copy()


def run_leave_one_out(
    *,
    df: pd.DataFrame,
    base_result: EstimationResult,
    outcome_var: str,
    predictor_vars,
    pre_periods,
    all_periods,
    estimator_fn: Callable[..., EstimationResult],
    estimator_kwargs: Dict[str, Any] | None = None,
    active_only: bool = True,
    active_tol: float = 1e-10,
) -> LeaveOneOutResult:
    """Run leave-one-out sensitivity analysis.

    FIX C5: guard against |A| == 1 — if dropping any donor would leave
    0 donors, return an empty result rather than crashing.
    """
    estimator_kwargs = estimator_kwargs or {}

    total_donors = len(base_result.donor_names)

    # FIX C5: need at least 2 donors to run LOO (drop one, keep ≥ 1)
    if total_donors < 2:
        return LeaveOneOutResult(
            base_result=base_result,
            dropped_donors=[],
            results_by_donor={},
            gaps_by_donor={},
        )

    donor_pairs = list(zip(base_result.donor_names, base_result.w))

    if active_only:
        dropped_donors = [name for name, weight in donor_pairs if float(weight) > active_tol]
    else:
        dropped_donors = [name for name, _ in donor_pairs]

    # After dropping one donor, we need at least 1 remaining
    # (edge-case: all donors are active and total == 1 already handled above)
    if len(dropped_donors) == total_donors and total_donors < 2:
        dropped_donors = []

    results_by_donor: Dict[str, EstimationResult] = {}
    gaps_by_donor: Dict[str, np.ndarray] = {}

    df_all = df.loc[df["annee"].isin(all_periods)].copy()

    for donor in dropped_donors:
        df_drop = drop_donor_from_df(df_all, donor)
        if predictor_vars is None:
            predictor_vars_used = None
        else:
            predictor_vars_used = select_valid_predictor_vars(
                df_drop,
                predictor_vars=predictor_vars,
                pre_periods=pre_periods,
            )
        res = estimator_fn(
            df=df_drop,
            outcome_var=outcome_var,
            predictor_vars=predictor_vars_used,
            pre_periods=pre_periods,
            **estimator_kwargs,
        )

        df_y = df_drop.loc[df_drop["vars"].astype(str) == str(outcome_var)].copy()
        df_y = df_y.sort_values("annee")

        y_true = df_y["ville_traite"].to_numpy(dtype=float)
        y_synth = df_y[res.donor_names].to_numpy(dtype=float) @ res.w

        results_by_donor[donor] = res
        gaps_by_donor[donor] = gap(y_true, y_synth)

    return LeaveOneOutResult(
        base_result=base_result,
        dropped_donors=dropped_donors,
        results_by_donor=results_by_donor,
        gaps_by_donor=gaps_by_donor,
    )


def run_leave_one_out_from_scformat(
    *,
    sc_format,
    base_result: EstimationResult,
    estimator_fn,
    estimator_kwargs: Dict[str, Any] | None = None,
    active_only: bool = True,
    active_tol: float = 1e-10,
) -> LeaveOneOutResult:
    """
    Wrapper backend natif SCFormat.

    Il reconstruit lui-même le panel robustesse canonique, dérive l'outcome backend
    et filtre les prédicteurs admissibles sur le pré-traitement.
    """
    estimator_kwargs = estimator_kwargs or {}

    panel_df = build_robustness_panel_from_scformat(sc_format)
    outcome_var, predictor_vars = get_backend_outcome_and_predictors(sc_format, panel_df)
    pre_periods, _, all_periods = get_pre_post_periods_from_scformat(sc_format)

    return run_leave_one_out(
        df=panel_df,
        base_result=base_result,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        pre_periods=pre_periods,
        all_periods=all_periods,
        estimator_fn=estimator_fn,
        estimator_kwargs=estimator_kwargs,
        active_only=active_only,
        active_tol=active_tol,
    )