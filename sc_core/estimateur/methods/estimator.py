from __future__ import annotations

import pandas as pd
import numpy as np

from ..core.types import EstimationResult
from ..utils.data_prep import build_prepared_matrices
from ..validation.time_split import split_pre_periods_last_k
from .inner.slsqp import SLSQPInnerSolver
from .outer.random_search import fit_via_random_search
from .outer.bilevel import fit_via_bilevel_scipy
from .outer.trainval import fit_via_trainval_random_search
from sc_core.data_management import build_x_design_from_scformat


def fit_xv_random_search_from_df(
    *,
    df: pd.DataFrame,
    outcome_var: str,
    predictor_vars,
    pre_periods,
    n_iter: int = 200,
    seed: int = 123,
    callback=None,
) -> EstimationResult:
    prepared = build_prepared_matrices(
        df=df,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        pre_periods=pre_periods,
    )

    inner_solver = SLSQPInnerSolver()

    return fit_via_random_search(
        X1=prepared.X1,
        X0=prepared.X0,
        Y1_pre=prepared.Y1_pre,
        Y0_pre=prepared.Y0_pre,
        row_var=prepared.row_var,
        group_names=prepared.group_names,
        donor_names=prepared.donor_names,
        inner_solver=inner_solver,
        n_iter=n_iter,
        seed=seed,
        callback=callback,
    )


def fit_xv_bilevel_from_df(
    *,
    df: pd.DataFrame,
    outcome_var: str,
    predictor_vars,
    pre_periods,
    n_restarts: int = 5,
    maxiter: int = 200,
    seed: int = 123,
    method: str = "Powell",
    callback=None,
) -> EstimationResult:
    prepared = build_prepared_matrices(
        df=df,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        pre_periods=pre_periods,
    )

    inner_solver = SLSQPInnerSolver()

    return fit_via_bilevel_scipy(
        X1=prepared.X1,
        X0=prepared.X0,
        Y1_pre=prepared.Y1_pre,
        Y0_pre=prepared.Y0_pre,
        row_var=prepared.row_var,
        group_names=prepared.group_names,
        donor_names=prepared.donor_names,
        inner_solver=inner_solver,
        n_restarts=n_restarts,
        maxiter=maxiter,
        seed=seed,
        method=method,
        callback=callback,
    )


def fit_xv_trainval_from_df(
    *,
    df: pd.DataFrame,
    outcome_var: str,
    predictor_vars,
    pre_periods,
    val_last_k: int = 3,
    n_iter: int = 300,
    seed: int = 123,
    callback=None,
) -> EstimationResult:
    prepared = build_prepared_matrices(
        df=df,
        outcome_var=outcome_var,
        predictor_vars=predictor_vars,
        pre_periods=pre_periods,
    )

    split = split_pre_periods_last_k(prepared.pre_periods, val_last_k=val_last_k)

    row_time_arr = np.asarray(prepared.row_time, dtype=object)
    train_feature_mask = np.array([t in set(split.train_periods) for t in row_time_arr], dtype=bool)

    X1_train = prepared.X1[train_feature_mask]
    X0_train = prepared.X0[train_feature_mask, :]
    row_var_train = [v for v, keep in zip(prepared.row_var, train_feature_mask) if keep]

    inner_solver = SLSQPInnerSolver()

    return fit_via_trainval_random_search(
        X1_train=X1_train,
        X0_train=X0_train,
        row_var_train=row_var_train,
        X1_full=prepared.X1,
        X0_full=prepared.X0,
        row_var_full=prepared.row_var,
        group_names=prepared.group_names,
        donor_names=prepared.donor_names,
        Y1_pre=prepared.Y1_pre,
        Y0_pre=prepared.Y0_pre,
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        inner_solver=inner_solver,
        n_iter=n_iter,
        seed=seed,
        callback=callback,
    )


# =========================================================
# New facade: from SCFormat
# =========================================================

def fit_xv_random_search_from_scformat(
    *,
    sc_format,
    n_iter: int = 200,
    seed: int = 123,
    callback=None,
) -> EstimationResult:
    inner_solver = SLSQPInnerSolver()

    return fit_via_random_search(
        X1=sc_format.X1,
        X0=sc_format.X0,
        Y1_pre=sc_format.Y1[sc_format.pre_mask],
        Y0_pre=sc_format.Y0[sc_format.pre_mask, :],
        row_var=sc_format.row_var,
        group_names=sc_format.group_names,
        donor_names=sc_format.donors,
        inner_solver=inner_solver,
        n_iter=n_iter,
        seed=seed,
        callback=callback,
    )


def fit_xv_bilevel_from_scformat(
    *,
    sc_format,
    n_restarts: int = 5,
    maxiter: int = 200,
    seed: int = 123,
    method: str = "Powell",
    callback=None,
) -> EstimationResult:
    inner_solver = SLSQPInnerSolver()

    return fit_via_bilevel_scipy(
        X1=sc_format.X1,
        X0=sc_format.X0,
        Y1_pre=sc_format.Y1[sc_format.pre_mask],
        Y0_pre=sc_format.Y0[sc_format.pre_mask, :],
        row_var=sc_format.row_var,
        group_names=sc_format.group_names,
        donor_names=sc_format.donors,
        inner_solver=inner_solver,
        n_restarts=n_restarts,
        maxiter=maxiter,
        seed=seed,
        method=method,
        callback=callback,
    )


def fit_xv_trainval_from_scformat(
    *,
    sc_format,
    val_last_k: int = 3,
    n_iter: int = 300,
    seed: int = 123,
    callback=None,
) -> EstimationResult:
    pre_periods = sc_format.years[sc_format.pre_mask].tolist()
    split = split_pre_periods_last_k(pre_periods, val_last_k=val_last_k)

    x_train = build_x_design_from_scformat(
        sc_format,
        feature_years=split.train_periods,
        include_static=True,
        normalize_X=True,
        normalize_method="robust",
    )

    inner_solver = SLSQPInnerSolver()

    return fit_via_trainval_random_search(
        X1_train=x_train.X1,
        X0_train=x_train.X0,
        row_var_train=x_train.row_var,
        X1_full=sc_format.X1,
        X0_full=sc_format.X0,
        row_var_full=sc_format.row_var,
        group_names=sc_format.group_names,
        donor_names=sc_format.donors,
        Y1_pre=sc_format.Y1[sc_format.pre_mask],
        Y0_pre=sc_format.Y0[sc_format.pre_mask, :],
        train_idx=split.train_idx,
        val_idx=split.val_idx,
        inner_solver=inner_solver,
        n_iter=n_iter,
        seed=seed,
        callback=callback,
    )