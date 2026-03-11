

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd


@dataclass
class FitSummaryData:
    time_index: np.ndarray
    y_treated: np.ndarray
    y_synth: np.ndarray
    gap: np.ndarray
    cumulative_gap: np.ndarray
    unit_weights: pd.Series
    covariate_weights: pd.Series
    objective_history: List[float]
    T0: int


def build_synthetic_series(
    *,
    Y0: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    Y0 = np.asarray(Y0, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)

    if Y0.ndim != 2:
        raise ValueError("Y0 must be a 2D array of shape (T, J).")
    if Y0.shape[1] != w.shape[0]:
        raise ValueError(
            f"Dimension mismatch: Y0 has {Y0.shape[1]} donor columns, weights has length {w.shape[0]}."
        )

    return Y0 @ w


def build_gap(
    *,
    y_treated: np.ndarray,
    y_synth: np.ndarray,
) -> np.ndarray:
    y_treated = np.asarray(y_treated, dtype=float).reshape(-1)
    y_synth = np.asarray(y_synth, dtype=float).reshape(-1)

    if y_treated.shape[0] != y_synth.shape[0]:
        raise ValueError("y_treated and y_synth must have the same length.")

    return y_treated - y_synth


def build_cumulative_gap(gap: np.ndarray) -> np.ndarray:
    g = np.asarray(gap, dtype=float).reshape(-1)
    return np.cumsum(g)


def build_unit_weights_series(
    *,
    donor_names,
    weights: np.ndarray,
    drop_zeros: bool = False,
    tol: float = 1e-12,
) -> pd.Series:
    s = pd.Series(np.asarray(weights, dtype=float), index=list(donor_names), name="unit_weight")
    if drop_zeros:
        s = s[s.abs() > tol]
    return s.sort_values(ascending=False)


def build_covariate_weights_series(
    *,
    group_names,
    Vvar: np.ndarray,
    drop_zeros: bool = False,
    tol: float = 1e-12,
) -> pd.Series:
    s = pd.Series(np.asarray(Vvar, dtype=float), index=list(group_names), name="covariate_weight")
    if drop_zeros:
        s = s[s.abs() > tol]
    return s.sort_values(ascending=False)


def build_fit_summary_data(
    *,
    sc_format,
    estimation_result,
    drop_zero_weights: bool = False,
    zero_tol: float = 1e-12,
) -> FitSummaryData:
    y_synth = build_synthetic_series(
        Y0=sc_format.Y0,
        weights=estimation_result.w,
    )

    gap = build_gap(
        y_treated=sc_format.Y1,
        y_synth=y_synth,
    )

    cumulative_gap = build_cumulative_gap(gap)

    unit_weights = build_unit_weights_series(
        donor_names=estimation_result.donor_names,
        weights=estimation_result.w,
        drop_zeros=drop_zero_weights,
        tol=zero_tol,
    )

    covariate_weights = build_covariate_weights_series(
        group_names=estimation_result.group_names,
        Vvar=estimation_result.Vvar,
        drop_zeros=drop_zero_weights,
        tol=zero_tol,
    )

    return FitSummaryData(
        time_index=np.asarray(sc_format.years),
        y_treated=np.asarray(sc_format.Y1, dtype=float),
        y_synth=np.asarray(y_synth, dtype=float),
        gap=np.asarray(gap, dtype=float),
        cumulative_gap=np.asarray(cumulative_gap, dtype=float),
        unit_weights=unit_weights,
        covariate_weights=covariate_weights,
        objective_history=list(estimation_result.objective_history),
        T0=int(sc_format.T0),
    )


def build_result_dict_for_ui(
    *,
    sc_format,
    estimation_result,
    drop_zero_weights: bool = False,
    zero_tol: float = 1e-12,
) -> Dict[str, Any]:
    fit = build_fit_summary_data(
        sc_format=sc_format,
        estimation_result=estimation_result,
        drop_zero_weights=drop_zero_weights,
        zero_tol=zero_tol,
    )

    return {
        "time_index": fit.time_index,
        "y_treated": fit.y_treated,
        "y_synth": fit.y_synth,
        "gap": fit.gap,
        "cumulative_gap": fit.cumulative_gap,
        "unit_weights": fit.unit_weights,
        "covariate_weights": fit.covariate_weights,
        "objective_history": fit.objective_history,
        "T0": fit.T0,
    }