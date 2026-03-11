from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sc_core.estimateur.robustness.placebo_space import run_placebo_space_from_scformat
from sc_core.estimateur.robustness.leave_one_out import run_leave_one_out_from_scformat
from sc_core.estimateur.robustness.backdating import run_backdating_from_scformat
from sc_core.estimateur.robustness.rmspe import (
    compute_rmspe_metrics,
    compute_rmspe_ratio_series,
)
from sc_core.estimateur.methods.estimator import (
    fit_xv_random_search_from_df,
    fit_xv_bilevel_from_df,
    fit_xv_trainval_from_df,
)

try:
    from sc_core.plotting.manager import PlotManager
except Exception:
    PlotManager = None


# =========================================================
# Public output container
# =========================================================

@dataclass
class RobustnessRunOutput:
    placebo_result: Any = None
    leave_one_out_result: Any = None
    backdating_result: Any = None
    rmspe_metrics: Optional[Dict[str, Any]] = None
    rmspe_ratio_series: Any = None


# =========================================================
# Generic helpers
# =========================================================

def _filter_kwargs_for_callable(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _call_backend(func, **kwargs):
    return func(**_filter_kwargs_for_callable(func, kwargs))


def _resolve_df_estimator(method_name: str):
    method_name = str(method_name).strip().lower()
    if method_name == "random_search":
        return fit_xv_random_search_from_df
    if method_name == "bilevel":
        return fit_xv_bilevel_from_df
    if method_name == "trainval":
        return fit_xv_trainval_from_df
    raise ValueError(f"Unknown method_name: {method_name}")


def _extract_treated_unit(sc_format: Any) -> Optional[str]:
    treated = getattr(sc_format, "treated", None)
    if treated is None:
        treated = getattr(sc_format, "treated_unit", None)
    if treated is None:
        return None
    treated = str(treated).strip()
    return treated or None


def _extract_time_index(sc_format: Any) -> np.ndarray:
    """Extract time_index (years) from sc_format."""
    years = getattr(sc_format, "years", None)
    if years is None:
        return np.array([])
    return np.asarray(years)


def _extract_T0(sc_format: Any) -> Optional[Any]:
    return getattr(sc_format, "T0", None)


def _compute_base_gap(sc_format: Any, estimation_result: Any) -> Optional[np.ndarray]:
    """Compute the original treated gap from sc_format + estimation_result."""
    try:
        Y1 = np.asarray(sc_format.Y1, dtype=float).reshape(-1)
        Y0 = np.asarray(sc_format.Y0, dtype=float)
        w = np.asarray(estimation_result.w, dtype=float).reshape(-1)
        y_synth = Y0 @ w
        return Y1 - y_synth
    except Exception:
        return None


# =========================================================
# Robustness runners
# =========================================================

def run_placebo_space_service(
    *,
    sc_format: Any,
    method_name: str,
    estimation_config: Dict[str, Any],
) -> Any:
    estimator_fn = _resolve_df_estimator(method_name)
    estimator_kwargs = dict(estimation_config)
    estimator_kwargs.pop("method_name", None)
    estimator_kwargs = _filter_kwargs_for_callable(estimator_fn, estimator_kwargs)
    return _call_backend(
        run_placebo_space_from_scformat,
        sc_format=sc_format,
        estimator_fn=estimator_fn,
        estimator_kwargs=estimator_kwargs,
    )


def run_leave_one_out_service(
    *,
    sc_format: Any,
    estimation_result: Any,
    method_name: str,
    estimation_config: Dict[str, Any],
) -> Any:
    estimator_fn = _resolve_df_estimator(method_name)
    estimator_kwargs = dict(estimation_config)
    estimator_kwargs.pop("method_name", None)
    estimator_kwargs = _filter_kwargs_for_callable(estimator_fn, estimator_kwargs)

    if estimation_result is None:
        raise ValueError("Leave-one-out nécessite estimation_result comme base_result.")

    return _call_backend(
        run_leave_one_out_from_scformat,
        sc_format=sc_format,
        base_result=estimation_result,
        estimator_fn=estimator_fn,
        estimator_kwargs=estimator_kwargs,
    )


def run_backdating_service(
    *,
    sc_format: Any,
    method_name: str,
    estimation_config: Dict[str, Any],
) -> Any:
    estimator_fn = _resolve_df_estimator(method_name)
    estimator_kwargs = dict(estimation_config)
    estimator_kwargs.pop("method_name", None)
    estimator_kwargs = _filter_kwargs_for_callable(estimator_fn, estimator_kwargs)
    return _call_backend(
        run_backdating_from_scformat,
        sc_format=sc_format,
        estimator_fn=estimator_fn,
        estimator_kwargs=estimator_kwargs,
    )


def run_rmspe_service(
    *,
    placebo_result: Any,
    estimation_result: Any,
    sc_format: Any = None,
    fit_summary_data: Any = None,
    treated_unit: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Any]:
    """Compute RMSPE metrics and inject the treated unit's own ratio into the series.

    The treated unit ratio is computed from fit_summary_data (pre/post gap) so that
    the ranking chart includes the real treated unit alongside the placebo units,
    enabling proper p-value calculation.
    """
    metrics = None
    ratio_series = None

    if placebo_result is None:
        return metrics, ratio_series

    # PlaceboSpaceResult carries ratio_series and rmspe_info
    ratio_series = getattr(placebo_result, "ratio_series", None)
    rmspe_info   = getattr(placebo_result, "rmspe_info", None)

    if rmspe_info is not None:
        metrics = rmspe_info
        if ratio_series is None:
            try:
                ratio_series = pd.Series(
                    {u: float(info["ratio"]) for u, info in rmspe_info.items()},
                    name="rmspe_ratio",
                ).sort_values(ascending=False)
            except Exception:
                ratio_series = None
    elif getattr(placebo_result, "results_by_unit", None) is not None:
        metrics = placebo_result.results_by_unit
        try:
            ratio_series = compute_rmspe_ratio_series(metrics)
        except Exception:
            ratio_series = None

    # ------------------------------------------------------------------
    # Inject treated unit's own RMSPE ratio into the series
    # ------------------------------------------------------------------
    if ratio_series is not None and fit_summary_data is not None and sc_format is not None:
        try:
            treated_label = str(treated_unit or getattr(sc_format, "treated", "traité"))
            gap_arr   = np.asarray(fit_summary_data.gap, dtype=float)
            time_idx  = np.asarray(sc_format.years)
            T0        = int(sc_format.T0)
            pre_mask  = time_idx < T0
            post_mask = time_idx >= T0

            pre_rmspe  = float(np.sqrt(np.mean(gap_arr[pre_mask]  ** 2))) if pre_mask.any()  else 0.0
            post_rmspe = float(np.sqrt(np.mean(gap_arr[post_mask] ** 2))) if post_mask.any() else 0.0

            if pre_rmspe > 0:
                treated_ratio = post_rmspe / pre_rmspe
            elif post_rmspe > 0:
                treated_ratio = float("inf")
            else:
                treated_ratio = 1.0

            # Add treated to the series under its real name
            extra = pd.Series({treated_label: treated_ratio}, name="rmspe_ratio")
            ratio_series = pd.concat([ratio_series, extra]).sort_values(ascending=False)
            ratio_series = ratio_series[~ratio_series.index.duplicated(keep="first")]
        except Exception:
            pass  # Non-critical: ratio_series stays as-is

    return metrics, ratio_series


# =========================================================
# Plot builders — CORRECTED
# =========================================================

def build_placebo_figure(
    placebo_result: Any,
    sc_format: Any = None,
    estimation_result: Any = None,
):
    """
    Build placebo gaps figure.

    Requires sc_format to extract time_index and T0.
    estimation_result is used to compute the original treated unit gap.
    """
    if PlotManager is None:
        raise RuntimeError("PlotManager not available.")

    pm = PlotManager()

    time_index = _extract_time_index(sc_format) if sc_format is not None else None
    T0 = _extract_T0(sc_format) if sc_format is not None else None
    treated_unit = _extract_treated_unit(sc_format) if sc_format is not None else "traité"

    base_gap = None
    if sc_format is not None and estimation_result is not None:
        base_gap = _compute_base_gap(sc_format, estimation_result)

    # Extract gaps_dict from PlaceboSpaceResult
    gaps_dict = getattr(placebo_result, "gaps_dict", None)
    if gaps_dict is None:
        raise RuntimeError("placebo_result.gaps_dict not found.")

    if time_index is None or len(time_index) == 0:
        raise RuntimeError("time_index is required to build placebo figure.")

    # PlaceboSpaceResult uses all_periods (pre + post) — ensure time_index matches
    # gaps_dict values have length = len(all_periods) from placebo_space
    # sc_format.years = all periods in order
    return pm.placebo_gaps(
        time_index=time_index,
        gaps_dict=gaps_dict,
        treated_unit=treated_unit,
        base_gap=base_gap,
        T0=T0,
    )


def build_rmspe_distribution_figure(rmspe_metrics: Any):
    """Build RMSPE pre/post distribution figure."""
    if PlotManager is None:
        raise RuntimeError("PlotManager not available.")
    pm = PlotManager()
    return pm.rmspe_distribution(rmspe_metrics)


def build_rmspe_ratio_figure(ratio_series: Any, treated_unit: Optional[str] = None):
    """Build RMSPE ratio ranking figure."""
    if PlotManager is None:
        raise RuntimeError("PlotManager not available.")

    if not isinstance(ratio_series, pd.Series):
        # Try to convert from dict
        if isinstance(ratio_series, dict):
            ratio_series = pd.Series(
                {u: float(v) for u, v in ratio_series.items()},
                name="rmspe_ratio",
            )
        else:
            raise RuntimeError(f"ratio_series must be pd.Series, got {type(ratio_series)}")

    pm = PlotManager()
    return pm.rmspe_ratio(ratio_series, treated=treated_unit)


def build_leave_one_out_figure(
    leave_one_out_result: Any,
    sc_format: Any = None,
    estimation_result: Any = None,
):
    """Build leave-one-out gap figure."""
    if PlotManager is None:
        raise RuntimeError("PlotManager not available.")

    time_index = _extract_time_index(sc_format) if sc_format is not None else None
    T0 = _extract_T0(sc_format) if sc_format is not None else None

    base_gap = None
    if sc_format is not None and estimation_result is not None:
        base_gap = _compute_base_gap(sc_format, estimation_result)

    if base_gap is None:
        # Try to get from base_result inside LOO
        base_res = getattr(leave_one_out_result, "base_result", None)
        if base_res is not None and sc_format is not None:
            base_gap = _compute_base_gap(sc_format, base_res)

    gaps_by_donor = getattr(leave_one_out_result, "gaps_by_donor", {})

    if time_index is None or len(time_index) == 0:
        raise RuntimeError("time_index est requis pour la figure LOO.")

    if base_gap is None:
        raise RuntimeError("base_gap introuvable pour la figure LOO.")

    pm = PlotManager()
    return pm.leave_one_out_gaps(
        time_index=time_index,
        base_gap=base_gap,
        gaps_by_donor=gaps_by_donor,
        T0=T0,
    )


def build_backdating_figure(
    backdating_result: Any,
    sc_format: Any = None,
):
    """Build backdating gaps figure."""
    if PlotManager is None:
        raise RuntimeError("PlotManager not available.")

    time_index = _extract_time_index(sc_format) if sc_format is not None else None
    T0 = _extract_T0(sc_format) if sc_format is not None else None

    runs = getattr(backdating_result, "runs", [])

    if time_index is None or len(time_index) == 0:
        raise RuntimeError("time_index est requis pour la figure backdating.")

    pm = PlotManager()
    return pm.backdating_gaps(time_index=time_index, runs=runs, real_T0=T0)


def build_backdating_ratio_figure(backdating_result: Any):
    """Build backdating RMSPE ratio bar chart."""
    if PlotManager is None:
        raise RuntimeError("PlotManager not available.")
    pm = PlotManager()
    runs = getattr(backdating_result, "runs", [])
    return pm.backdating_ratio_bars(runs)
