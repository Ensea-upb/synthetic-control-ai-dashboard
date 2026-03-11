# app_ui/services/estimation_service.py

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sc_core.estimateur.methods.estimator import (
    fit_xv_random_search_from_scformat,
    fit_xv_bilevel_from_scformat,
    fit_xv_trainval_from_scformat,
)
from sc_core.results.postprocess import (
    build_fit_summary_data,
    build_result_dict_for_ui,
)


@dataclass
class EstimationRunOutput:
    method_name: str
    estimation_result: Any
    fit_summary_data: Any
    ui_result_dict: Optional[Dict[str, Any]] = None


def _filter_kwargs_for_callable(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs accepted by the callable.
    This makes the frontend resilient to backend signature drift.
    """
    sig = inspect.signature(func)
    accepted = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _call_backend(func, **kwargs):
    filtered_kwargs = _filter_kwargs_for_callable(func, kwargs)
    return func(**filtered_kwargs)


def _resolve_estimator(method_name: str):
    method_name = method_name.lower().strip()

    if method_name == "random_search":
        return fit_xv_random_search_from_scformat
    if method_name == "bilevel":
        return fit_xv_bilevel_from_scformat
    if method_name == "trainval":
        return fit_xv_trainval_from_scformat

    raise ValueError(f"Unknown estimation method: {method_name}")

def build_estimation_input_snapshot(
    *,
    sc_format: Any,
    method_name: str,
) -> Dict[str, Any]:
    """
    Build an exact snapshot of the data passed to backend estimation.
    This snapshot is intended for UI inspection/debugging only.
    """
    years = list(getattr(sc_format, "years", []))
    treated = getattr(sc_format, "treated", "treated")
    donors = list(getattr(sc_format, "donors", []))

    Y1 = getattr(sc_format, "Y1", None)
    Y0 = getattr(sc_format, "Y0", None)
    X1 = getattr(sc_format, "X1", None)
    X0 = getattr(sc_format, "X0", None)
    pre_mask = getattr(sc_format, "pre_mask", None)
    post_mask = getattr(sc_format, "post_mask", None)
    row_var = list(getattr(sc_format, "row_var", []))

    return {
        "method_name": method_name,
        "treated": treated,
        "donors": donors,
        "years": years,
        "Y1": None if Y1 is None else Y1.copy(),
        "Y0": None if Y0 is None else Y0.copy(),
        "X1": None if X1 is None else X1.copy(),
        "X0": None if X0 is None else X0.copy(),
        "pre_mask": None if pre_mask is None else pre_mask.copy(),
        "post_mask": None if post_mask is None else post_mask.copy(),
        "row_var": row_var,
        "X_long": getattr(sc_format, "X_long", None),
    }


def run_estimation_from_scformat(
    *,
    sc_format: Any,
    estimation_config: Dict[str, Any],
    progress_callback=None,
) -> Any:
    """
    Dispatch to the appropriate modern estimator from SCFormat.
    """
    if sc_format is None:
        raise ValueError("sc_format is required.")

    method_name = str(estimation_config.get("method_name", "random_search"))
    estimator = _resolve_estimator(method_name)

    kwargs = dict(estimation_config)
    kwargs.pop("method_name", None)
    kwargs["sc_format"] = sc_format
    kwargs["callback"] = progress_callback

    return _call_backend(estimator, **kwargs)


def build_fit_summary_from_result(
    *,
    sc_format: Any,
    estimation_result: Any,
) -> Any:
    """
    Build FitSummaryData using the backend post-process layer.
    """
    if sc_format is None:
        raise ValueError("sc_format is required.")
    if estimation_result is None:
        raise ValueError("estimation_result is required.")

    return _call_backend(
        build_fit_summary_data,
        sc_format=sc_format,
        result=estimation_result,
        estimation_result=estimation_result,
    )


def build_result_ui_payload(
    *,
    sc_format: Any,
    estimation_result: Any,
    fit_summary_data: Any,
) -> Optional[Dict[str, Any]]:
    """
    Optional UI-ready dict from backend if available.
    """
    try:
        return _call_backend(
            build_result_dict_for_ui,
            sc_format=sc_format,
            result=estimation_result,
            estimation_result=estimation_result,
            fit_summary_data=fit_summary_data,
        )
    except Exception:
        return None


def run_full_estimation_pipeline(
    *,
    sc_format: Any,
    estimation_config: Dict[str, Any],
    progress_callback=None,
) -> EstimationRunOutput:
    """
    End-to-end estimation + fit summary + optional UI payload.
    """
    result = run_estimation_from_scformat(
        sc_format=sc_format,
        estimation_config=estimation_config,
        progress_callback=progress_callback,
    )

    fit_summary = build_fit_summary_from_result(
        sc_format=sc_format,
        estimation_result=result,
    )

    ui_payload = build_result_ui_payload(
        sc_format=sc_format,
        estimation_result=result,
        fit_summary_data=fit_summary,
    )

    return EstimationRunOutput(
        method_name=str(estimation_config.get("method_name", "random_search")),
        estimation_result=result,
        fit_summary_data=fit_summary,
        ui_result_dict=ui_payload,
    )