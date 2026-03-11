# app_ui/controllers/estimation_controller.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

from app_ui.services.estimation_service import (
    EstimationRunOutput,
    build_estimation_input_snapshot,
    run_full_estimation_pipeline,
)
from app_ui.state import keys
from app_ui.state.invalidation import maybe_invalidate_on_estimation_config_change
from app_ui.state.workflow import recompute_workflow_from_state


@dataclass
class EstimationConfigValidation:
    ok: bool
    errors: List[str]


def build_estimation_config(
    *,
    method_name: str,
    **kwargs,
) -> Dict[str, Any]:
    """
    Build a normalized estimation config from UI inputs.

    This function is intentionally permissive at the app layer:
    it stores UI-level estimation parameters, while the backend
    service later filters unsupported kwargs per estimator.
    """
    if not method_name or not str(method_name).strip():
        raise ValueError("method_name is required.")

    config = {"method_name": str(method_name).strip()}

    for key, value in kwargs.items():
        config[key] = value

    return config


def validate_estimation_config(
    sc_format: Any,
    config: Dict[str, Any],
) -> EstimationConfigValidation:
    errors: List[str] = []

    if sc_format is None:
        errors.append("SCFormat is missing. Build data configuration first.")

    method_name = str(config.get("method_name", "")).strip()
    if method_name not in {"random_search", "bilevel", "trainval"}:
        errors.append("method_name must be one of {'random_search', 'bilevel', 'trainval'}.")

    if method_name == "random_search":
        if int(config.get("n_iter", 0)) <= 0:
            errors.append("random_search requires n_iter > 0.")

    if method_name == "bilevel":
        if int(config.get("maxiter", 0)) <= 0:
            errors.append("bilevel requires maxiter > 0.")

    if method_name == "trainval":
        val_last_k = int(config.get("val_last_k", 0))
        if val_last_k <= 0:
            errors.append("trainval requires val_last_k > 0.")
        if int(config.get("n_iter", 0)) <= 0:
            errors.append("trainval requires n_iter > 0.")
        # Guard: train set must keep at least 2 periods
        if sc_format is not None and val_last_k > 0:
            pre_mask = getattr(sc_format, "pre_mask", None)
            n_pre = int(pre_mask.sum()) if pre_mask is not None else 0
            if n_pre > 0 and val_last_k >= n_pre - 1:
                errors.append(
                    f"trainval: val_last_k={val_last_k} trop grand pour "
                    f"{n_pre} périodes pré-traitement. "
                    f"Utilise val_last_k ≤ {max(1, n_pre - 2)}."
                )

    return EstimationConfigValidation(ok=(len(errors) == 0), errors=errors)


def run_and_persist_estimation(
    *,
    sc_format: Any,
    estimation_config: Dict[str, Any],
    progress_callback=None,
) -> EstimationRunOutput:
    """
    Persist config, invalidate downstream outputs if needed, run estimation,
    then store estimation_result and fit_summary_data.
    """
    maybe_invalidate_on_estimation_config_change(estimation_config)

    snapshot = build_estimation_input_snapshot(
        sc_format=sc_format,
        method_name=str(estimation_config.get("method_name", "random_search")),
    )
    st.session_state[keys.ESTIMATION_INPUT_SNAPSHOT] = snapshot

    output = run_full_estimation_pipeline(
        sc_format=sc_format,
        estimation_config=estimation_config,
        progress_callback=progress_callback,
    )

    st.session_state[keys.ESTIMATION_CONFIG] = estimation_config
    st.session_state[keys.ESTIMATION_RESULT] = output.estimation_result
    st.session_state[keys.FIT_SUMMARY_DATA] = output.fit_summary_data

    export_payload = st.session_state.get(keys.EXPORT_PAYLOAD) or {}
    export_payload["estimation"] = {
        "config": estimation_config,
        "ui_result_dict": output.ui_result_dict,
    }
    st.session_state[keys.EXPORT_PAYLOAD] = export_payload

    recompute_workflow_from_state()
    return output


def build_estimation_summary(estimation_result: Any) -> Dict[str, Any]:
    """
    Lightweight result summary for UI cards.

    Designed to be robust against backend attribute drift.
    """
    summary: Dict[str, Any] = {}

    candidate_attrs = [
        "loss",
        "objective_value",
        "best_objective",
        "n_iter",
        "status",
        "solver_status",
        "method",
    ]
    for attr in candidate_attrs:
        try:
            value = getattr(estimation_result, attr)
            summary[attr] = value
        except Exception:
            pass

    try:
        summary["unit_weights"] = estimation_result.donor_weights_dict
    except Exception:
        pass

    try:
        summary["covariate_weights"] = estimation_result.covariate_weights_dict
    except Exception:
        pass

    try:
        summary["objective_history"] = estimation_result.objective_history
    except Exception:
        try:
            summary["objective_history"] = estimation_result.best_objective_history
        except Exception:
            pass

    return summary