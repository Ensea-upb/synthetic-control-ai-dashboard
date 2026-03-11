from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import streamlit as st

from app_ui.services.robustness_service import (
    RobustnessRunOutput,
    run_placebo_space_service,
    run_leave_one_out_service,
    run_backdating_service,
    run_rmspe_service,
)
from app_ui.state import keys
from app_ui.state.workflow import recompute_workflow_from_state


@dataclass
class RobustnessConfigValidation:
    ok: bool
    errors: List[str]


def build_robustness_config(
    *,
    run_placebo: bool,
    run_leave_one_out: bool,
    run_backdating: bool,
    run_rmspe: bool,
) -> Dict[str, Any]:
    return {
        "run_placebo": bool(run_placebo),
        "run_leave_one_out": bool(run_leave_one_out),
        "run_backdating": bool(run_backdating),
        "run_rmspe": bool(run_rmspe),
    }


def validate_robustness_config(
    *,
    df: Any,  # conservé pour compatibilité d'API, non requis fonctionnellement
    sc_format: Any,
    estimation_result: Any,
    estimation_config: Optional[Dict[str, Any]],
    robustness_config: Dict[str, Any],
) -> RobustnessConfigValidation:
    errors: List[str] = []

    if sc_format is None:
        errors.append("SCFormat manquant.")
    if estimation_result is None:
        errors.append("Résultat d’estimation manquant.")
    if estimation_config is None:
        errors.append("Configuration d’estimation manquante.")
    if not any(robustness_config.values()):
        errors.append("Sélectionne au moins un test de robustesse.")

    return RobustnessConfigValidation(ok=(len(errors) == 0), errors=errors)


def run_and_persist_robustness(
    *,
    df: Any,               # kept for API compatibility, not used functionally
    sc_format: Any,
    estimation_result: Any,
    estimation_config: Dict[str, Any],
    robustness_config: Dict[str, Any],
    fit_summary_data: Any = None,
) -> RobustnessRunOutput:
    """Orchestrate all robustness tests and persist results to session_state."""
    method_name = str(estimation_config.get("method_name", "random_search"))
    treated_unit = str(getattr(sc_format, "treated", "") or "")

    output = RobustnessRunOutput()

    if robustness_config.get("run_placebo", False):
        output.placebo_result = run_placebo_space_service(
            sc_format=sc_format,
            method_name=method_name,
            estimation_config=estimation_config,
        )

    if robustness_config.get("run_leave_one_out", False):
        loo = run_leave_one_out_service(
            sc_format=sc_format,
            estimation_result=estimation_result,
            method_name=method_name,
            estimation_config=estimation_config,
        )
        output.leave_one_out_result = loo

    if robustness_config.get("run_backdating", False):
        output.backdating_result = run_backdating_service(
            sc_format=sc_format,
            method_name=method_name,
            estimation_config=estimation_config,
        )

    # RMSPE — can run even without explicit placebo if fit_summary_data is available
    if robustness_config.get("run_rmspe", False) and output.placebo_result is not None:
        output.rmspe_metrics, output.rmspe_ratio_series = run_rmspe_service(
            placebo_result=output.placebo_result,
            estimation_result=estimation_result,
            sc_format=sc_format,
            fit_summary_data=fit_summary_data,
            treated_unit=treated_unit or None,
        )

    st.session_state[keys.ROBUSTNESS_CONFIG] = robustness_config
    st.session_state[keys.ROBUSTNESS_RESULTS] = {
        "placebo_result": output.placebo_result,
        "leave_one_out_result": output.leave_one_out_result,
        "backdating_result": output.backdating_result,
        "rmspe_metrics": output.rmspe_metrics,
        "rmspe_ratio_series": output.rmspe_ratio_series,
    }
    recompute_workflow_from_state()
    return output


def get_robustness_results_from_state() -> Dict[str, Any]:
    return st.session_state.get(keys.ROBUSTNESS_RESULTS) or {}