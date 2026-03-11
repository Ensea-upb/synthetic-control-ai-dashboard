# app_ui/controllers/results_controller.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st

from app_ui.state import keys


@dataclass
class ResultsPayload:
    fit_summary_data: Any
    estimation_result: Any
    donor_weights: Optional[Dict[str, float]]
    covariate_weights: Optional[Dict[str, float]]
    objective_history: Optional[Any]


def build_results_payload() -> ResultsPayload:
    """
    Extract results from session_state and prepare UI payload.
    """

    fit_summary_data = st.session_state.get(keys.FIT_SUMMARY_DATA)
    estimation_result = st.session_state.get(keys.ESTIMATION_RESULT)

    donor_weights = None
    cov_weights = None
    history = None

    if estimation_result is not None:

        try:
            donor_weights = estimation_result.donor_weights_dict
        except Exception:
            donor_weights = None

        try:
            cov_weights = estimation_result.covariate_weights_dict
        except Exception:
            cov_weights = None

        try:
            history = estimation_result.objective_history
        except Exception:
            try:
                history = estimation_result.best_objective_history
            except Exception:
                history = None

    return ResultsPayload(
        fit_summary_data=fit_summary_data,
        estimation_result=estimation_result,
        donor_weights=donor_weights,
        covariate_weights=cov_weights,
        objective_history=history,
    )


def validate_results_ready(payload: ResultsPayload) -> bool:
    if payload.fit_summary_data is None:
        return False
    if payload.estimation_result is None:
        return False
    return True