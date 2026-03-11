# app_ui/state/workflow.py

from __future__ import annotations

from typing import Dict

import streamlit as st

from . import keys


WORKFLOW_ORDER = [
    "data_ready",
    "exploration_ready",
    "estimation_ready",
    "results_ready",
    "robustness_ready",
]


def get_workflow_status() -> Dict[str, bool]:
    workflow = st.session_state.get(keys.WORKFLOW, {})
    return {step: bool(workflow.get(step, False)) for step in WORKFLOW_ORDER}


def set_workflow_status(step: str, value: bool) -> None:
    workflow = st.session_state.get(keys.WORKFLOW, {})
    workflow[step] = bool(value)
    st.session_state[keys.WORKFLOW] = workflow

    if value:
        st.session_state[keys.LAST_VALID_STEP] = step


def mark_step_complete(step: str) -> None:
    set_workflow_status(step, True)


def mark_step_incomplete(step: str) -> None:
    set_workflow_status(step, False)


def recompute_workflow_from_state() -> Dict[str, bool]:
    """
    Derive workflow flags exclusively from canonical session-state objects.
    """
    sc_format = st.session_state.get(keys.SC_FORMAT)
    estimation_result = st.session_state.get(keys.ESTIMATION_RESULT)
    fit_summary_data = st.session_state.get(keys.FIT_SUMMARY_DATA)
    robustness_results = st.session_state.get(keys.ROBUSTNESS_RESULTS)

    workflow = {
        "data_ready": sc_format is not None,
        "exploration_ready": sc_format is not None,
        "estimation_ready": estimation_result is not None,
        "results_ready": estimation_result is not None and fit_summary_data is not None,
        "robustness_ready": robustness_results is not None,
    }

    st.session_state[keys.WORKFLOW] = workflow

    last_valid_step = None
    for step in reversed(WORKFLOW_ORDER):
        if workflow[step]:
            last_valid_step = step
            break

    st.session_state[keys.LAST_VALID_STEP] = last_valid_step
    return workflow