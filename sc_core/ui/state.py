from __future__ import annotations

"""
Legacy compatibility layer.

Do not use this module as an authority for application state.
The canonical session-state API lives in app_ui.state.*.
"""

import streamlit as st

from app_ui.state import keys
from app_ui.state.workflow import recompute_workflow_from_state


def get_workflow() -> dict:
    return st.session_state.get(keys.WORKFLOW, {})


def get_df_raw():
    return st.session_state.get(keys.DF_RAW, None)


def get_sc_format():
    return st.session_state.get(keys.SC_FORMAT, None)


def get_estimation_result():
    return st.session_state.get(keys.ESTIMATION_RESULT, None)


def get_fit_summary_data():
    return st.session_state.get(keys.FIT_SUMMARY_DATA, None)


def set_estimation_result(result) -> None:
    st.session_state[keys.ESTIMATION_RESULT] = result
    recompute_workflow_from_state()


def set_fit_summary_data(fit_summary) -> None:
    st.session_state[keys.FIT_SUMMARY_DATA] = fit_summary
    recompute_workflow_from_state()


def reset_estimation_outputs() -> None:
    st.session_state[keys.ESTIMATION_RESULT] = None
    st.session_state[keys.FIT_SUMMARY_DATA] = None
    st.session_state[keys.ROBUSTNESS_RESULTS] = None
    st.session_state[keys.FIGURE_CACHE] = {}
    recompute_workflow_from_state()