# app_ui/state/invalidation.py

from __future__ import annotations

from typing import Any

import streamlit as st

from . import keys
from .workflow import recompute_workflow_from_state


def _safe_changed(old: Any, new: Any) -> bool:
    """Safe equality check that handles numpy arrays, DataFrames, and complex objects.

    Returns True (assume changed) when equality cannot be determined as a scalar bool.
    This is conservative: it may trigger unnecessary invalidations, but never
    silently skip a real change.
    """
    if old is None and new is None:
        return False
    if old is None or new is None:
        return True
    try:
        result = old != new
        # numpy / pandas comparisons return arrays — treat as "changed"
        if hasattr(result, "__iter__") or hasattr(result, "__len__"):
            return True
        return bool(result)
    except Exception:
        # Cannot compare — conservatively assume changed
        return True


def invalidate_from_data_change() -> None:
    """Invalidate all downstream objects after a data or data-configuration change."""
    st.session_state[keys.SC_FORMAT] = None
    st.session_state[keys.DATA_SUMMARY] = None
    st.session_state[keys.EXPLORATION_SUMMARY] = None
    st.session_state[keys.ESTIMATION_CONFIG] = None
    st.session_state[keys.ESTIMATION_RESULT] = None
    st.session_state[keys.FIT_SUMMARY_DATA] = None
    st.session_state[keys.ROBUSTNESS_CONFIG] = None
    st.session_state[keys.ROBUSTNESS_RESULTS] = None
    st.session_state[keys.EXPORT_PAYLOAD] = None
    st.session_state[keys.FIGURE_CACHE] = {}
    st.session_state[keys.EXPLORATION_CONFIGS] = []
    st.session_state[keys.EXPLORATION_RENDERED] = {}
    st.session_state[keys.ESTIMATION_STALE] = False
    recompute_workflow_from_state()


def invalidate_from_estimation_change() -> None:
    """Invalidate all objects derived from estimation settings."""
    st.session_state[keys.ESTIMATION_RESULT] = None
    st.session_state[keys.FIT_SUMMARY_DATA] = None
    st.session_state[keys.ROBUSTNESS_CONFIG] = None
    st.session_state[keys.ROBUSTNESS_RESULTS] = None
    st.session_state[keys.EXPORT_PAYLOAD] = None
    st.session_state[keys.FIGURE_CACHE] = {}
    st.session_state[keys.ESTIMATION_INPUT_SNAPSHOT] = None
    st.session_state[keys.ESTIMATION_STALE] = False
    recompute_workflow_from_state()


def mark_estimation_stale() -> None:
    """Mark current estimation as stale (SCFormat changed but estimation not re-run)."""
    st.session_state[keys.ESTIMATION_STALE] = True


def maybe_invalidate_on_data_config_change(new_config: Any) -> bool:
    old_config = st.session_state.get(keys.DATA_CONFIG)
    changed = _safe_changed(old_config, new_config)
    if changed:
        invalidate_from_data_change()
        st.session_state[keys.DATA_CONFIG] = new_config
    return changed


def maybe_invalidate_on_estimation_config_change(new_config: Any) -> bool:
    old_config = st.session_state.get(keys.ESTIMATION_CONFIG)
    changed = _safe_changed(old_config, new_config)
    if changed:
        invalidate_from_estimation_change()
        st.session_state[keys.ESTIMATION_CONFIG] = new_config
    return changed