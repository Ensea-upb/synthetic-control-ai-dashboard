# app_ui/state/initialization.py

from __future__ import annotations

import threading
from typing import Any, Dict

import streamlit as st

from . import keys


def _default_workflow() -> Dict[str, bool]:
    return {
        "data_ready": False,
        "exploration_ready": False,
        "estimation_ready": False,
        "results_ready": False,
        "robustness_ready": False,
    }


def _default_state() -> Dict[str, Any]:
    return {
        keys.DF_RAW: None,
        keys.DFS_BY_NAME: {},
        keys.PREPARED_DFS_BY_NAME: {},
        keys.DATA_CONFIG: None,
        keys.SC_FORMAT: None,
        keys.DATA_SUMMARY: None,
        keys.EXPLORATION_SUMMARY: None,
        keys.ESTIMATION_CONFIG: None,
        keys.ESTIMATION_RESULT: None,
        keys.FIT_SUMMARY_DATA: None,
        keys.ROBUSTNESS_CONFIG: None,
        keys.ROBUSTNESS_RESULTS: None,
        keys.FIGURE_CACHE: {},
        keys.WORKFLOW: _default_workflow(),
        keys.CURRENT_PAGE: "Accueil",
        keys.LAST_VALID_STEP: None,
        keys.APP_MESSAGES: [],
        keys.AI_ENABLED: True,
        keys.AI_BACKEND: "local",
        keys.AI_MODEL_STATUS: "unknown",
        keys.AI_MESSAGES: [],
        keys.AI_CONTEXT_CACHE: {},
        keys.AI_LAST_RESPONSE: None,
        keys.AI_LAST_ERROR: None,
        keys.EXPORT_PAYLOAD: None,
        keys.APP_INITIALIZED: True,
        keys.EXPLORATION_CONFIGS: [],
        keys.EXPLORATION_RENDERED: {},
        keys.ESTIMATION_INPUT_SNAPSHOT: None,
        keys.ESTIMATION_STALE: False,
    }


def initialize_app_state(force: bool = False) -> None:
    """Initialize session_state keys once per session.

    Parameters
    ----------
    force : bool
        If True, reset every managed key to its default value (full reset).
    """
    defaults = _default_state()

    if force:
        for key, value in defaults.items():
            st.session_state[key] = value
        return

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def trigger_ai_model_loading() -> None:
    """Trigger AI model pre-loading in a background thread.

    Called from the Accueil page to warm up the model before the user
    reaches pages that need it.  The thread is non-blocking and is
    silently abandoned if the model is unavailable.
    """
    def _load_in_background() -> None:
        try:
            from IA_integration.model_loader import get_local_ai_manager
            get_local_ai_manager()
        except Exception:
            pass  # Never block the application if AI is unavailable

    # Only spawn the thread once: check if the model is already being loaded
    status = st.session_state.get(keys.AI_MODEL_STATUS, "unknown")
    if status not in ("ready", "loading"):
        t = threading.Thread(target=_load_in_background, daemon=True)
        t.start()
