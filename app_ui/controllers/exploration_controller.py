from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List
from uuid import uuid4

import streamlit as st

from app_ui.state import keys
from app_ui.services.exploration_service import build_exploration_payload


def _get_configs() -> List[Dict[str, Any]]:
    configs = st.session_state.get(keys.EXPLORATION_CONFIGS)
    if configs is None:
        configs = []
        st.session_state[keys.EXPLORATION_CONFIGS] = configs
    return configs


def add_exploration_graph_config(defaults: Dict[str, Any] | None = None) -> str:
    graph_id = f"exp_{uuid4().hex[:8]}"
    config = {
        "graph_id": graph_id,
        "variable": None,
        "treated_unit": None,
        "control_units": [],
        "max_missing_by_city": 2,
        "intervention_time": None,
        "show_envelope": True,
        "comment_user": "",
        "comment_ai": None,
        "last_error": None,
    }
    if defaults:
        config.update(defaults)

    configs = _get_configs()
    configs.append(config)
    st.session_state[keys.EXPLORATION_CONFIGS] = configs
    return graph_id


def remove_exploration_graph_config(graph_id: str) -> None:
    configs = _get_configs()
    configs = [cfg for cfg in configs if cfg.get("graph_id") != graph_id]
    st.session_state[keys.EXPLORATION_CONFIGS] = configs


def update_exploration_graph_config(graph_id: str, patch: Dict[str, Any]) -> None:
    configs = _get_configs()
    for cfg in configs:
        if cfg.get("graph_id") == graph_id:
            cfg.update(patch)
            break
    st.session_state[keys.EXPLORATION_CONFIGS] = configs


def build_exploration_result(
    graph_config: Dict[str, Any],
    df_raw,
    *,
    city_col: str = "ville",
    date_col: str = "date",
):
    payload = build_exploration_payload(
        df=df_raw,
        variable=graph_config["variable"],
        treated_unit=graph_config["treated_unit"],
        control_units=graph_config["control_units"],
        city_col=city_col,
        date_col=date_col,
        max_missing_by_city=int(graph_config.get("max_missing_by_city", 2)),
        intervention_time=graph_config.get("intervention_time"),
        show_envelope=bool(graph_config.get("show_envelope", True)),
    )
    return payload


def list_exploration_configs() -> List[Dict[str, Any]]:
    return deepcopy(_get_configs())