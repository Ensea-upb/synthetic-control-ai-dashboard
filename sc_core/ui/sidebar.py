from __future__ import annotations
import streamlit as st

from .cache_control import (
    clear_estimation_outputs,
    clear_sc_format_and_downstream,
    clear_all_app_state,
)


def render_workflow_sidebar() -> None:
    """Render the workflow status and quick-action sidebar."""
    wf = st.session_state.get("workflow", {})
    sc_format = st.session_state.get("sc_format", None)
    res = st.session_state.get("estimation_result", None)
    stale = st.session_state.get("estimation_stale", False)

    with st.sidebar:
        st.header("Workflow")

        # Keys aligned with app_ui/state/keys.py workflow dict
        st.write(f"Données    : {'✅' if wf.get('data_ready', False) else '❌'}")
        st.write(f"Exploration: {'✅' if wf.get('exploration_ready', False) else '❌'}")
        stale_suffix = " ⚠️ périmée" if stale else ""
        st.write(f"Estimation : {'✅' if wf.get('estimation_ready', False) else '❌'}{stale_suffix}")
        st.write(f"Résultats  : {'✅' if wf.get('results_ready', False) else '❌'}")
        st.write(f"Robustesse : {'✅' if wf.get('robustness_ready', False) else '❌'}")

        st.divider()

        if sc_format is not None:
            st.subheader("Configuration active")
            st.write(f"**Traitée** : {sc_format.treated}")
            st.write(f"**Donneurs** : {len(sc_format.donors)}")
            st.write(f"**T0** : {sc_format.T0}")
            n_pre = int(sc_format.pre_mask.sum())
            n_total = len(sc_format.years)
            st.write(f"**Périodes pré** : {n_pre} / {n_total} total")
            st.write(f"**Features X** : {len(sc_format.feature_names)}")

        if res is not None:
            st.divider()
            st.subheader("Dernière estimation")
            st.write(f"**Loss** : {res.loss:.6f}")
            st.write(f"**Itérations** : {res.n_iter}")
            n_active = int((res.w > 1e-4).sum())
            st.write(f"**Donneurs actifs** : {n_active} / {len(res.w)}")

        st.divider()
        st.subheader("Réinitialisation")

        if st.button("Effacer estimation", use_container_width=True):
            clear_estimation_outputs()
            st.success("Estimation et résultats supprimés.")

        if st.button("Effacer SCFormat + estimation", use_container_width=True):
            clear_sc_format_and_downstream()
            st.success("SCFormat et résultats supprimés.")

        if st.button("Tout effacer", use_container_width=True):
            clear_all_app_state()
            st.success("Session réinitialisée.")
