# pages/0_Accueil.py

from __future__ import annotations

import streamlit as st

from app_ui.state.initialization import initialize_app_state, trigger_ai_model_loading
from app_ui.state.workflow import get_workflow_status
from app_ui.state import keys
from app_ui.components.ai_panel import render_ai_panel

try:
    from sc_core.ui.sidebar import render_workflow_sidebar
except Exception:
    render_workflow_sidebar = None


st.set_page_config(page_title="Synthetic Control App", layout="wide")

initialize_app_state()
trigger_ai_model_loading()  # préchauffage du modèle IA local

st.session_state[keys.CURRENT_PAGE] = "Accueil"

if render_workflow_sidebar:
    render_workflow_sidebar()

st.title("Synthetic Control Application")
st.caption("Application professionnelle pour l’analyse causale par Synthetic Control.")

workflow = get_workflow_status()

st.subheader("État du workflow")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Données", "OK" if workflow["data_ready"] else "❌")

with c2:
    st.metric("Exploration", "OK" if workflow["exploration_ready"] else "❌")

with c3:
    st.metric("Estimation", "OK" if workflow["estimation_ready"] else "❌")

with c4:
    st.metric("Résultats", "OK" if workflow["results_ready"] else "❌")

with c5:
    st.metric("Robustesse", "OK" if workflow["robustness_ready"] else "❌")

st.divider()

st.subheader("Description du pipeline")

st.markdown(
"""
1️⃣ Charger un **dataset panel**

2️⃣ Explorer les données

3️⃣ Estimer le **Synthetic Control**

4️⃣ Visualiser les résultats

5️⃣ Tester la **robustesse**

6️⃣ Générer une interprétation
"""
)

st.divider()

col1, col2 = st.columns(2)

with col1:

    if st.button("Réinitialiser l'application", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.rerun()

with col2:

    st.info("Toutes les étapes utilisent le même état reproductible.")

st.divider()

render_ai_panel(
    page_name="Accueil",
    available_actions=[
        "Expliquer le Synthetic Control",
        "Que dois-je faire ensuite ?",
    ],
)