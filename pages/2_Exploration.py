from __future__ import annotations

import streamlit as st

from app_ui.state.initialization import initialize_app_state
from app_ui.state.workflow import recompute_workflow_from_state
from app_ui.state import keys

from app_ui.controllers.exploration_controller import (
    add_exploration_graph_config,
    remove_exploration_graph_config,
    update_exploration_graph_config,
    list_exploration_configs,
    build_exploration_result,
)
from app_ui.services.exploration_service import build_exploration_comment_context
from app_ui.services.plotting_service import (
    build_exploration_dynamic_plot,
    build_exploration_static_plot,
)
from app_ui.services.exploration_ai_service import generate_exploration_chart_comment

try:
    from sc_core.ui.sidebar import render_workflow_sidebar
except Exception:
    render_workflow_sidebar = None


st.set_page_config(page_title="Exploration | Synthetic Control", layout="wide")

initialize_app_state()
st.session_state[keys.CURRENT_PAGE] = "Exploration"
recompute_workflow_from_state()

if render_workflow_sidebar:
    render_workflow_sidebar()

st.title("3) Exploration des données")

df_raw = st.session_state.get(keys.DF_RAW)
if df_raw is None:
    st.warning("Aucune donnée chargée.")
    st.stop()

config = st.session_state.get(keys.DATA_CONFIG) or {}
city_col = config.get("city_col", "ville")
date_col = config.get("date_col", "date")

if city_col not in df_raw.columns:
    st.warning(f"Colonne ville introuvable : {city_col}")
    st.stop()

available_units = sorted(df_raw[city_col].dropna().astype(str).unique().tolist())
available_variables = [
    c for c in df_raw.columns
    if c not in {city_col, date_col}
]

st.subheader("Résumé dataset")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Lignes", df_raw.shape[0])
with c2:
    st.metric("Colonnes", df_raw.shape[1])
with c3:
    st.metric("Valeurs manquantes", int(df_raw.isna().sum().sum()))

st.divider()

if st.button("Ajouter un nouveau graphique", type="primary"):
    default_treated = available_units[0] if available_units else None
    default_controls = available_units[1:4] if len(available_units) > 1 else []
    add_exploration_graph_config(
        {
            "treated_unit": default_treated,
            "control_units": default_controls,
        }
    )
    st.rerun()

configs = list_exploration_configs()

if not configs:
    st.info("Aucun graphique ajouté pour le moment.")
    st.stop()

for idx, graph_cfg in enumerate(configs, start=1):
    graph_id = graph_cfg["graph_id"]

    with st.container(border=True):
        st.markdown(f"### Graphique {idx}")

        col_a, col_b = st.columns([6, 1])
        with col_b:
            if st.button("Supprimer", key=f"delete_{graph_id}"):
                remove_exploration_graph_config(graph_id)
                st.rerun()

        variable = st.selectbox(
            "Variable",
            options=available_variables,
            index=available_variables.index(graph_cfg["variable"]) if graph_cfg["variable"] in available_variables else 0,
            key=f"variable_{graph_id}",
        )

        treated_unit = st.selectbox(
            "Unité traitée",
            options=available_units,
            index=available_units.index(graph_cfg["treated_unit"]) if graph_cfg["treated_unit"] in available_units else 0,
            key=f"treated_{graph_id}",
        )

        candidate_controls = [u for u in available_units if u != treated_unit]
        current_controls = [u for u in graph_cfg.get("control_units", []) if u in candidate_controls]

        control_units = st.multiselect(
            "Unités de contrôle",
            options=candidate_controls,
            default=current_controls[:],
            key=f"controls_{graph_id}",
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            max_missing_by_city = st.number_input(
                "Nombre max de NA par ville",
                min_value=0,
                value=int(graph_cfg.get("max_missing_by_city", 2)),
                step=1,
                key=f"na_{graph_id}",
            )

        with c2:
            intervention_time = st.text_input(
                "Date/année de traitement (optionnel)",
                value="" if graph_cfg.get("intervention_time") is None else str(graph_cfg.get("intervention_time")),
                key=f"intervention_{graph_id}",
            )

        with c3:
            show_envelope = st.checkbox(
                "Afficher l'enveloppe min-max",
                value=bool(graph_cfg.get("show_envelope", True)),
                key=f"envelope_{graph_id}",
            )

        intervention_value = None
        if str(intervention_time).strip() != "":
            raw = str(intervention_time).strip()
            try:
                intervention_value = int(raw)
            except ValueError:
                intervention_value = raw

        update_exploration_graph_config(
            graph_id,
            {
                "variable": variable,
                "treated_unit": treated_unit,
                "control_units": control_units,
                "max_missing_by_city": int(max_missing_by_city),
                "intervention_time": intervention_value,
                "show_envelope": bool(show_envelope),
            },
        )

        current_cfg = {
            "graph_id": graph_id,
            "variable": variable,
            "treated_unit": treated_unit,
            "control_units": control_units,
            "max_missing_by_city": int(max_missing_by_city),
            "intervention_time": intervention_value,
            "show_envelope": bool(show_envelope),
            "comment_user": graph_cfg.get("comment_user", ""),
            "comment_ai": graph_cfg.get("comment_ai"),
            "last_error": None,
        }

        if not variable:
            st.info("Choisis une variable pour afficher le graphique.")
            continue

        if not treated_unit:
            st.info("Choisis une unité traitée.")
            continue

        if not control_units:
            st.info("Choisis au moins une unité de contrôle.")
            continue

        try:
            payload = build_exploration_result(
                current_cfg,
                df_raw,
                city_col=city_col,
                date_col=date_col,
            )

            if payload["variable_type"] == "dynamic":
                fig = build_exploration_dynamic_plot(**payload)
                st.caption("Type détecté : dynamique")
            else:
                fig = build_exploration_static_plot(**payload)
                st.caption("Type détecté : statique")

            st.pyplot(fig)

            comment_context = build_exploration_comment_context(payload)

            st.markdown("#### Commentaire")
            user_comment = st.text_area(
                "Commentaire utilisateur",
                value=graph_cfg.get("comment_user", ""),
                key=f"user_comment_{graph_id}",
                height=120,
            )

            update_exploration_graph_config(
                graph_id,
                {
                    "comment_user": user_comment,
                    "last_error": None,
                },
            )

            if st.button("Générer commentaire IA", key=f"ai_comment_{graph_id}"):
                response = generate_exploration_chart_comment(
                    chart_context=comment_context,
                    user_comment=user_comment,
                    backend=st.session_state.get(keys.AI_BACKEND, "local"),
                    model_name=None,
                )

                if response.ok:
                    update_exploration_graph_config(
                        graph_id,
                        {
                            "comment_ai": response.content,
                            "last_error": None,
                        },
                    )
                else:
                    update_exploration_graph_config(
                        graph_id,
                        {
                            "last_error": response.error or "Erreur IA inconnue.",
                        },
                    )
                st.rerun()

            refreshed_configs = list_exploration_configs()
            refreshed_cfg = next((x for x in refreshed_configs if x["graph_id"] == graph_id), None)

            if refreshed_cfg and refreshed_cfg.get("comment_ai"):
                st.markdown("**Commentaire IA**")
                st.write(refreshed_cfg["comment_ai"])

            if refreshed_cfg and refreshed_cfg.get("last_error"):
                st.error(refreshed_cfg["last_error"])

        except Exception as exc:
            update_exploration_graph_config(
                graph_id,
                {
                    "last_error": str(exc),
                },
            )
            st.error(f"Impossible de construire le graphique : {exc}")