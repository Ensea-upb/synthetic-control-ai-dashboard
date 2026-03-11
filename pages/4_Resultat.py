from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_ui.components.ai_panel import render_ai_panel
from app_ui.controllers.results_controller import (
    build_results_payload,
    validate_results_ready,
)
from app_ui.services.plotting_service import (
    build_treated_vs_synthetic_plot,
    build_gap_plot,
    build_cumulative_gap_plot,
    build_donor_weights_plot,
    build_covariate_weights_plot,
    build_objective_history_plot,
    fig_to_png_bytes,
)
from app_ui.state import keys
from app_ui.state.initialization import initialize_app_state
from app_ui.state.workflow import recompute_workflow_from_state

try:
    from sc_core.ui.sidebar import render_workflow_sidebar
except Exception:
    render_workflow_sidebar = None


# =========================================================
# Page init
# =========================================================

st.set_page_config(page_title="Résultats | Synthetic Control", layout="wide")

initialize_app_state()
st.session_state[keys.CURRENT_PAGE] = "Resultat"
recompute_workflow_from_state()

if render_workflow_sidebar is not None:
    render_workflow_sidebar()

st.title("5) Résultats — Analyse du Synthetic Control")
st.caption("Cette page présente les résultats analytiques de l'estimation SCM déjà exécutée.")

payload = build_results_payload()

if not validate_results_ready(payload):
    st.warning("Aucun résultat disponible. Lance d'abord une estimation.")
    st.stop()

fit_summary = payload.fit_summary_data


# =========================================================
# SECTION 0 — Trajectoires : Traité vs Synthétique
# =========================================================

st.subheader("1) Trajectoires : Traité vs Contrôle Synthétique")

try:
    fig_traj = build_treated_vs_synthetic_plot(fit_summary)
    st.pyplot(fig_traj)
    png_traj = fig_to_png_bytes(fig_traj)
    st.download_button(
        label="Télécharger (PNG)",
        data=png_traj,
        file_name="trajectoires_traite_synthetique.png",
        mime="image/png",
        key="dl_traj",
    )
except Exception as exc:
    st.error(f"Impossible de construire la figure des trajectoires : {exc}")


st.divider()


# =========================================================
# SECTION 1 — Effets estimés
# =========================================================

st.subheader("2) Effets estimés")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Gap instantané**")
    try:
        fig_gap = build_gap_plot(fit_summary)
        st.pyplot(fig_gap)
        png_gap = fig_to_png_bytes(fig_gap)
        st.download_button(
            label="Télécharger (PNG)",
            data=png_gap,
            file_name="gap_instantane.png",
            mime="image/png",
            key="dl_gap",
        )
    except Exception as exc:
        st.error(f"Impossible de construire le gap plot : {exc}")

with col2:
    st.markdown("**Effet cumulé**")
    try:
        fig_cumgap = build_cumulative_gap_plot(fit_summary)
        st.pyplot(fig_cumgap)
        png_cumgap = fig_to_png_bytes(fig_cumgap)
        st.download_button(
            label="Télécharger (PNG)",
            data=png_cumgap,
            file_name="effet_cumule.png",
            mime="image/png",
            key="dl_cumgap",
        )
    except Exception as exc:
        st.error(f"Impossible de construire le cumulative gap plot : {exc}")


st.divider()


# =========================================================
# SECTION 2 — Diagnostique RMSPE
# =========================================================

st.subheader("3) Diagnostic RMSPE")

try:
    t_idx  = np.asarray(fit_summary.time_index, dtype=float)
    y_t    = np.asarray(fit_summary.y_treated,  dtype=float).reshape(-1)
    y_s    = np.asarray(fit_summary.y_synth,    dtype=float).reshape(-1)
    T0     = fit_summary.T0

    if T0 is not None:
        pre_mask  = t_idx <  T0
        post_mask = t_idx >= T0

        gap = y_t - y_s

        def _rmspe(g):
            return float(np.sqrt(np.mean(g ** 2))) if g.size > 0 else float("nan")

        pre_rmspe  = _rmspe(gap[pre_mask])
        post_rmspe = _rmspe(gap[post_mask])
        ratio      = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("nan")

        rmspe_df = pd.DataFrame(
            {
                "Métrique": ["RMSPE pré-traitement", "RMSPE post-traitement", "Ratio post/pré"],
                "Valeur":   [f"{pre_rmspe:.4f}", f"{post_rmspe:.4f}", f"{ratio:.2f}"],
            }
        )
        st.dataframe(rmspe_df, use_container_width=True, hide_index=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("RMSPE pré",   f"{pre_rmspe:.4f}")
        col_b.metric("RMSPE post",  f"{post_rmspe:.4f}")
        col_c.metric("Ratio",       f"{ratio:.2f}",
                     delta=None,
                     help="Un ratio > 1 indique un effet post-traitement plus grand que le bruit pré-traitement.")
    else:
        st.info("T0 non disponible — impossible de calculer les métriques RMSPE.")

except Exception as exc:
    st.warning(f"Calcul RMSPE impossible : {exc}")


st.divider()


# =========================================================
# SECTION 3 — Poids des donneurs
# =========================================================

st.subheader("4) Poids des donneurs")

if payload.donor_weights:
    donor_df = (
        pd.DataFrame(
            {
                "donor":  list(payload.donor_weights.keys()),
                "weight": list(payload.donor_weights.values()),
            }
        )
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(donor_df, use_container_width=True)

    try:
        fig_donor = build_donor_weights_plot(payload.donor_weights)
        st.pyplot(fig_donor)
        st.download_button(
            label="Télécharger (PNG)",
            data=fig_to_png_bytes(fig_donor),
            file_name="poids_donneurs.png",
            mime="image/png",
            key="dl_donor",
        )
    except Exception as exc:
        st.error(f"Impossible de construire la figure des poids donneurs : {exc}")
else:
    st.info("Aucun poids donneur disponible.")


st.divider()


# =========================================================
# SECTION 4 — Poids des covariables
# =========================================================

st.subheader("5) Poids des covariables")

if payload.covariate_weights:
    cov_df = (
        pd.DataFrame(
            {
                "covariate": list(payload.covariate_weights.keys()),
                "weight":    list(payload.covariate_weights.values()),
            }
        )
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(cov_df, use_container_width=True)

    try:
        fig_cov = build_covariate_weights_plot(payload.covariate_weights)
        st.pyplot(fig_cov)
        st.download_button(
            label="Télécharger (PNG)",
            data=fig_to_png_bytes(fig_cov),
            file_name="poids_covariables.png",
            mime="image/png",
            key="dl_cov",
        )
    except Exception as exc:
        st.error(f"Impossible de construire la figure des poids covariables : {exc}")
else:
    st.info("Aucun poids covariable disponible.")


st.divider()


# =========================================================
# SECTION 5 — Historique de convergence
# =========================================================

if payload.objective_history is not None:
    st.subheader("6) Historique de convergence de l'optimiseur")
    try:
        fig_obj = build_objective_history_plot(payload.objective_history)
        st.pyplot(fig_obj)
        st.download_button(
            label="Télécharger (PNG)",
            data=fig_to_png_bytes(fig_obj),
            file_name="historique_convergence.png",
            mime="image/png",
            key="dl_obj",
        )
    except Exception as exc:
        st.error(f"Impossible de construire l'historique d'objectif : {exc}")

    st.divider()


# =========================================================
# SECTION 6 — Résumé technique
# =========================================================

st.subheader("7) Résumé technique")

with st.expander("Afficher le résumé technique du fit", expanded=False):
    st.write(repr(fit_summary))


# =========================================================
# AI interpretation
# =========================================================

st.divider()

render_ai_panel(
    page_name="Resultat",
    available_actions=[
        "Interpréter les résultats",
        "Analyser la qualité du fit",
        "Commenter les poids des donneurs",
        "Évaluer la significativité du ratio RMSPE",
    ],
)
