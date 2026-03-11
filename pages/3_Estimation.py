from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app_ui.components.ai_panel import render_ai_panel
from app_ui.controllers.estimation_controller import (
    build_estimation_config,
    build_estimation_summary,
    run_and_persist_estimation,
    validate_estimation_config,
)
from app_ui.services.estimation_progress_service import build_live_fit_summary_like
from app_ui.services.plotting_service import build_main_fit_figure
from app_ui.state import keys
from app_ui.state.initialization import initialize_app_state
from app_ui.state.workflow import recompute_workflow_from_state
from app_ui.state.invalidation import invalidate_from_estimation_change

try:
    from sc_core.ui.sidebar import render_workflow_sidebar
except Exception:
    render_workflow_sidebar = None


# =========================================================
# Helpers
# =========================================================

def _to_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return list(value.tolist())
        except Exception:
            pass
    try:
        return list(value)
    except Exception:
        return []


def build_outcome_table_from_snapshot(snapshot: dict) -> pd.DataFrame:
    if not snapshot:
        return pd.DataFrame()

    years = list(snapshot.get("years", []))
    treated = str(snapshot.get("treated", "treated"))
    donors = [str(x) for x in snapshot.get("donors", [])]

    y1 = snapshot.get("Y1")
    y0 = snapshot.get("Y0")

    if y1 is None or y0 is None:
        return pd.DataFrame()

    y1 = np.asarray(y1, dtype=float).reshape(-1)
    y0 = np.asarray(y0, dtype=float)

    if y0.ndim != 2:
        return pd.DataFrame()

    n_donors = y0.shape[1]
    donor_cols = donors[:n_donors]
    if len(donor_cols) < n_donors:
        donor_cols.extend([f"donor_{j}" for j in range(len(donor_cols), n_donors)])

    values = np.column_stack([y1, y0])
    df = pd.DataFrame(values, columns=[treated] + donor_cols)

    if years and len(years) == df.shape[0]:
        df.insert(0, "date", years)

    df.insert(0, "vars", "y")
    return df


def build_covariate_table_from_snapshot(snapshot: dict) -> pd.DataFrame:
    if not snapshot:
        return pd.DataFrame()

    treated = str(snapshot.get("treated", "treated"))
    donors = [str(x) for x in snapshot.get("donors", [])]
    row_var = [str(x) for x in snapshot.get("row_var", [])]

    x1 = snapshot.get("X1")
    x0 = snapshot.get("X0")

    if x1 is None or x0 is None:
        return pd.DataFrame()

    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x0 = np.asarray(x0, dtype=float)

    if x0.ndim != 2:
        return pd.DataFrame()

    n_donors = x0.shape[1]
    donor_cols = donors[:n_donors]
    if len(donor_cols) < n_donors:
        donor_cols.extend([f"donor_{j}" for j in range(len(donor_cols), n_donors)])

    values = np.column_stack([x1, x0])
    df = pd.DataFrame(values, columns=[treated] + donor_cols)

    if row_var and len(row_var) == df.shape[0]:
        df.insert(0, "vars", row_var)
    else:
        df.insert(0, "vars", [f"x_{i}" for i in range(df.shape[0])])

    return df


def _get_live_plot_inputs(sc_format: Any) -> dict:
    donor_names = _to_list(getattr(sc_format, "donors", None))
    if not donor_names:
        donor_names = _to_list(getattr(sc_format, "donor_units", None))
    if not donor_names:
        donor_names = _to_list(getattr(sc_format, "group_names", None))

    feature_names = _to_list(getattr(sc_format, "row_var", None))
    time_index = _to_list(getattr(sc_format, "years", None))

    y_treated = np.asarray(getattr(sc_format, "Y1"), dtype=float).reshape(-1)
    y0_full = np.asarray(getattr(sc_format, "Y0"), dtype=float)

    if not time_index:
        time_index = list(range(len(y_treated)))

    return {
        "donor_names": donor_names,
        "feature_names": feature_names,
        "time_index": time_index,
        "y_treated": y_treated,
        "y0_full": y0_full,
        "T0": getattr(sc_format, "T0", None),
    }


# =========================================================
# Page init
# =========================================================

st.set_page_config(page_title="Estimation | Synthetic Control App", layout="wide")
initialize_app_state()
st.session_state[keys.CURRENT_PAGE] = "Estimation"
recompute_workflow_from_state()

if render_workflow_sidebar is not None:
    render_workflow_sidebar()

st.title("4) Estimation — Solveur moderne SCM")
st.caption("Cette page pilote l’estimation SCM à partir de SCFormat et permet un suivi live de l’exécution.")

sc_format = st.session_state.get(keys.SC_FORMAT)
if sc_format is None:
    st.warning("Aucun SCFormat détecté. Va d’abord sur la page Données.")
    st.stop()

cfg_prev = st.session_state.get(keys.ESTIMATION_CONFIG) or {}

# =========================================================
# SECTION 1 — Données d’estimation
# =========================================================

st.subheader("1) Données utilisées pour l’estimation")

snapshot = st.session_state.get(keys.ESTIMATION_INPUT_SNAPSHOT)

if snapshot is None:
    st.info(
        "Aucun snapshot d’entrée d’estimation disponible pour le moment. "
        "Lance une estimation pour générer l’instantané exact utilisé par le backend."
    )
else:
    outcome_df = build_outcome_table_from_snapshot(snapshot)
    cov_df = build_covariate_table_from_snapshot(snapshot)

    with st.expander("Outcome utilisé par l’estimation", expanded=True):
        if outcome_df.empty:
            st.info("Outcome indisponible dans le snapshot.")
        else:
            st.dataframe(outcome_df, use_container_width=True)

    with st.expander("Covariables utilisées par l’estimation", expanded=False):
        if cov_df.empty:
            st.info("Covariables indisponibles dans le snapshot.")
        else:
            st.dataframe(cov_df, use_container_width=True)

st.divider()

# =========================================================
# SECTION 2 — Configuration estimateur
# =========================================================

st.subheader("2) Configuration de l’estimateur")

method_options = {
    "Random Search V": "random_search",
    "Bilevel optimization": "bilevel",
    "Train / Validation": "trainval",
}

prev_method = cfg_prev.get("method_name", "random_search")
prev_label = next(
    (label for label, value in method_options.items() if value == prev_method),
    "Random Search V",
)

method_label = st.radio(
    "Méthode d’estimation",
    options=list(method_options.keys()),
    index=list(method_options.keys()).index(prev_label),
    horizontal=True,
)

method_name = method_options[method_label]

inner_solver = st.selectbox(
    "Inner solver",
    options=["default", "scipy"],
    index=0 if cfg_prev.get("inner_solver", "default") == "default" else 1,
    help="Conserver 'default' tant que le backend n’impose pas un solveur spécifique.",
)

verbose = st.checkbox(
    "Mode verbeux",
    value=bool(cfg_prev.get("verbose", False)),
)

if method_name == "random_search":
    c1, c2 = st.columns(2)
    with c1:
        n_iter = st.number_input(
            "Nombre d’itérations",
            min_value=10,
            max_value=100000,
            value=int(cfg_prev.get("n_iter", 500)),
            step=10,
        )
    with c2:
        seed = st.number_input(
            "Seed",
            min_value=0,
            value=int(cfg_prev.get("seed", 42)),
            step=1,
        )

    estimation_config = build_estimation_config(
        method_name=method_name,
        n_iter=int(n_iter),
        seed=int(seed),
        inner_solver=inner_solver,
        verbose=verbose,
    )

elif method_name == "bilevel":
    c1, c2, c3 = st.columns(3)
    with c1:
        n_restarts = st.number_input(
            "Nombre de restarts",
            min_value=1,
            max_value=1000,
            value=int(cfg_prev.get("n_restarts", 5)),
            step=1,
        )
    with c2:
        maxiter = st.number_input(
            "Nombre max d’itérations",
            min_value=1,
            max_value=10000,
            value=int(cfg_prev.get("maxiter", 100)),
            step=1,
        )
    with c3:
        seed = st.number_input(
            "Seed",
            min_value=0,
            value=int(cfg_prev.get("seed", 42)),
            step=1,
        )

    method_opt = st.selectbox(
        "Méthode d’optimisation",
        options=["L-BFGS-B", "Nelder-Mead"],
        index=0 if cfg_prev.get("method", "L-BFGS-B") == "L-BFGS-B" else 1,
    )

    estimation_config = build_estimation_config(
        method_name=method_name,
        n_restarts=int(n_restarts),
        maxiter=int(maxiter),
        seed=int(seed),
        method=method_opt,
        inner_solver=inner_solver,
        verbose=verbose,
    )

else:
    c1, c2, c3 = st.columns(3)
    with c1:
        val_last_k = st.number_input(
            "Nombre de périodes de validation",
            min_value=1,
            max_value=100,
            value=int(cfg_prev.get("val_last_k", 3)),
            step=1,
        )
    with c2:
        n_iter = st.number_input(
            "Nombre d’itérations",
            min_value=10,
            max_value=100000,
            value=int(cfg_prev.get("n_iter", 300)),
            step=10,
        )
    with c3:
        seed = st.number_input(
            "Seed",
            min_value=0,
            value=int(cfg_prev.get("seed", 42)),
            step=1,
        )

    estimation_config = build_estimation_config(
        method_name=method_name,
        val_last_k=int(val_last_k),
        n_iter=int(n_iter),
        seed=int(seed),
        inner_solver=inner_solver,
        verbose=verbose,
    )

validation = validate_estimation_config(sc_format, estimation_config)

with st.expander("Configuration d’estimation", expanded=False):
    st.json(estimation_config)

if not validation.ok:
    for err in validation.errors:
        st.error(err)

st.divider()

# =========================================================
# SECTION 3 — Exécution & monitoring
# =========================================================

st.subheader("3) Exécution et monitoring live")

progress_bar = st.progress(0)
status_placeholder = st.empty()
metrics_placeholder = st.empty()
figure_placeholder = st.empty()

live_objective_history: list[float] = []
live_inputs = _get_live_plot_inputs(sc_format)

c_run1, c_run2 = st.columns([2, 1])

with c_run1:
    run_clicked = st.button(
        "Lancer l’estimation",
        type="primary",
        use_container_width=True,
        disabled=not validation.ok,
    )

with c_run2:
    clear_clicked = st.button(
        "Effacer estimation courante",
        use_container_width=True,
    )

if clear_clicked:
    st.session_state[keys.ESTIMATION_CONFIG] = None
    invalidate_from_estimation_change()
    progress_bar.progress(0)
    figure_placeholder.empty()
    metrics_placeholder.empty()
    status_placeholder.success("Les sorties d’estimation et leurs dépendances ont été effacées.")


def ui_progress_callback(**snapshot):
    iteration = snapshot.get("iteration")
    n_total = snapshot.get("n_iterations_total")
    method = snapshot.get("method", method_name)
    status_message = snapshot.get("status_message", "")
    objective_value = snapshot.get("objective_value", snapshot.get("loss_current"))
    best_objective = snapshot.get("best_objective_value", snapshot.get("loss_best"))

    if objective_value is not None:
        try:
            live_objective_history.append(float(objective_value))
        except Exception:
            pass

    if iteration is not None and n_total:
        try:
            pct = int(min(100, max(0, round(100 * float(iteration) / float(n_total)))))
            progress_bar.progress(pct)
        except Exception:
            pass

    status_lines = [
        f"**Méthode** : {method}",
        f"**Itération** : {iteration}" + (f" / {n_total}" if n_total is not None else ""),
        f"**Objectif courant** : {objective_value}",
        f"**Meilleur objectif** : {best_objective}",
    ]
    if status_message:
        status_lines.append(f"**Message** : {status_message}")

    status_placeholder.markdown("  \n".join(status_lines))

    with metrics_placeholder.container():
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Méthode", method)
        with m2:
            st.metric(
                "Progression",
                f"{iteration}/{n_total}" if iteration is not None and n_total is not None else (iteration if iteration is not None else "n/a"),
            )
        with m3:
            st.metric(
                "Meilleur objectif",
                best_objective if best_objective is not None else "n/a",
            )

    try:
        live_fit = build_live_fit_summary_like(
            snapshot=snapshot,
            donor_names=live_inputs["donor_names"],
            feature_names=live_inputs["feature_names"],
            y_treated=live_inputs["y_treated"],
            y0_full=live_inputs["y0_full"],
            time_index=live_inputs["time_index"],
            T0=live_inputs["T0"],
            objective_history=live_objective_history,
        )
        fig_live = build_main_fit_figure(live_fit)
        figure_placeholder.pyplot(fig_live, clear_figure=True)
    except Exception as exc:
        figure_placeholder.info(f"Figure live indisponible : {exc}")


if run_clicked:
    try:
        st.session_state[keys.ESTIMATION_CONFIG] = estimation_config
        progress_bar.progress(0)
        figure_placeholder.empty()
        metrics_placeholder.empty()
        status_placeholder.info("Initialisation de l’estimation...")

        with st.spinner("Estimation en cours..."):
            run_and_persist_estimation(
                sc_format=sc_format,
                estimation_config=estimation_config,
                progress_callback=ui_progress_callback,
            )

        progress_bar.progress(100)
        status_placeholder.success("Estimation terminée avec succès.")
        st.success("Estimation terminée avec succès. Les résultats analytiques sont disponibles dans la page 4.")
    except Exception as exc:
        status_placeholder.error(f"Erreur d’estimation : {exc}")
        st.error(f"Erreur d’estimation : {exc}")

st.divider()

# =========================================================
# SECTION 4 — Résumé final d’exécution
# =========================================================

st.subheader("4) Résumé final d’exécution")

estimation_result = st.session_state.get(keys.ESTIMATION_RESULT)

if estimation_result is None:
    st.info("Aucune estimation terminée n’est disponible pour le moment.")
else:
    summary = build_estimation_summary(estimation_result)

    loss_value = summary.get(
        "loss",
        summary.get("objective_value", summary.get("best_objective", None)),
    )
    n_iter_value = summary.get("n_iter", summary.get("iterations", "n/a"))
    status_value = summary.get("status", summary.get("solver_status", "n/a"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Loss / objectif final", loss_value if loss_value is not None else "n/a")
    with c2:
        st.metric("Itérations", n_iter_value)
    with c3:
        st.metric("Statut", status_value)

    with st.expander("Résumé technique d’exécution", expanded=False):
        filtered_summary = {
            "loss": loss_value,
            "n_iter": n_iter_value,
            "status": status_value,
            "solver_status": summary.get("solver_status", None),
            "method": summary.get("method", None),
            "message": summary.get("message", None),
        }
        st.json(filtered_summary)

# =========================================================
# Page-level AI assistant
# =========================================================

st.divider()

render_ai_panel(
    page_name="Estimation",
    available_actions=[
        "Expliquer la méthode d'estimation",
        "Interpréter le statut d'exécution",
    ],
)