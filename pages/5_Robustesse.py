# pages/5_Robustesse.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_ui.components.ai_panel import render_ai_panel
from app_ui.controllers.robustness_controller import (
    build_robustness_config,
    validate_robustness_config,
    run_and_persist_robustness,
    get_robustness_results_from_state,
)
from app_ui.services.robustness_service import (
    build_placebo_figure,
    build_rmspe_distribution_figure,
    build_rmspe_ratio_figure,
    build_leave_one_out_figure,
    build_backdating_figure,
    build_backdating_ratio_figure,
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

st.set_page_config(page_title="Robustesse | Synthetic Control App", layout="wide")
initialize_app_state()
st.session_state[keys.CURRENT_PAGE] = "Robustesse"
recompute_workflow_from_state()

if render_workflow_sidebar is not None:
    render_workflow_sidebar()

st.title("6) Robustesse — Tests de validité du Synthetic Control")
st.caption(
    "Placebo space, Leave-one-out, Backdating et RMSPE pour valider l'effet estimé."
)

# =========================================================
# Pre-flight checks
# =========================================================

sc_format = st.session_state.get(keys.SC_FORMAT)
estimation_result = st.session_state.get(keys.ESTIMATION_RESULT)
estimation_config = st.session_state.get(keys.ESTIMATION_CONFIG)
fit_summary_data = st.session_state.get(keys.FIT_SUMMARY_DATA)
df_raw = st.session_state.get(keys.DF_RAW)

if sc_format is None or estimation_result is None:
    st.warning("⚠️ Il faut d'abord construire le SCFormat (page Données) et lancer une estimation (page Estimation).")
    st.stop()

# Extract key quantities from sc_format once
try:
    time_index = np.asarray(sc_format.years)
    T0 = getattr(sc_format, "T0", None)
    treated_unit = str(getattr(sc_format, "treated", "traité"))
except Exception as e:
    st.error(f"Impossible de lire sc_format : {e}")
    st.stop()

# =========================================================
# Configuration section
# =========================================================

cfg_prev = st.session_state.get(keys.ROBUSTNESS_CONFIG) or {}

st.subheader("Sélection des tests de robustesse")

c1, c2, c3, c4 = st.columns(4)
with c1:
    run_placebo = st.checkbox(
        "🎲 Placebo space",
        value=cfg_prev.get("run_placebo", True),
        help="Applique le même estimateur à chaque unité de contrôle comme si elle était traitée.",
    )
with c2:
    run_leave_one_out = st.checkbox(
        "🔍 Leave-one-out",
        value=cfg_prev.get("run_leave_one_out", True),
        help="Retire chaque donneur actif un par un et réestime le contrôle synthétique.",
    )
with c3:
    run_backdating = st.checkbox(
        "📅 Backdating",
        value=cfg_prev.get("run_backdating", True),
        help="Déplace artificiellement T0 dans le pré-traitement pour tester les fausses alarmes.",
    )
with c4:
    run_rmspe = st.checkbox(
        "📊 RMSPE",
        value=cfg_prev.get("run_rmspe", True),
        help="Calcule le ratio RMSPE post/pré pour tous les placebos (requiert Placebo space).",
    )

robustness_config = build_robustness_config(
    run_placebo=run_placebo,
    run_leave_one_out=run_leave_one_out,
    run_backdating=run_backdating,
    run_rmspe=run_rmspe,
)

validation = validate_robustness_config(
    df=df_raw,
    sc_format=sc_format,
    estimation_result=estimation_result,
    estimation_config=estimation_config,
    robustness_config=robustness_config,
)

if not validation.ok:
    for err in validation.errors:
        st.error(err)

with st.expander("Voir la configuration robustesse", expanded=False):
    st.json(robustness_config)

col_run, col_clear = st.columns([3, 1])
with col_run:
    run_clicked = st.button(
        "▶ Lancer les tests de robustesse",
        type="primary",
        use_container_width=True,
        disabled=not validation.ok,
    )
with col_clear:
    clear_clicked = st.button("🗑 Effacer", use_container_width=True)

if clear_clicked:
    st.session_state[keys.ROBUSTNESS_CONFIG] = None
    st.session_state[keys.ROBUSTNESS_RESULTS] = None
    st.session_state[keys.FIGURE_CACHE] = {}
    recompute_workflow_from_state()
    st.success("Résultats de robustesse effacés.")
    st.rerun()

if run_clicked:
    progress = st.progress(0, text="Initialisation des tests...")
    try:
        run_and_persist_robustness(
            df=df_raw,
            sc_format=sc_format,
            estimation_result=estimation_result,
            estimation_config=estimation_config,
            robustness_config=robustness_config,
            fit_summary_data=fit_summary_data,
        )
        progress.progress(100, text="Tests terminés ✅")
        st.success("Tests de robustesse terminés avec succès.")
        st.rerun()
    except Exception as exc:
        st.error(f"Erreur lors des tests de robustesse : {exc}")
        import traceback
        with st.expander("Détail de l'erreur"):
            st.code(traceback.format_exc())

# =========================================================
# Retrieve results
# =========================================================

results = get_robustness_results_from_state()

placebo_result = results.get("placebo_result")
leave_one_out_result = results.get("leave_one_out_result")
backdating_result = results.get("backdating_result")
rmspe_metrics = results.get("rmspe_metrics")
rmspe_ratio_series = results.get("rmspe_ratio_series")

if not any(v is not None for v in results.values()):
    st.info("Lance les tests ci-dessus pour afficher les résultats.")
    st.stop()

# =========================================================
# SECTION 1 — Placebo space
# =========================================================

if placebo_result is not None:
    st.divider()
    st.subheader("1) Test Placebo (espace)")
    st.markdown(
        """
        Le test placebo attribue fictivement le traitement à chaque unité de contrôle.
        Si l'effet estimé sur l'unité traitée (rouge) est nettement plus grand
        que sur les placebos (gris), le résultat est statistiquement significatif.
        """
    )

    col_info1, col_info2, col_info3 = st.columns(3)
    n_placebos = len(getattr(placebo_result, "placebo_units", []))
    with col_info1:
        st.metric("Unités placebo", n_placebos)
    with col_info2:
        st.metric("Unité traitée", treated_unit)
    with col_info3:
        st.metric("T0", str(T0) if T0 is not None else "N/A")

    try:
        fig = build_placebo_figure(
            placebo_result=placebo_result,
            sc_format=sc_format,
            estimation_result=estimation_result,
        )
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"Figure placebo indisponible : {exc}")

    with st.expander("Données brutes placebo", expanded=False):
        rmspe_info = getattr(placebo_result, "rmspe_info", None)
        if rmspe_info:
            rows = []
            for unit, info in rmspe_info.items():
                rows.append({
                    "Unité": unit,
                    "RMSPE pré": round(float(info.get("pre_rmspe", 0)), 4),
                    "RMSPE post": round(float(info.get("post_rmspe", 0)), 4),
                    "Ratio post/pré": round(float(info.get("ratio", 0)), 4),
                })
            st.dataframe(pd.DataFrame(rows).sort_values("Ratio post/pré", ascending=False),
                         use_container_width=True)

# =========================================================
# SECTION 2 — RMSPE
# =========================================================

if rmspe_metrics is not None or rmspe_ratio_series is not None:
    st.divider()
    st.subheader("2) RMSPE — Ratio post / pré traitement")
    st.markdown(
        """
        Le ratio RMSPE post/pré mesure si l'écart entre traité et synthétique
        est plus grand après le traitement qu'avant. Un ratio élevé pour l'unité
        traitée, et faible pour les placebos, est une preuve de significativité.
        """
    )

    col_rmspe1, col_rmspe2 = st.columns(2)

    with col_rmspe1:
        st.markdown("**Distribution RMSPE pré / post**")
        if rmspe_metrics is not None:
            try:
                fig = build_rmspe_distribution_figure(rmspe_metrics)
                st.pyplot(fig)
            except Exception as exc:
                st.info(f"Figure distribution RMSPE indisponible : {exc}")
        else:
            st.info("Données RMSPE indisponibles.")

    with col_rmspe2:
        st.markdown("**Classement des ratios par unité**")
        if rmspe_ratio_series is not None:
            try:
                # Inject treated unit's own ratio if available
                treated_ratio_series = rmspe_ratio_series
                if rmspe_metrics is not None and isinstance(rmspe_metrics, dict):
                    # The treated unit ("ville_traite") is in rmspe_info if placebo ran
                    # but we need to add the real treated unit ratio
                    pass

                fig = build_rmspe_ratio_figure(
                    rmspe_ratio_series,
                    treated_unit=getattr(placebo_result, "treated_unit", None) if placebo_result else None,
                )
                st.pyplot(fig)
            except Exception as exc:
                st.info(f"Figure ratio RMSPE indisponible : {exc}")

    # p-value interpretation
    if rmspe_ratio_series is not None and isinstance(rmspe_ratio_series, pd.Series) and len(rmspe_ratio_series) > 0:
        with st.expander("Interprétation statistique", expanded=True):
            n_total = len(rmspe_ratio_series)
            if rmspe_metrics and isinstance(rmspe_metrics, dict):
                # Find rank of treated in placebo ratios
                placebo_ratios = [float(info.get("ratio", 0)) for info in rmspe_metrics.values()]
                if fit_summary_data is not None:
                    gap_arr = np.asarray(fit_summary_data.gap, dtype=float)
                    pre_mask = time_index < T0 if T0 is not None else np.ones(len(time_index), bool)
                    post_mask = time_index >= T0 if T0 is not None else np.ones(len(time_index), bool)
                    pre_rmspe = float(np.sqrt(np.mean(gap_arr[pre_mask] ** 2))) if pre_mask.sum() > 0 else 0
                    post_rmspe = float(np.sqrt(np.mean(gap_arr[post_mask] ** 2))) if post_mask.sum() > 0 else 0
                    treated_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf")
                    rank = sum(1 for r in placebo_ratios if r >= treated_ratio)
                    p_val = rank / (n_total + 1)

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("RMSPE pré (traité)", f"{pre_rmspe:.4f}")
                    with c2:
                        st.metric("RMSPE post (traité)", f"{post_rmspe:.4f}")
                    with c3:
                        st.metric("Ratio (traité)", f"{treated_ratio:.2f}")
                    with c4:
                        st.metric("p-valeur placebo", f"{p_val:.3f}")

                    if p_val <= 0.1:
                        st.success(f"✅ L'effet est statistiquement significatif (p ≈ {p_val:.3f} ≤ 0.10).")
                    else:
                        st.warning(f"⚠️ L'effet n'est pas significatif au seuil de 10% (p ≈ {p_val:.3f}).")

# =========================================================
# SECTION 3 — Leave-one-out
# =========================================================

if leave_one_out_result is not None:
    st.divider()
    st.subheader("3) Leave-one-out (LOO)")
    st.markdown(
        """
        Le test LOO retire chaque donneur actif un par un pour vérifier que
        le contrôle synthétique ne dépend pas d'un seul donneur.
        Si toutes les courbes restent proches de la baseline (rouge), l'estimateur est robuste.
        """
    )

    dropped = getattr(leave_one_out_result, "dropped_donors", [])
    gaps_by_donor = getattr(leave_one_out_result, "gaps_by_donor", {})

    # Guard: LOO skipped when only 1 active donor (cannot drop without leaving 0 donors)
    if len(dropped) == 0:
        st.info(
            "ℹ️ Leave-one-out non exécuté : le modèle ne possède qu'un seul donneur actif "
            "(ou tous les poids sont nuls). "
            "Il faut au moins 2 donneurs actifs pour effectuer le test LOO."
        )

    col_loo1, col_loo2 = st.columns(2)
    with col_loo1:
        st.metric("Donneurs actifs retirés", len(dropped))
    with col_loo2:
        st.metric("Unité traitée", treated_unit)

    try:
        fig_loo = build_leave_one_out_figure(
            leave_one_out_result=leave_one_out_result,
            sc_format=sc_format,
            estimation_result=estimation_result,
        )
        st.pyplot(fig_loo)
    except Exception as exc:
        st.error(f"Figure LOO indisponible : {exc}")

    with st.expander("Tableau récapitulatif LOO", expanded=False):
        if gaps_by_donor:
            rows = []
            base_gap = None
            if fit_summary_data is not None:
                base_gap = np.asarray(fit_summary_data.gap, dtype=float)

            for donor, gap_arr in gaps_by_donor.items():
                gap_arr = np.asarray(gap_arr, dtype=float)
                post_mask = time_index >= T0 if T0 is not None else np.ones(len(time_index), bool)
                if len(gap_arr) == len(time_index):
                    post_rmspe = float(np.sqrt(np.mean(gap_arr[post_mask] ** 2))) if post_mask.sum() > 0 else float("nan")
                else:
                    post_rmspe = float("nan")
                rows.append({"Donneur retiré": donor, "RMSPE post (LOO)": round(post_rmspe, 4)})

            if base_gap is not None:
                post_mask = time_index >= T0 if T0 is not None else np.ones(len(time_index), bool)
                base_post_rmspe = float(np.sqrt(np.mean(base_gap[post_mask] ** 2))) if post_mask.sum() > 0 else float("nan")
                rows.insert(0, {"Donneur retiré": "— Baseline —", "RMSPE post (LOO)": round(base_post_rmspe, 4)})

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =========================================================
# SECTION 4 — Backdating
# =========================================================

if backdating_result is not None:
    st.divider()
    st.subheader("4) Backdating (faux T0)")
    st.markdown(
        """
        Le backdating place artificiellement T0 à différents moments dans le pré-traitement.
        Si aucun « effet » significatif n'est détecté pour les faux T0, cela confirme que
        l'effet détecté au vrai T0 n'est pas dû au hasard.
        """
    )

    runs = getattr(backdating_result, "runs", [])
    st.metric("Nombre de pseudo-T0 testés", len(runs))

    col_bd1, col_bd2 = st.columns(2)

    with col_bd1:
        st.markdown("**Gaps sous différents pseudo-T0**")
        try:
            fig_bd = build_backdating_figure(
                backdating_result=backdating_result,
                sc_format=sc_format,
            )
            st.pyplot(fig_bd)
        except Exception as exc:
            st.error(f"Figure backdating indisponible : {exc}")

    with col_bd2:
        st.markdown("**Ratio RMSPE post/pré par pseudo-T0**")
        try:
            fig_bd_ratio = build_backdating_ratio_figure(backdating_result)
            st.pyplot(fig_bd_ratio)
        except Exception as exc:
            st.error(f"Figure ratio backdating indisponible : {exc}")

    with st.expander("Tableau récapitulatif backdating", expanded=False):
        if runs:
            rows = []
            for run in runs:
                gap_arr = np.asarray(run.gaps, dtype=float)
                n_pre = len(run.pre_periods_used)
                pre_rmspe = float(np.sqrt(np.mean(gap_arr[:n_pre] ** 2))) if n_pre > 0 else float("nan")
                post_rmspe = float(np.sqrt(np.mean(gap_arr[n_pre:] ** 2))) if len(gap_arr) > n_pre else float("nan")
                ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf")
                rows.append({
                    "Pseudo-T0": run.pseudo_t0,
                    "Périodes pré utilisées": len(run.pre_periods_used),
                    "RMSPE pré": round(pre_rmspe, 4),
                    "RMSPE post": round(post_rmspe, 4),
                    "Ratio post/pré": round(ratio, 4),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =========================================================
# SECTION 5 — Tableau de synthèse global
# =========================================================

if any(v is not None for v in results.values()):
    st.divider()
    st.subheader("5) Tableau de synthèse — Validité globale")

    summary_rows = []

    # Placebo p-value
    if rmspe_ratio_series is not None and isinstance(rmspe_ratio_series, pd.Series) and len(rmspe_ratio_series) > 0:
        treated_label = str(treated_unit)
        if fit_summary_data is not None and T0 is not None:
            try:
                gap_arr   = np.asarray(fit_summary_data.gap, dtype=float)
                pre_mask  = time_index < T0
                post_mask = time_index >= T0
                pre_rmspe  = float(np.sqrt(np.mean(gap_arr[pre_mask]  ** 2))) if pre_mask.any()  else 0.0
                post_rmspe = float(np.sqrt(np.mean(gap_arr[post_mask] ** 2))) if post_mask.any() else 0.0
                treated_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf")

                # Placebo ratios (exclude treated from comparison if present)
                placebo_ratios = [
                    float(v) for k, v in rmspe_ratio_series.items()
                    if str(k) != treated_label and np.isfinite(float(v))
                ]
                rank = sum(1 for r in placebo_ratios if r >= treated_ratio)
                p_val = rank / (len(placebo_ratios) + 1) if placebo_ratios else float("nan")

                summary_rows.append({
                    "Test": "Placebo space",
                    "Statistique": f"Ratio RMSPE traité = {treated_ratio:.2f}",
                    "p-valeur": f"{p_val:.3f}" if np.isfinite(p_val) else "N/A",
                    "Verdict": "✅ Significatif" if p_val <= 0.10 else "⚠️ Non significatif",
                })
            except Exception:
                pass

    # LOO stability
    if leave_one_out_result is not None:
        dropped = getattr(leave_one_out_result, "dropped_donors", [])
        gaps_by_donor = getattr(leave_one_out_result, "gaps_by_donor", {})
        if dropped and fit_summary_data is not None and T0 is not None:
            try:
                base_gap  = np.asarray(fit_summary_data.gap, dtype=float)
                post_mask = time_index >= T0
                base_post = float(np.sqrt(np.mean(base_gap[post_mask] ** 2))) if post_mask.any() else 0.0
                ratios_loo = []
                for g in gaps_by_donor.values():
                    g = np.asarray(g, dtype=float)
                    if len(g) == len(time_index):
                        rr = float(np.sqrt(np.mean(g[post_mask] ** 2))) if post_mask.any() else 0.0
                        if base_post > 0:
                            ratios_loo.append(rr / base_post)
                if ratios_loo:
                    max_dev = max(abs(r - 1) for r in ratios_loo) * 100
                    verdict = "✅ Stable" if max_dev < 20 else "⚠️ Sensible"
                    summary_rows.append({
                        "Test": "Leave-one-out",
                        "Statistique": f"Déviation max = {max_dev:.1f}% vs baseline",
                        "p-valeur": "—",
                        "Verdict": verdict,
                    })
            except Exception:
                pass
        elif len(dropped) == 0:
            summary_rows.append({
                "Test": "Leave-one-out",
                "Statistique": "Non exécuté (1 seul donneur actif)",
                "p-valeur": "—",
                "Verdict": "ℹ️ N/A",
            })

    # Backdating
    if backdating_result is not None:
        runs = getattr(backdating_result, "runs", [])
        if runs and fit_summary_data is not None and T0 is not None:
            try:
                gap_arr   = np.asarray(fit_summary_data.gap, dtype=float)
                pre_mask  = time_index < T0
                post_mask = time_index >= T0
                pre_rmspe  = float(np.sqrt(np.mean(gap_arr[pre_mask]  ** 2))) if pre_mask.any()  else 0.0
                post_rmspe = float(np.sqrt(np.mean(gap_arr[post_mask] ** 2))) if post_mask.any() else 0.0
                treated_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("inf")

                alarm_count = 0
                for run in runs:
                    g = np.asarray(run.gaps, dtype=float)
                    n_pre = len(run.pre_periods_used)
                    if len(g) > n_pre and n_pre > 0:
                        bd_pre  = float(np.sqrt(np.mean(g[:n_pre] ** 2)))
                        bd_post = float(np.sqrt(np.mean(g[n_pre:] ** 2)))
                        bd_ratio = bd_post / bd_pre if bd_pre > 0 else float("inf")
                        if bd_ratio > 0.5 * treated_ratio:
                            alarm_count += 1

                verdict = "✅ Pas de fausse alarme" if alarm_count == 0 else f"⚠️ {alarm_count} pseudo-T0 alarmant(s)"
                summary_rows.append({
                    "Test": "Backdating",
                    "Statistique": f"{alarm_count}/{len(runs)} fausses alarmes",
                    "p-valeur": "—",
                    "Verdict": verdict,
                })
            except Exception:
                pass

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    else:
        st.info("Lancez les tests pour afficher le tableau de synthèse.")

# =========================================================
# AI panel
# =========================================================

st.divider()

render_ai_panel(
    page_name="Robustesse",
    available_actions=[
        "Interpréter les résultats de robustesse",
        "Le test placebo est-il significatif ?",
        "Interpréter le ratio RMSPE",
    ],
)