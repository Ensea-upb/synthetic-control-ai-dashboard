# app_ui/components/ai_panel.py
from __future__ import annotations

from typing import Iterable, Optional

import streamlit as st

from app_ui.state import keys
from IA_integration.ai_manager import AIManager
from IA_integration.context_builder import build_page_context


# Correspondance action rapide → clé de tâche
_TASK_MAP = {
    "Expliquer le Synthetic Control":          "data_config",
    "Que dois-je faire ensuite ?":             "free",
    "Expliquer ma configuration":              "data_config",
    "Interpréter l'exploration":               "exploration",
    "Expliquer la méthode d'estimation":       "estimation",
    "Interpréter le statut d'exécution":       "estimation",
    "Interpréter les résultats":               "results",
    "Analyser la qualité du fit":              "fit_quality",
    "Commenter les poids des donneurs":        "donor_weights",
    "Évaluer la significativité du ratio RMSPE": "rmspe_ratio",
    "Interpréter la robustesse":               "robustness",
    "Interpréter les résultats de robustesse": "robustness",
    "Le test placebo est-il significatif ?":   "robustness",
    "Interpréter le ratio RMSPE":              "rmspe_ratio",
}


def _model_loading_banner() -> None:
    """Affiche un bandeau de chargement tant que le modèle n'est pas prêt."""
    try:
        from IA_integration.model_loader import get_local_ai_manager
        from sc_core.IA_integration.codes.status import AIStatus as CoreStatus

        # Vérifie le cache sans déclencher un chargement bloquant
        # (get_local_ai_manager est appelé ici pour la première fois si besoin)
        manager = get_local_ai_manager()
        status = manager.status()

        if status == CoreStatus.LOADING:
            st.info("⏳ Chargement du modèle IA en cours… cela peut prendre 1-2 minutes.")
        elif status == CoreStatus.ERROR:
            err = manager.error()
            st.warning(f"⚠️ Modèle IA indisponible : {err}")
        elif status == CoreStatus.READY:
            st.caption("✅ Modèle local prêt.")
    except Exception:
        pass   # Le banner est optionnel, ne pas bloquer la page


def render_ai_panel(
    page_name: str,
    available_actions: Optional[Iterable[str]] = None,
    title: str = "Assistant IA",
    fig=None,    # passer une figure matplotlib pour activer la vision
) -> None:
    """
    Panel IA réutilisable sur toutes les pages.

    Parameters
    ----------
    page_name         : identifiant logique de la page
    available_actions : actions rapides affichées en selectbox
    title             : titre du panel
    fig               : figure matplotlib (active la branche VLM vision)
    """
    st.subheader(title)

    ai_enabled = st.session_state.get(keys.AI_ENABLED, True)
    ai_backend = st.session_state.get(keys.AI_BACKEND, "local")

    if not ai_enabled:
        st.info("L'assistant IA est désactivé.")
        return

    if ai_backend == "none":
        st.info("Backend IA désactivé. Modifiez le paramètre AI_BACKEND pour l'activer.")
        return

    # ---- Statut modèle ----
    manager = AIManager(backend=ai_backend)
    status = manager.get_status()

    status_icons = {
        "ready":    "🟢",
        "loading":  "🟡",
        "idle":     "⚪",
        "error":    "🔴",
        "disabled": "⚫",
    }
    icon = status_icons.get(status.state, "❓")
    st.caption(f"{icon} Backend : **{status.backend}** | Modèle : {status.model_name} | État : {status.state}")

    if status.state == "loading":
        st.info("⏳ Chargement du modèle en cours… Réessayez dans quelques instants.")
        return

    if status.state == "error":
        st.error(f"❌ {status.message}")
        return

    # ---- Contexte ----
    context = build_page_context(page_name=page_name)
    if keys.AI_CONTEXT_CACHE not in st.session_state:
        st.session_state[keys.AI_CONTEXT_CACHE] = {}
    st.session_state[keys.AI_CONTEXT_CACHE][page_name] = context

    # ---- Actions rapides ----
    if available_actions:
        action = st.selectbox(
            "Action rapide",
            options=["Aucune"] + list(available_actions),
            index=0,
            key=f"ai_action_{page_name}",
        )
    else:
        action = "Aucune"

    # ---- Question libre ----
    user_prompt = st.text_area(
        "Question à l'assistant",
        value="",
        height=100,
        key=f"ai_prompt_{page_name}",
        placeholder="Ex: Interprète la qualité du fit pré-traitement.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        ask_clicked = st.button(
            "🤖 Lancer l'analyse IA",
            key=f"ai_run_{page_name}",
            use_container_width=True,
            disabled=(status.state != "ready"),
        )
    with col2:
        show_ctx = st.button(
            "🔍 Voir le contexte",
            key=f"ai_ctx_{page_name}",
            use_container_width=True,
        )

    if show_ctx:
        with st.expander("Contexte transmis au modèle", expanded=True):
            st.json(context)

    # ---- Réponse précédente ----
    prev_response = st.session_state.get(f"ai_resp_{page_name}")
    if prev_response:
        with st.expander("Dernière réponse IA", expanded=True):
            st.markdown(prev_response)

    # ---- Appel modèle ----
    if ask_clicked:
        task = _TASK_MAP.get(action, "free")
        effective_message = user_prompt.strip() if user_prompt.strip() else (
            action if action != "Aucune" else ""
        )

        with st.spinner("Analyse IA en cours…"):
            response = manager.ask(
                task=task,
                context=context,
                user_message=effective_message or None,
                fig=fig,
            )

        if response.ok:
            st.session_state[keys.AI_LAST_RESPONSE] = response.content
            st.session_state[f"ai_resp_{page_name}"] = response.content
            st.success("Réponse générée.")
            st.markdown(response.content)
        else:
            st.session_state[keys.AI_LAST_ERROR] = response.error
            st.error(f"Erreur IA : {response.error}")