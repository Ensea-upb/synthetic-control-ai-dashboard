from __future__ import annotations

from typing import Optional

from IA_integration.ai_manager import AIManager
from IA_integration.ai_manager import AIResponse


def generate_exploration_chart_comment(
    chart_context: dict,
    user_comment: Optional[str] = None,
    *,
    backend: str = "local",
    model_name: Optional[str] = None,
    fig=None,    # figure matplotlib — active la vision du VLM
) -> AIResponse:
    """
    Génère un commentaire IA pour un graphique d'exploration.

    Si `fig` est fourni et que le backend local est actif,
    le modèle VLM reçoit l'image du graphique en plus du contexte texte.
    """
    manager = AIManager(backend=backend, model_name=model_name)
    return manager.ask(
        task="exploration_chart_comment",
        context=chart_context,
        user_message=user_comment,
        fig=fig,
    )