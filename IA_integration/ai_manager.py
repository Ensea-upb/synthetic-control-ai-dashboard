# IA_integration/ai_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .prompt_builder import build_prompt
from .status import AIStatus


@dataclass
class AIResponse:
    ok: bool
    content: str
    error: Optional[str] = None


class AIManager:
    """
    Façade frontend pour l'assistant IA de l'application SCM.

    Backends disponibles :
    - "local"  → modèle Qwen2.5-VL local via OpenVINO GenAI
    - "none"   → désactivé
    """

    def __init__(self, backend: str = "local", model_name: Optional[str] = None) -> None:
        self.backend = backend
        self.model_name = model_name or "Qwen2.5-VL-7B-Instruct-int4-ov"

    # --------------------------------------------------
    # Statut
    # --------------------------------------------------

    def get_status(self) -> AIStatus:
        if self.backend == "none":
            return AIStatus(
                backend="none",
                model_name=None,
                state="disabled",
                message="Assistant IA désactivé.",
            )

        if self.backend == "local":
            try:
                from sc_core.IA_integration.codes.status import AIStatus as CoreStatus
                manager = self._get_core_manager()
                core_status = manager.status()

                state_map = {
                    CoreStatus.IDLE:    "idle",
                    CoreStatus.LOADING: "loading",
                    CoreStatus.READY:   "ready",
                    CoreStatus.ERROR:   "error",
                }
                state = state_map.get(core_status, "unknown")
                error = manager.error()
                msg = str(error) if error else f"Modèle local : {self.model_name}"

                return AIStatus(
                    backend="local",
                    model_name=self.model_name,
                    state=state,
                    message=msg,
                )
            except Exception as exc:
                return AIStatus(
                    backend="local",
                    model_name=self.model_name,
                    state="error",
                    message=str(exc),
                )

        return AIStatus(
            backend=self.backend,
            model_name=self.model_name,
            state="error",
            message=f"Backend inconnu : {self.backend}",
        )

    # --------------------------------------------------
    # Accès au manager core (via singleton Streamlit)
    # --------------------------------------------------

    def _get_core_manager(self):
        from .model_loader import get_local_ai_manager
        return get_local_ai_manager()

    # --------------------------------------------------
    # Requête principale
    # --------------------------------------------------

    def ask(
        self,
        task: str,
        context: dict,
        user_message: Optional[str] = None,
        fig=None,            # figure matplotlib optionnelle pour vision
    ) -> AIResponse:
        """
        Construit le prompt et appelle le backend configuré.

        Parameters
        ----------
        task         : clé de tâche (data_config, exploration, results, robustness, free, …)
        context      : dictionnaire de contexte construit depuis l'état de session
        user_message : question libre de l'utilisateur
        fig          : figure matplotlib (active la branche vision du VLM si fournie)
        """
        if self.backend == "none":
            return AIResponse(ok=False, content="", error="Backend IA désactivé.")

        try:
            prompt = build_prompt(task=task, context=context, user_message=user_message)

            if self.backend == "local":
                return self._ask_local(prompt=prompt, fig=fig)

            return AIResponse(ok=False, content="", error=f"Backend inconnu : {self.backend}")

        except Exception as exc:
            return AIResponse(ok=False, content="", error=str(exc))

    # --------------------------------------------------
    # Backend local
    # --------------------------------------------------

    def _ask_local(self, prompt: str, fig=None) -> AIResponse:
        try:
            manager = self._get_core_manager()

            if not manager.is_ready():
                status = manager.status()
                err = manager.error()
                if err:
                    return AIResponse(ok=False, content="", error=str(err))
                return AIResponse(
                    ok=False,
                    content="",
                    error=f"Modèle en cours de chargement (état={status.value}). Réessayez dans quelques instants.",
                )

            if fig is not None:
                result = manager.comment_figure(fig=fig, prompt=prompt)
            else:
                result = manager.generate_text(prompt=prompt)

            return AIResponse(ok=True, content=result)

        except Exception as exc:
            return AIResponse(ok=False, content="", error=str(exc))