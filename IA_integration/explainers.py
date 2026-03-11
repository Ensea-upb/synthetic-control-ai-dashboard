# IA_integration/explainers.py

from __future__ import annotations

from typing import Optional

from .ai_manager import AIManager, AIResponse
from .context_builder import build_page_context


def explain_page(
    page_name: str,
    task: str,
    user_message: Optional[str] = None,
    backend: str = "local",
    model_name: Optional[str] = None,
) -> AIResponse:
    manager = AIManager(backend=backend, model_name=model_name)
    context = build_page_context(page_name=page_name)
    return manager.ask(task=task, context=context, user_message=user_message)