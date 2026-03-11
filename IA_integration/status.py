# IA_integration/status.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AIStatus:
    backend: str = "none"
    model_name: Optional[str] = None
    state: str = "disabled"   # disabled, unknown, loading, ready, error
    message: str = ""