

from __future__ import annotations
from typing import Any, Dict, List

from ..core.types import IterationRecord


def new_history() -> List[IterationRecord]:
    return []


def append_history(
    history: List[IterationRecord],
    *,
    stage: str,
    iteration: int,
    loss_current: float,
    loss_best: float,
    elapsed_sec: float | None = None,
    payload: Dict[str, Any] | None = None,
) -> None:
    history.append(
        IterationRecord(
            stage=stage,
            iteration=int(iteration),
            loss_current=float(loss_current),
            loss_best=float(loss_best),
            elapsed_sec=elapsed_sec,
            payload=payload or {},
        )
    )