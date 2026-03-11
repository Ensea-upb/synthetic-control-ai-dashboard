from __future__ import annotations
from typing import Optional, Protocol
import numpy as np


class OptimizationCallback(Protocol):
    def __call__(
        self,
        *,
        stage: str,
        iteration: int,
        loss_current: float,
        loss_best: float,
        w_current: Optional[np.ndarray] = None,
        Vvar_current: Optional[np.ndarray] = None,
        Vdiag_current: Optional[np.ndarray] = None,
    ) -> None:
        ...