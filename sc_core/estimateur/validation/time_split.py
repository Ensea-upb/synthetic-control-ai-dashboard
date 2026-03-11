

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Any
import numpy as np

from ..core.exceptions import InvalidConfigError


@dataclass
class TrainValSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_periods: List[Any]
    val_periods: List[Any]


def split_pre_periods_last_k(
    pre_periods: Sequence[Any],
    val_last_k: int,
) -> TrainValSplit:
    periods = list(pre_periods)
    k = int(val_last_k)

    if len(periods) == 0:
        raise InvalidConfigError("pre_periods cannot be empty.")
    if k <= 0:
        raise InvalidConfigError(f"val_last_k must be >= 1, got {k}.")
    if len(periods) <= k:
        raise InvalidConfigError(
            f"Need len(pre_periods) > val_last_k. Got len={len(periods)}, k={k}."
        )

    train_periods = periods[:-k]
    val_periods = periods[-k:]

    train_idx = np.arange(0, len(train_periods), dtype=int)
    val_idx = np.arange(len(train_periods), len(periods), dtype=int)

    return TrainValSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        train_periods=train_periods,
        val_periods=val_periods,
    )