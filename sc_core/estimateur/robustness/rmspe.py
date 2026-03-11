

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd

from ..validation.scoring import rmspe, gap


@dataclass
class RMSPEResult:
    pre_rmspe: float
    post_rmspe: float
    ratio: float
    gaps: np.ndarray


def compute_rmspe_metrics(
    *,
    y_true: np.ndarray,
    y_synth: np.ndarray,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
) -> RMSPEResult:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_synth = np.asarray(y_synth, dtype=float).reshape(-1)
    pre_idx = np.asarray(pre_idx, dtype=int)
    post_idx = np.asarray(post_idx, dtype=int)

    g = gap(y_true, y_synth)
    pre = float(rmspe(y_true[pre_idx], y_synth[pre_idx]))
    post = float(rmspe(y_true[post_idx], y_synth[post_idx]))

    if pre <= 0:
        ratio = float("inf") if post > 0 else 1.0
    else:
        ratio = float(post / pre)

    return RMSPEResult(
        pre_rmspe=pre,
        post_rmspe=post,
        ratio=ratio,
        gaps=g,
    )


def compute_rmspe_ratio_series(
    results_by_unit: Dict[str, Dict[str, Any]],
) -> pd.Series:
    """
    results_by_unit structure expected:
    {
        "treated_or_placebo_unit": {
            "pre_rmspe": float,
            "post_rmspe": float,
            "ratio": float,
            ...
        },
        ...
    }
    """
    data = {unit: float(info["ratio"]) for unit, info in results_by_unit.items()}
    return pd.Series(data, name="rmspe_ratio").sort_values(ascending=False)