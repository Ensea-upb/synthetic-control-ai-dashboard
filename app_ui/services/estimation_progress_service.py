from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class LiveFitSummaryLike:
    time_index: Any
    y_treated: Any
    y_synth: Any
    unit_weights: pd.Series
    covariate_weights: pd.Series
    objective_history: list[float]
    T0: Optional[int]


def build_live_fit_summary_like(
    *,
    snapshot: dict,
    donor_names,
    feature_names,
    y_treated,
    y0_full,
    time_index,
    T0,
    objective_history,
) -> LiveFitSummaryLike:
    raw_w = snapshot.get("weights", snapshot.get("w_current"))
    if raw_w is None:
        raise ValueError("Live snapshot does not contain donor weights.")

    w = np.asarray(raw_w, dtype=float).reshape(-1)
    y0 = np.asarray(y0_full, dtype=float)

    if y0.ndim != 2:
        raise ValueError("y0_full must be a 2D donor outcome matrix.")
    if y0.shape[1] != w.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: y0_full has {y0.shape[1]} donor columns, "
            f"but live weights have length {w.shape[0]}."
        )

    y_synth = y0 @ w

    donor_labels = [str(x) for x in list(donor_names)[: len(w)]]
    if len(donor_labels) < len(w):
        donor_labels.extend([f"donor_{i}" for i in range(len(donor_labels), len(w))])

    unit_weights = pd.Series(
        data=[float(val) for val in w],
        index=donor_labels,
        dtype=float,
    )

    raw_v = snapshot.get("covariate_weights", snapshot.get("Vdiag_current"))
    if raw_v is None:
        covariate_weights = pd.Series(dtype=float)
    else:
        vdiag = np.asarray(raw_v, dtype=float).reshape(-1)

        feature_labels = [str(x) for x in list(feature_names)[: len(vdiag)]]
        if len(feature_labels) < len(vdiag):
            feature_labels.extend(
                [f"feature_{i}" for i in range(len(feature_labels), len(vdiag))]
            )

        covariate_weights = pd.Series(
            data=[float(val) for val in vdiag],
            index=feature_labels,
            dtype=float,
        )

    return LiveFitSummaryLike(
        time_index=time_index,
        y_treated=y_treated,
        y_synth=y_synth,
        unit_weights=unit_weights,
        covariate_weights=covariate_weights,
        objective_history=list(objective_history),
        T0=T0,
    )