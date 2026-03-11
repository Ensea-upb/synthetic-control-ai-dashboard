

from __future__ import annotations
import numpy as np


def normalize_nonnegative(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    v = np.clip(v, 0.0, None)

    if v.size == 0:
        return v

    s = float(np.sum(v))
    if (not np.isfinite(s)) or s <= 0:
        return np.ones_like(v) / float(len(v))
    return v / s


def safe_softmax(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(-1)

    if theta.size == 0:
        return theta

    theta = theta - np.max(theta)
    exp_theta = np.exp(theta)
    s = float(np.sum(exp_theta))

    if (not np.isfinite(s)) or s <= 0:
        return np.ones_like(theta) / float(len(theta))
    return exp_theta / s