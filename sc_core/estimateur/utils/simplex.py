

from __future__ import annotations
import numpy as np

from ..core.exceptions import InvalidSimplexError


def clean_simplex_weights(w: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype=float).reshape(-1)

    if w.size == 0:
        raise InvalidSimplexError("Weight vector is empty.")
    if not np.all(np.isfinite(w)):
        raise InvalidSimplexError("Weight vector contains non-finite values.")

    w = np.where(w < tol, 0.0, w)
    w = np.clip(w, 0.0, 1.0)

    s = float(np.sum(w))
    if s <= tol:
        raise InvalidSimplexError("Weight vector sum is non-positive after cleanup.")

    return w / s


def simplex_bounds(J: int) -> list[tuple[float, float]]:
    if J <= 0:
        raise InvalidSimplexError(f"J must be >= 1, got {J}.")
    return [(0.0, 1.0)] * int(J)


def is_on_simplex(w: np.ndarray, tol: float = 1e-8) -> bool:
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size == 0:
        return False
    if not np.all(np.isfinite(w)):
        return False
    if np.any(w < -tol):
        return False
    return abs(float(np.sum(w)) - 1.0) <= tol