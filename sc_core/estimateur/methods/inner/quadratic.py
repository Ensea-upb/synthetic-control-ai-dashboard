from __future__ import annotations
import numpy as np

from ...core.exceptions import ShapeMismatchError
from ...utils.arrays import as_1d_float, as_2d_float, ensure_finite
from ...utils.scaling import normalize_nonnegative


def build_qp_from_outcome_only(
    Y1_pre: np.ndarray,
    Y0_pre: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Y1 = as_1d_float(Y1_pre, "Y1_pre")
    Y0 = as_2d_float(Y0_pre, "Y0_pre")

    if Y0.shape[0] != Y1.shape[0]:
        raise ShapeMismatchError("Y0_pre and Y1_pre must have same number of rows.")

    ensure_finite(Y1, "Y1_pre")
    ensure_finite(Y0, "Y0_pre")

    Q = 2.0 * (Y0.T @ Y0)
    p = -2.0 * (Y0.T @ Y1)
    return Q, p


def build_qp_from_xv(
    X1: np.ndarray,
    X0: np.ndarray,
    Vdiag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X1 = as_1d_float(X1, "X1")
    X0 = as_2d_float(X0, "X0")
    Vdiag = normalize_nonnegative(Vdiag)

    K, _ = X0.shape
    if X1.shape[0] != K:
        raise ShapeMismatchError("X1 and X0 are not conformable.")
    if Vdiag.shape[0] != K:
        raise ShapeMismatchError("Vdiag length must match number of rows of X0.")

    ensure_finite(X1, "X1")
    ensure_finite(X0, "X0")
    ensure_finite(Vdiag, "Vdiag")

    sqrtV = np.sqrt(Vdiag)
    X0w = X0 * sqrtV[:, None]
    X1w = X1 * sqrtV

    Q = 2.0 * (X0w.T @ X0w)
    p = -2.0 * (X0w.T @ X1w)
    return Q, p