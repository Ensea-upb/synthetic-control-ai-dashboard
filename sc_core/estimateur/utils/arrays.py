

from __future__ import annotations
import numpy as np

from ..core.exceptions import ShapeMismatchError


def as_1d_float(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    return arr


def as_2d_float(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ShapeMismatchError(f"{name} must be 2D, got ndim={arr.ndim}.")
    return arr


def ensure_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ShapeMismatchError(f"{name} contains non-finite values.")


def ensure_same_n_rows(x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
    if x.shape[0] != y.shape[0]:
        raise ShapeMismatchError(
            f"{x_name} and {y_name} must have same number of rows. "
            f"Got {x.shape[0]} and {y.shape[0]}."
        )


def ensure_same_n_cols(x: np.ndarray, y: np.ndarray, x_name: str, y_name: str) -> None:
    if x.shape[1] != y.shape[1]:
        raise ShapeMismatchError(
            f"{x_name} and {y_name} must have same number of columns. "
            f"Got {x.shape[1]} and {y.shape[1]}."
        )