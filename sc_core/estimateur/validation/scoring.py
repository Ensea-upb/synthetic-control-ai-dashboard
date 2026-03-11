

from __future__ import annotations
import numpy as np


def mspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_true - y_pred
    return float(np.mean(err ** 2))


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mspe(y_true, y_pred)))


def gap(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return y_true - y_pred