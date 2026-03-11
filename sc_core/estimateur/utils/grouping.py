

from __future__ import annotations
from typing import Sequence
import numpy as np

from ..core.exceptions import ShapeMismatchError


def build_group_index(row_var: Sequence[str], group_names: Sequence[str]) -> np.ndarray:
    group_names = [str(g) for g in list(group_names)]
    mapping = {g: i for i, g in enumerate(group_names)}

    idx = np.empty(len(row_var), dtype=int)
    for k, var_name in enumerate(row_var):
        key = str(var_name)
        if key not in mapping:
            raise ShapeMismatchError(
                f"Variable '{key}' appears in row_var but not in group_names."
            )
        idx[k] = mapping[key]
    return idx


def expand_group_weights(Vvar: np.ndarray, group_idx: np.ndarray) -> np.ndarray:
    Vvar = np.asarray(Vvar, dtype=float).reshape(-1)
    group_idx = np.asarray(group_idx, dtype=int).reshape(-1)

    if group_idx.size == 0:
        return np.array([], dtype=float)

    max_idx = int(np.max(group_idx))
    if max_idx >= Vvar.size:
        raise ShapeMismatchError(
            f"group_idx requires index {max_idx}, but Vvar has length {Vvar.size}."
        )

    Vdiag = Vvar[group_idx].astype(float)
    s = float(np.sum(Vdiag))
    if (not np.isfinite(s)) or s <= 0:
        return np.ones_like(Vdiag) / float(len(Vdiag))
    return Vdiag / s