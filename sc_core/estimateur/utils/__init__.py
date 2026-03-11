from .data_prep import build_prepared_matrices
from .grouping import build_group_index, expand_group_weights
from .history import new_history, append_history
from .scaling import normalize_nonnegative, safe_softmax
from .simplex import clean_simplex_weights, simplex_bounds, is_on_simplex

__all__ = [
    "build_prepared_matrices",
    "build_group_index",
    "expand_group_weights",
    "new_history",
    "append_history",
    "normalize_nonnegative",
    "safe_softmax",
    "clean_simplex_weights",
    "simplex_bounds",
    "is_on_simplex",
]