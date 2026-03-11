"""Unit tests for _safe_changed (pure logic, no Streamlit required)."""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal Streamlit stub so we can import invalidation without a real session
# ---------------------------------------------------------------------------

_st_stub = types.ModuleType("streamlit")
_st_stub.session_state = {}  # type: ignore[attr-defined]
sys.modules.setdefault("streamlit", _st_stub)

# Stub the workflow helper that invalidation imports
_workflow_stub = types.ModuleType("app_ui.state.workflow")
_workflow_stub.recompute_workflow_from_state = lambda: None  # type: ignore[attr-defined]
sys.modules.setdefault("app_ui.state.workflow", _workflow_stub)

# Stub keys so we don't need the full package tree
_keys_stub = types.ModuleType("app_ui.state.keys")
for _k in [
    "SC_FORMAT", "DATA_SUMMARY", "EXPLORATION_SUMMARY", "ESTIMATION_CONFIG",
    "ESTIMATION_RESULT", "FIT_SUMMARY_DATA", "ROBUSTNESS_CONFIG",
    "ROBUSTNESS_RESULTS", "EXPORT_PAYLOAD", "FIGURE_CACHE",
    "EXPLORATION_CONFIGS", "EXPLORATION_RENDERED", "ESTIMATION_STALE",
    "ESTIMATION_INPUT_SNAPSHOT", "DATA_CONFIG",
]:
    setattr(_keys_stub, _k, _k.lower())
sys.modules.setdefault("app_ui.state.keys", _keys_stub)

# Ensure app_ui.state is treated as a package (stub the __init__)
_state_pkg = types.ModuleType("app_ui.state")
_state_pkg.keys = _keys_stub  # type: ignore[attr-defined]
sys.modules.setdefault("app_ui.state", _state_pkg)
sys.modules.setdefault("app_ui", types.ModuleType("app_ui"))

# Load the module inside the app_ui.state package so relative imports resolve
import importlib.util, pathlib

_path = pathlib.Path(__file__).parent.parent / "app_ui" / "state" / "invalidation.py"
_spec = importlib.util.spec_from_file_location(
    "app_ui.state.invalidation",
    _path,
    submodule_search_locations=[],
)
_mod = importlib.util.module_from_spec(_spec)            # type: ignore[arg-type]
_mod.__package__ = "app_ui.state"
sys.modules["app_ui.state.invalidation"] = _mod
_spec.loader.exec_module(_mod)                           # type: ignore[union-attr]

_safe_changed = _mod._safe_changed


class TestSafeChanged:
    # ---- None cases --------------------------------------------------------

    def test_both_none_not_changed(self):
        assert _safe_changed(None, None) is False

    def test_old_none_new_value_changed(self):
        assert _safe_changed(None, 42) is True

    def test_new_none_old_value_changed(self):
        assert _safe_changed(42, None) is True

    # ---- Scalar equality ---------------------------------------------------

    def test_equal_ints(self):
        assert _safe_changed(5, 5) is False

    def test_different_ints(self):
        assert _safe_changed(5, 6) is True

    def test_equal_strings(self):
        assert _safe_changed("abc", "abc") is False

    def test_different_strings(self):
        assert _safe_changed("abc", "xyz") is True

    def test_equal_floats(self):
        assert _safe_changed(3.14, 3.14) is False

    # ---- Numpy arrays — always treat as changed (conservative) -------------

    def test_numpy_arrays_treated_as_changed(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        # Even if values are equal, array comparison returns array → assume changed
        assert _safe_changed(a, b) is True

    def test_numpy_array_vs_none(self):
        a = np.array([1.0])
        assert _safe_changed(a, None) is True
        assert _safe_changed(None, a) is True

    # ---- Pandas Series — always treat as changed (conservative) ------------

    def test_pandas_series_treated_as_changed(self):
        s1 = pd.Series([1, 2, 3])
        s2 = pd.Series([1, 2, 3])
        assert _safe_changed(s1, s2) is True

    # ---- Dicts (data config–like objects) ----------------------------------

    def test_equal_dicts_not_changed(self):
        d1 = {"treated": "Paris", "T0": 2010}
        d2 = {"treated": "Paris", "T0": 2010}
        assert _safe_changed(d1, d2) is False

    def test_different_dicts_changed(self):
        d1 = {"treated": "Paris",  "T0": 2010}
        d2 = {"treated": "London", "T0": 2010}
        assert _safe_changed(d1, d2) is True

    # ---- Objects that raise on comparison ----------------------------------

    class _BadCompare:
        def __eq__(self, other):
            raise TypeError("cannot compare")

        def __ne__(self, other):
            raise TypeError("cannot compare")

    def test_comparison_error_treated_as_changed(self):
        obj = self._BadCompare()
        assert _safe_changed(obj, obj) is True
