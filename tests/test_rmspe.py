"""Unit tests for sc_core.estimateur.robustness.rmspe."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sc_core.estimateur.robustness.rmspe import (
    RMSPEResult,
    compute_rmspe_metrics,
    compute_rmspe_ratio_series,
)


class TestComputeRmspeMetrics:
    def _make_arrays(self, n=10):
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(n)
        y_synth = rng.standard_normal(n)
        pre_idx = np.arange(n // 2)
        post_idx = np.arange(n // 2, n)
        return y_true, y_synth, pre_idx, post_idx

    def test_returns_rmspe_result(self):
        y_true, y_synth, pre_idx, post_idx = self._make_arrays()
        result = compute_rmspe_metrics(
            y_true=y_true, y_synth=y_synth, pre_idx=pre_idx, post_idx=post_idx
        )
        assert isinstance(result, RMSPEResult)

    def test_perfect_pre_fit_zero_rmspe(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pre_idx  = np.array([0, 1, 2])
        post_idx = np.array([3, 4])
        result = compute_rmspe_metrics(
            y_true=y, y_synth=y, pre_idx=pre_idx, post_idx=post_idx
        )
        assert result.pre_rmspe == pytest.approx(0.0)
        assert result.post_rmspe == pytest.approx(0.0)

    def test_ratio_equals_post_over_pre(self):
        y_true  = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
        # pre (idx 0,1): error 0 each → pre_rmspe = 0 would cause inf
        # use non-zero pre error
        y_synth = np.array([0.0, 1.0, 3.0, 5.0, 7.0])
        pre_idx  = np.array([0, 1])   # errors: [1, 1] → rmspe = 1.0
        post_idx = np.array([2, 3, 4])  # errors: [0, 0, 0] → rmspe = 0.0
        result = compute_rmspe_metrics(
            y_true=y_true, y_synth=y_synth, pre_idx=pre_idx, post_idx=post_idx
        )
        assert result.pre_rmspe  == pytest.approx(1.0)
        assert result.post_rmspe == pytest.approx(0.0)
        assert result.ratio      == pytest.approx(0.0)

    def test_zero_pre_rmspe_positive_post_gives_inf(self):
        y = np.arange(5, dtype=float)
        pre_idx  = np.array([0, 1])
        post_idx = np.array([2, 3, 4])
        # perfect pre-fit, non-zero post error
        y_synth = y.copy()
        y_synth[2:] -= 2.0
        result = compute_rmspe_metrics(
            y_true=y, y_synth=y_synth, pre_idx=pre_idx, post_idx=post_idx
        )
        assert result.ratio == float("inf")

    def test_gaps_shape(self):
        y_true, y_synth, pre_idx, post_idx = self._make_arrays(n=8)
        result = compute_rmspe_metrics(
            y_true=y_true, y_synth=y_synth, pre_idx=pre_idx, post_idx=post_idx
        )
        assert result.gaps.shape == (8,)


class TestComputeRmspeRatioSeries:
    def test_basic(self):
        data = {
            "A": {"ratio": 3.0, "pre_rmspe": 1.0, "post_rmspe": 3.0},
            "B": {"ratio": 1.5, "pre_rmspe": 2.0, "post_rmspe": 3.0},
            "C": {"ratio": 5.0, "pre_rmspe": 1.0, "post_rmspe": 5.0},
        }
        s = compute_rmspe_ratio_series(data)
        assert isinstance(s, pd.Series)
        assert s.name == "rmspe_ratio"
        assert s.index[0] == "C"  # highest ratio first
        assert s["A"] == pytest.approx(3.0)

    def test_empty(self):
        s = compute_rmspe_ratio_series({})
        assert len(s) == 0

    def test_sorted_descending(self):
        data = {
            "X": {"ratio": 1.0},
            "Y": {"ratio": 4.0},
            "Z": {"ratio": 2.5},
        }
        s = compute_rmspe_ratio_series(data)
        assert list(s.values) == sorted(s.values, reverse=True)
