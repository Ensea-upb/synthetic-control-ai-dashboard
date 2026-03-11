"""Unit tests for sc_core.results.postprocess."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sc_core.results.postprocess import (
    build_cumulative_gap,
    build_gap,
    build_synthetic_series,
    build_unit_weights_series,
    build_covariate_weights_series,
)


class TestBuildSyntheticSeries:
    def test_simple_weighted_sum(self):
        Y0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (T=3, J=2)
        w = np.array([0.6, 0.4])
        result = build_synthetic_series(Y0=Y0, weights=w)
        expected = Y0 @ w
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_donor_weight_one(self):
        Y0 = np.array([[10.0], [20.0], [30.0]])
        w = np.array([1.0])
        result = build_synthetic_series(Y0=Y0, weights=w)
        np.testing.assert_array_almost_equal(result, [10.0, 20.0, 30.0])

    def test_wrong_shape_raises(self):
        Y0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([0.5, 0.3, 0.2])  # 3 weights, 2 donors
        with pytest.raises(ValueError, match="Dimension mismatch"):
            build_synthetic_series(Y0=Y0, weights=w)

    def test_1d_y0_raises(self):
        Y0 = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0])
        with pytest.raises(ValueError, match="2D"):
            build_synthetic_series(Y0=Y0, weights=w)

    def test_weights_sum_not_one_allowed(self):
        """build_synthetic_series doesn't enforce simplex; it's the solver's job."""
        Y0 = np.array([[1.0, 0.0], [0.0, 1.0]])
        w = np.array([2.0, 0.0])
        result = build_synthetic_series(Y0=Y0, weights=w)
        np.testing.assert_array_almost_equal(result, [2.0, 0.0])


class TestBuildGap:
    def test_zero_gap(self):
        y = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(build_gap(y_treated=y, y_synth=y), 0)

    def test_positive_gap(self):
        y_treated = np.array([5.0, 6.0, 7.0])
        y_synth   = np.array([3.0, 3.0, 3.0])
        result = build_gap(y_treated=y_treated, y_synth=y_synth)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            build_gap(y_treated=np.array([1.0, 2.0]), y_synth=np.array([1.0]))


class TestBuildCumulativeGap:
    def test_cumsum(self):
        g = np.array([1.0, 2.0, -1.0])
        result = build_cumulative_gap(g)
        np.testing.assert_array_almost_equal(result, [1.0, 3.0, 2.0])

    def test_all_zeros(self):
        result = build_cumulative_gap(np.zeros(5))
        np.testing.assert_array_almost_equal(result, np.zeros(5))

    def test_monotone_positive(self):
        g = np.ones(4)
        result = build_cumulative_gap(g)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0, 4.0])


class TestBuildUnitWeightsSeries:
    def test_basic(self):
        s = build_unit_weights_series(
            donor_names=["A", "B", "C"],
            weights=np.array([0.2, 0.5, 0.3]),
        )
        assert isinstance(s, pd.Series)
        assert len(s) == 3
        assert s.index.tolist() == ["B", "C", "A"]  # sorted descending

    def test_drop_zeros(self):
        s = build_unit_weights_series(
            donor_names=["A", "B", "C"],
            weights=np.array([0.0, 0.7, 0.3]),
            drop_zeros=True,
        )
        assert "A" not in s.index
        assert len(s) == 2

    def test_no_drop_zeros_by_default(self):
        s = build_unit_weights_series(
            donor_names=["X", "Y"],
            weights=np.array([0.0, 1.0]),
        )
        assert len(s) == 2


class TestBuildCovariateWeightsSeries:
    def test_basic(self):
        s = build_covariate_weights_series(
            group_names=["g1", "g2"],
            Vvar=np.array([0.1, 0.9]),
        )
        assert isinstance(s, pd.Series)
        assert s.iloc[0] == pytest.approx(0.9)

    def test_drop_zeros(self):
        s = build_covariate_weights_series(
            group_names=["a", "b", "c"],
            Vvar=np.array([0.0, 0.5, 0.5]),
            drop_zeros=True,
        )
        assert len(s) == 2
