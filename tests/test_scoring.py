"""Unit tests for sc_core.estimateur.validation.scoring."""
from __future__ import annotations

import numpy as np
import pytest

from sc_core.estimateur.validation.scoring import gap, mspe, rmspe


class TestMspe:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mspe(y, y) == pytest.approx(0.0)

    def test_constant_error(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert mspe(y_true, y_pred) == pytest.approx(1.0)

    def test_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        # errors: [-1, 0, 1], squared: [1, 0, 1], mean: 2/3
        assert mspe(y_true, y_pred) == pytest.approx(2.0 / 3.0)

    def test_accepts_lists(self):
        assert mspe([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)


class TestRmspe:
    def test_perfect(self):
        y = np.array([5.0, 10.0, 15.0])
        assert rmspe(y, y) == pytest.approx(0.0)

    def test_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        expected = np.sqrt(2.0 / 3.0)
        assert rmspe(y_true, y_pred) == pytest.approx(expected)

    def test_nonnegative(self):
        rng = np.random.default_rng(42)
        y_true = rng.random(50)
        y_pred = rng.random(50)
        assert rmspe(y_true, y_pred) >= 0.0


class TestGap:
    def test_zero_gap(self):
        y = np.array([1.0, 2.0, 3.0])
        result = gap(y, y)
        np.testing.assert_array_almost_equal(result, np.zeros(3))

    def test_known_gap(self):
        y_true = np.array([3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(gap(y_true, y_pred), [2.0, 2.0, 2.0])

    def test_negative_gap(self):
        y_true = np.array([1.0, 1.0])
        y_pred = np.array([3.0, 3.0])
        np.testing.assert_array_almost_equal(gap(y_true, y_pred), [-2.0, -2.0])

    def test_returns_1d(self):
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[0.5], [1.5]])
        result = gap(y_true, y_pred)
        assert result.ndim == 1
