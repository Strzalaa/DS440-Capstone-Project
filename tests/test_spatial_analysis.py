"""Tests for the spatial_analysis module."""

import numpy as np
import pytest

from src.spatial_analysis import gaussian_decay, linear_decay, exponential_decay


class TestGaussianDecay:
    def test_zero_distance_returns_one(self):
        result = gaussian_decay(np.array([0.0]), threshold=10.0)
        assert np.isclose(result[0], 1.0)

    def test_at_threshold_returns_expected(self):
        result = gaussian_decay(np.array([10.0]), threshold=10.0)
        expected = np.exp(-1.0)
        assert np.isclose(result[0], expected)

    def test_large_distance_approaches_zero(self):
        result = gaussian_decay(np.array([100.0]), threshold=10.0)
        assert result[0] < 1e-10


class TestLinearDecay:
    def test_zero_distance_returns_one(self):
        result = linear_decay(np.array([0.0]), threshold=10.0)
        assert np.isclose(result[0], 1.0)

    def test_at_threshold_returns_zero(self):
        result = linear_decay(np.array([10.0]), threshold=10.0)
        assert np.isclose(result[0], 0.0)

    def test_beyond_threshold_clipped_to_zero(self):
        result = linear_decay(np.array([15.0]), threshold=10.0)
        assert result[0] == 0.0


class TestExponentialDecay:
    def test_zero_distance_returns_one(self):
        result = exponential_decay(np.array([0.0]), threshold=10.0)
        assert np.isclose(result[0], 1.0)

    def test_monotonically_decreasing(self):
        distances = np.array([0.0, 5.0, 10.0, 20.0])
        result = exponential_decay(distances, threshold=10.0)
        assert all(result[i] > result[i + 1] for i in range(len(result) - 1))
