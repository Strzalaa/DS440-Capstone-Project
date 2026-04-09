"""Tests for the spatial_analysis module."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src.spatial_analysis import (
    compute_drive_times,
    e2sfca,
    exponential_decay,
    gaussian_decay,
    linear_decay,
)


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


class TestNearestFacilityFallback:
    def test_e2sfca_reports_nearest_even_when_no_facility_in_catchment(self):
        tracts = gpd.GeoDataFrame(
            {
                "geoid": ["t1"],
                "total_population": [1000],
                "urbanicity": ["rural"],
            },
            geometry=[Point(-76.0, 41.0)],
            crs="EPSG:4326",
        )
        facilities = gpd.GeoDataFrame(
            {
                "facility_id": ["f1"],
                "provider_count": [1.0],
            },
            geometry=[Point(-76.7, 41.4)],
            crs="EPSG:4326",
        )
        drive_times = compute_drive_times(
            {},
            origins=tracts,
            destinations=facilities,
            max_minutes=30.0,
            urbanicity=tracts["urbanicity"],
        )
        assert drive_times.empty

        result = e2sfca(tracts, facilities, drive_times)
        assert pd.notna(result["nearest_facility_min"].iloc[0])
        assert result["nearest_facility_min"].iloc[0] > 30.0
        assert result["facilities_in_catchment"].iloc[0] == 0
