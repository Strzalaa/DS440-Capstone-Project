"""Tests for the data_collection module."""

import pytest

from src.config import STATE_FIPS


class TestFetchAcsData:
    def test_returns_dataframe(self):
        """fetch_acs_data should return a pandas DataFrame."""
        pytest.skip("Requires Census API key — run manually")

    def test_correct_state_fips(self):
        """All returned tracts should belong to the configured state."""
        pytest.skip("Requires Census API key — run manually")


class TestDownloadTigerShapefiles:
    def test_returns_tracts_and_roads(self):
        """Should return a dict with 'tracts' and 'roads' keys."""
        pytest.skip("Requires network access — run manually")
