"""Tests for the geocoding module."""

import pytest
import pandas as pd


class TestGecodeFacilities:
    def test_output_has_geometry(self):
        """Result should be a GeoDataFrame with a geometry column."""
        pytest.skip("Not yet implemented")

    def test_crs_is_epsg4326(self):
        """Output CRS should be EPSG:4326."""
        pytest.skip("Not yet implemented")


class TestValidateCoordinates:
    def test_flags_out_of_state(self):
        """Points outside the state boundary should be flagged."""
        pytest.skip("Not yet implemented")
