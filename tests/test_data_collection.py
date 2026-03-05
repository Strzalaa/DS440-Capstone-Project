"""Tests for the data_collection module."""

import pandas as pd
import pytest

from src.data_collection import _normalise_facility_columns, _pick_column, _load_tabular


class TestPickColumn:
    def test_finds_exact_match(self):
        assert _pick_column(["Name", "City", "State"], ["name"]) == "Name"

    def test_finds_first_candidate(self):
        result = _pick_column(["lon", "lng", "x"], ["longitude", "lon", "lng"])
        assert result == "lon"

    def test_returns_none_when_missing(self):
        assert _pick_column(["a", "b", "c"], ["x", "y"]) is None

    def test_case_insensitive(self):
        assert _pick_column(["LATITUDE"], ["latitude"]) == "LATITUDE"


class TestNormaliseFacilityColumns:
    def test_empty_dataframe_returns_all_columns(self):
        result = _normalise_facility_columns(pd.DataFrame(), source="test")
        expected_cols = [
            "facility_id", "facility_name", "address", "city",
            "state", "zip", "latitude", "longitude", "provider_count", "source",
        ]
        assert list(result.columns) == expected_cols
        assert len(result) == 0

    def test_source_label_applied(self):
        df = pd.DataFrame({"Name": ["Hospital A"], "City": ["York"]})
        result = _normalise_facility_columns(df, source="cms")
        assert result["source"].iloc[0] == "cms"
        assert result["facility_id"].iloc[0] == "cms_000000"

    def test_provider_count_defaults_to_one(self):
        df = pd.DataFrame({"Name": ["Clinic"], "latitude": [40.0], "longitude": [-76.0]})
        result = _normalise_facility_columns(df, source="hrsa")
        assert result["provider_count"].iloc[0] == 1.0

    def test_renames_known_columns(self):
        df = pd.DataFrame({
            "provider_name": ["Test"],
            "street_address": ["123 Main St"],
            "provider_city": ["Philly"],
            "provider_state": ["PA"],
            "postal_code": ["19102"],
            "lat": [39.95],
            "lng": [-75.16],
        })
        result = _normalise_facility_columns(df, source="cms")
        assert result["facility_name"].iloc[0] == "Test"
        assert result["address"].iloc[0] == "123 Main St"
        assert result["city"].iloc[0] == "Philly"
        assert abs(result["latitude"].iloc[0] - 39.95) < 0.001

    def test_numeric_coercion(self):
        df = pd.DataFrame({
            "Name": ["A"],
            "latitude": ["not_a_number"],
            "longitude": ["-76.0"],
        })
        result = _normalise_facility_columns(df, source="x")
        assert pd.isna(result["latitude"].iloc[0])
        assert abs(result["longitude"].iloc[0] - (-76.0)) < 0.01
