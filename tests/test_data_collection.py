"""Tests for the data_collection module."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from src.data_collection import (
    _normalise_facility_columns,
    _pick_column,
    _load_tabular,
    fetch_hifld_hospitals,
    fetch_hrsa_health_centers,
    fetch_hifld_urgent_care,
    fetch_cms_pos,
    _fetch_npi_facilities,
)


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


def _mock_arcgis_response(features):
    """Build a MagicMock matching ``requests.get`` for ArcGIS FeatureServer."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "type": "FeatureCollection",
        "features": features,
        "exceededTransferLimit": False,
    }
    return resp


class TestFetchHifldHospitals:
    def test_parses_geojson_into_normalised_columns(self):
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-77.85, 40.79]},
                "properties": {
                    "NAME": "MOUNT NITTANY MEDICAL CENTER",
                    "ADDRESS": "1800 E PARK AVE",
                    "CITY": "STATE COLLEGE",
                    "STATE": "PA",
                    "ZIP": "16803",
                    "STATUS": "OPEN",
                    "BEDS": 260,
                    "TYPE": "GENERAL ACUTE CARE",
                },
            }
        ]
        with patch(
            "src.data_collection.requests.get",
            return_value=_mock_arcgis_response(features),
        ):
            result = fetch_hifld_hospitals(state="PA")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["facility_name"] == "MOUNT NITTANY MEDICAL CENTER"
        assert row["city"] == "STATE COLLEGE"
        assert abs(row["latitude"] - 40.79) < 0.001
        assert abs(row["longitude"] - (-77.85)) < 0.001
        assert row["provider_count"] == 260.0
        assert row["source"] == "hifld_hospitals"

    def test_mount_nittany_regression(self):
        """Regression: the new HIFLD-based fetcher surfaces Mount Nittany.

        The previous NPI-based fetcher capped at 200 records per taxonomy and
        silently dropped Mount Nittany. This test ensures the HIFLD loader
        keeps it once it is in the feature list.
        """
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-77.85, 40.79]},
                "properties": {
                    "NAME": "MOUNT NITTANY MEDICAL CENTER",
                    "ADDRESS": "1800 E PARK AVE",
                    "CITY": "STATE COLLEGE",
                    "STATE": "PA",
                    "ZIP": "16803",
                    "STATUS": "OPEN",
                    "BEDS": 260,
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-76.88, 40.27]},
                "properties": {
                    "NAME": "HARRISBURG HOSPITAL",
                    "ADDRESS": "111 S FRONT ST",
                    "CITY": "HARRISBURG",
                    "STATE": "PA",
                    "ZIP": "17101",
                    "STATUS": "OPEN",
                    "BEDS": 400,
                },
            },
        ]
        with patch(
            "src.data_collection.requests.get",
            return_value=_mock_arcgis_response(features),
        ):
            result = fetch_hifld_hospitals(state="PA")
        names = set(result["facility_name"].str.upper())
        assert "MOUNT NITTANY MEDICAL CENTER" in names

    def test_empty_feature_list_returns_empty(self):
        with patch(
            "src.data_collection.requests.get",
            return_value=_mock_arcgis_response([]),
        ):
            result = fetch_hifld_hospitals(state="PA")
        assert len(result) == 0
        assert "facility_name" in result.columns


class TestFetchHrsaHealthCenters:
    def test_parses_hrsa_geojson(self):
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-75.16, 39.95]},
                "properties": {
                    "Site_Name": "PHILADELPHIA HEALTH CENTER 1",
                    "Site_Address": "500 S BROAD ST",
                    "Site_City": "PHILADELPHIA",
                    "Site_State_Abbreviation": "PA",
                    "Site_Postal_Code": "19146",
                },
            }
        ]
        with patch(
            "src.data_collection.requests.get",
            return_value=_mock_arcgis_response(features),
        ):
            result = fetch_hrsa_health_centers(state="PA")
        assert len(result) == 1
        row = result.iloc[0]
        assert row["facility_name"] == "PHILADELPHIA HEALTH CENTER 1"
        assert row["city"] == "PHILADELPHIA"
        assert row["source"] == "hrsa_hc"


class TestFetchHifldUrgentCare:
    def test_parses_urgent_care_geojson(self):
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-80.0, 40.44]},
                "properties": {
                    "NAME": "URGENT CARE PITTSBURGH",
                    "ADDRESS": "100 LIBERTY AVE",
                    "CITY": "PITTSBURGH",
                    "STATE": "PA",
                    "ZIP": "15222",
                },
            }
        ]
        with patch(
            "src.data_collection.requests.get",
            return_value=_mock_arcgis_response(features),
        ):
            result = fetch_hifld_urgent_care(state="PA")
        assert len(result) == 1
        assert result["source"].iloc[0] == "hifld_urgent_care"
        assert result["facility_name"].iloc[0] == "URGENT CARE PITTSBURGH"


class TestFetchCmsPos:
    def test_returns_empty_when_download_fails(self):
        with patch(
            "src.data_collection.requests.get",
            side_effect=Exception("network error"),
        ):
            result = fetch_cms_pos(state="PA")
        assert len(result) == 0
        assert "facility_name" in result.columns
        assert "source" in result.columns


class TestNpiUncapped:
    def test_limit_per_taxonomy_none_paginates_until_partial_page(self):
        """When no cap is set, pagination continues until the API returns a
        partial page, not until 200 records have been seen."""
        # Simulate 3 pages: full, full, partial.
        def make_payload(page_idx):
            return {
                "results": [
                    {
                        "number": f"{page_idx:04d}_{i:04d}",
                        "basic": {"organization_name": f"Clinic {page_idx}-{i}"},
                        "addresses": [
                            {
                                "address_purpose": "LOCATION",
                                "address_1": "1 MAIN ST",
                                "city": "PHILADELPHIA",
                                "state": "PA",
                                "postal_code": "19102",
                            }
                        ],
                    }
                    for i in range(200 if page_idx < 2 else 50)
                ]
            }

        responses = [MagicMock(raise_for_status=MagicMock()) for _ in range(3)]
        for i, resp in enumerate(responses):
            resp.json.return_value = make_payload(i)

        with patch("src.data_collection.requests.get", side_effect=responses):
            result = _fetch_npi_facilities(
                state="PA",
                taxonomy_descriptions=["General Acute Care Hospital"],
                source_label="test",
                limit_per_taxonomy=None,
            )
        # Expected: 200 + 200 + 50 = 450 unique records
        assert len(result) == 450

    def test_limit_per_taxonomy_200_keeps_legacy_cap(self):
        """With the explicit legacy cap, only 200 records are returned even
        if the API would supply more."""
        def make_payload(page_idx):
            return {
                "results": [
                    {
                        "number": f"{page_idx:04d}_{i:04d}",
                        "basic": {"organization_name": f"Clinic {page_idx}-{i}"},
                        "addresses": [
                            {
                                "address_purpose": "LOCATION",
                                "address_1": "1 MAIN ST",
                                "city": "PHILADELPHIA",
                                "state": "PA",
                                "postal_code": "19102",
                            }
                        ],
                    }
                    for i in range(200)
                ]
            }

        responses = [MagicMock(raise_for_status=MagicMock()) for _ in range(3)]
        for i, resp in enumerate(responses):
            resp.json.return_value = make_payload(i)

        with patch("src.data_collection.requests.get", side_effect=responses):
            result = _fetch_npi_facilities(
                state="PA",
                taxonomy_descriptions=["General Acute Care Hospital"],
                source_label="test",
                limit_per_taxonomy=200,
            )
        assert len(result) == 200
