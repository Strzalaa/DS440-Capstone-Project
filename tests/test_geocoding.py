"""Tests for the geocoding module."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from src.geocoding import geocode_facilities, validate_coordinates, cross_reference_sources


class TestGeocodeFacilities:
    def test_output_is_geodataframe(self):
        df = pd.DataFrame({
            "facility_name": ["Test Facility"],
            "address": ["123 Main St"],
            "city": ["Harrisburg"],
            "state": ["PA"],
            "zip": ["17101"],
            "latitude": [40.2732],
            "longitude": [-76.8867],
        })
        result = geocode_facilities(df)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_crs_is_epsg4326(self):
        df = pd.DataFrame({
            "facility_name": ["Test"],
            "latitude": [40.0],
            "longitude": [-76.0],
        })
        result = geocode_facilities(df)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

    def test_provided_coords_used(self):
        df = pd.DataFrame({
            "facility_name": ["With Coords"],
            "latitude": [40.5],
            "longitude": [-76.5],
            "address": [""],
            "city": [""],
            "state": ["PA"],
            "zip": [""],
        })
        result = geocode_facilities(df)
        assert result["geocode_status"].iloc[0] == "provided"
        assert result.geometry.iloc[0] is not None
        assert abs(result.geometry.iloc[0].x - (-76.5)) < 0.001

    def test_empty_input_returns_empty_gdf(self):
        result = geocode_facilities(pd.DataFrame())
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0

    def test_geocode_status_column_added(self):
        df = pd.DataFrame({
            "facility_name": ["A"],
            "latitude": [None],
            "longitude": [None],
        })
        result = geocode_facilities(df)
        assert "geocode_status" in result.columns


class TestValidateCoordinates:
    def test_inside_boundary_is_valid(self):
        gdf = gpd.GeoDataFrame(
            {"facility_id": ["f1"]},
            geometry=[Point(-76.88, 40.27)],
            crs="EPSG:4326",
        )
        boundary = gpd.GeoDataFrame(
            geometry=[box(-81, 39.5, -74.5, 42.5)],
            crs="EPSG:4326",
        )
        result = validate_coordinates(gdf, boundary)
        assert bool(result["is_valid"].iloc[0]) is True

    def test_outside_boundary_is_invalid(self):
        gdf = gpd.GeoDataFrame(
            {"facility_id": ["f1"]},
            geometry=[Point(-90.0, 30.0)],
            crs="EPSG:4326",
        )
        boundary = gpd.GeoDataFrame(
            geometry=[box(-81, 39.5, -74.5, 42.5)],
            crs="EPSG:4326",
        )
        result = validate_coordinates(gdf, boundary)
        assert bool(result["is_valid"].iloc[0]) is False

    def test_empty_gdf_returns_empty(self):
        gdf = gpd.GeoDataFrame(columns=["facility_id", "geometry"], geometry="geometry", crs="EPSG:4326")
        boundary = gpd.GeoDataFrame(geometry=[box(-81, 39.5, -74.5, 42.5)], crs="EPSG:4326")
        result = validate_coordinates(gdf, boundary)
        assert len(result) == 0


class TestCrossReferenceSources:
    def test_both_empty_returns_empty(self):
        cms = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
        hrsa = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
        result = cross_reference_sources(cms, hrsa)
        assert len(result) == 0

    def test_cms_only(self):
        cms = gpd.GeoDataFrame(
            {"facility_name": ["Hospital A"]},
            geometry=[Point(-76.0, 40.0)],
            crs="EPSG:4326",
        )
        hrsa = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
        result = cross_reference_sources(cms, hrsa)
        assert len(result) == 1
        assert result["source_provenance"].iloc[0] == "cms"

    def test_hrsa_only(self):
        cms = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
        hrsa = gpd.GeoDataFrame(
            {"facility_name": ["Health Center"]},
            geometry=[Point(-76.0, 40.0)],
            crs="EPSG:4326",
        )
        result = cross_reference_sources(cms, hrsa)
        assert len(result) == 1
        assert result["source_provenance"].iloc[0] == "hrsa"

    def test_deduplicates_exact_same_address_after_merge(self):
        cms = gpd.GeoDataFrame(
            {
                "facility_name": ["Hospital A", "Hospital A Duplicate"],
                "address": ["123 Main St", "123 Main St"],
                "city": ["York", "York"],
                "state": ["PA", "PA"],
                "zip": ["17401", "17401"],
                "provider_count": [1.0, 1.0],
                "source": ["cms", "cms"],
            },
            geometry=[Point(-76.73, 39.96), Point(-76.73, 39.96)],
            crs="EPSG:4326",
        )
        hrsa = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")
        result = cross_reference_sources(cms, hrsa)
        assert len(result) == 1
        assert result["provider_count"].iloc[0] == 2.0


class TestNwayMerge:
    """Tests for the N-way priority merge + fuzzy name matching added on top
    of the legacy CMS+HRSA signature."""

    def _mk(self, source, name, lon, lat, address="1 Main St", city="Anywhere", zip_="17101"):
        return gpd.GeoDataFrame(
            {
                "facility_name": [name],
                "address": [address],
                "city": [city],
                "state": ["PA"],
                "zip": [zip_],
                "provider_count": [1.0],
                "source": [source],
            },
            geometry=[Point(lon, lat)],
            crs="EPSG:4326",
        )

    def test_three_sources_spatial_match_collapses_to_one(self):
        hifld = self._mk("hifld_hospitals", "Mount Nittany Medical Center", -77.85, 40.79)
        hrsa = self._mk("hrsa_hc", "Mount Nittany Medical Center", -77.850005, 40.790003)
        npi = self._mk("npi", "Mount Nittany Medical Center", -77.850002, 40.790001)
        result = cross_reference_sources(
            hifld,
            hrsa,
            npi,
            priority=["hifld_hospitals", "hrsa_hc", "npi"],
        )
        assert len(result) == 1
        prov = result["source_provenance"].iloc[0]
        parts = set(prov.split("+"))
        assert {"hifld_hospitals", "hrsa_hc", "npi"}.issubset(parts)

    def test_priority_winner_keeps_highest_priority_source(self):
        """When three sources match, the record from the highest-priority
        source supplies the geometry/name."""
        hifld = self._mk("hifld_hospitals", "HIFLD Name", -77.85, 40.79)
        hrsa = self._mk("hrsa_hc", "HRSA Name", -77.850005, 40.790003)
        npi = self._mk("npi", "NPI Name", -77.850002, 40.790001)
        result = cross_reference_sources(
            hifld,
            hrsa,
            npi,
            priority=["hifld_hospitals", "hrsa_hc", "npi"],
        )
        assert len(result) == 1
        assert result["facility_name"].iloc[0] == "HIFLD Name"

    def test_fuzzy_name_match_within_radius_merges(self):
        """Two records ~800m apart with fuzzy-similar names should collapse
        when the fuzzy-name radius is large enough."""
        a = self._mk("hifld_hospitals", "Geisinger Medical Center", -76.60, 41.00)
        # ~1 km away — beyond the 250m spatial threshold but within the
        # 1500m fuzzy-name radius — with a near-identical name.
        b = self._mk("hrsa_hc", "Geisinger Med Center", -76.60, 41.009)
        result = cross_reference_sources(a, b, priority=["hifld_hospitals", "hrsa_hc"])
        assert len(result) == 1

    def test_far_apart_records_do_not_merge(self):
        a = self._mk(
            "hifld_hospitals",
            "Hospital X",
            -76.60,
            41.00,
            address="100 Center St",
            city="Williamsport",
            zip_="17701",
        )
        b = self._mk(
            "hrsa_hc",
            "Hospital X",
            -80.00,
            40.44,
            address="200 Liberty Ave",
            city="Pittsburgh",
            zip_="15222",
        )
        result = cross_reference_sources(a, b, priority=["hifld_hospitals", "hrsa_hc"])
        assert len(result) == 2

    def test_provider_count_sums_across_matched_sources(self):
        a = gpd.GeoDataFrame(
            {
                "facility_name": ["Mt Nittany"],
                "address": ["1800 E Park Ave"],
                "city": ["State College"],
                "state": ["PA"],
                "zip": ["16803"],
                "provider_count": [260.0],
                "source": ["hifld_hospitals"],
            },
            geometry=[Point(-77.85, 40.79)],
            crs="EPSG:4326",
        )
        b = gpd.GeoDataFrame(
            {
                "facility_name": ["Mt Nittany"],
                "address": ["1800 E Park Ave"],
                "city": ["State College"],
                "state": ["PA"],
                "zip": ["16803"],
                "provider_count": [15.0],
                "source": ["npi"],
            },
            geometry=[Point(-77.850001, 40.790001)],
            crs="EPSG:4326",
        )
        result = cross_reference_sources(a, b, priority=["hifld_hospitals", "npi"])
        assert len(result) == 1
        assert result["provider_count"].iloc[0] >= 260.0

    def test_empty_sources_filtered_out(self):
        hifld = self._mk("hifld_hospitals", "Hospital A", -76.60, 41.00)
        empty = gpd.GeoDataFrame(columns=["geometry", "source"], geometry="geometry", crs="EPSG:4326")
        result = cross_reference_sources(hifld, empty, empty, priority=["hifld_hospitals"])
        assert len(result) == 1
        assert result["facility_name"].iloc[0] == "Hospital A"
