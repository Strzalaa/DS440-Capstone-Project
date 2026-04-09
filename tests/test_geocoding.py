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
