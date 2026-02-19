"""Geocoding utilities for healthcare facility records.

Provides functions to geocode facility addresses to latitude/longitude,
validate coordinate accuracy, and cross-reference across data sources.
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point


def _pick_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in lowered:
            return lowered[key]
    return None


def _census_geocode_oneline(address: str, timeout: int = 15) -> tuple[Optional[float], Optional[float]]:
    if not address or not isinstance(address, str):
        return None, None
    url = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        matches = payload.get("result", {}).get("addressMatches", [])
        if not matches:
            return None, None
        coords = matches[0]["coordinates"]
        return float(coords["y"]), float(coords["x"])
    except Exception:
        return None, None


def geocode_facilities(
    facilities: pd.DataFrame,
    address_col: str = "address",
    city_col: str = "city",
    state_col: str = "state",
    zip_col: str = "zip",
) -> gpd.GeoDataFrame:
    """Geocode facility addresses to point geometries.

    Uses the Census Bureau geocoder as the primary service.  Falls back to
    coordinates already present in the source data when available.

    Parameters
    ----------
    facilities : pd.DataFrame
        Raw facility records.
    address_col, city_col, state_col, zip_col : str
        Column names for address components.

    Returns
    -------
    gpd.GeoDataFrame
        Input records with added ``geometry`` column (EPSG:4326).
    """
    df = facilities.copy()
    columns = list(df.columns)

    lat_col = _pick_column(columns, ["latitude", "lat", "y"])
    lon_col = _pick_column(columns, ["longitude", "lon", "lng", "x"])

    if lat_col is None:
        df["latitude"] = None
        lat_col = "latitude"
    if lon_col is None:
        df["longitude"] = None
        lon_col = "longitude"

    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["geocode_status"] = "missing"
    has_source_coords = df["latitude"].notna() & df["longitude"].notna()
    df.loc[has_source_coords, "geocode_status"] = "provided"

    missing_coords = ~has_source_coords
    if missing_coords.any():
        address_pieces = []
        for _, row in df.iterrows():
            address_pieces.append(
                ", ".join(
                    str(row.get(col, "")).strip()
                    for col in [address_col, city_col, state_col, zip_col]
                    if str(row.get(col, "")).strip()
                )
            )
        for idx in df.index[missing_coords]:
            lat, lon = _census_geocode_oneline(address_pieces[idx])
            if lat is not None and lon is not None:
                df.at[idx, "latitude"] = lat
                df.at[idx, "longitude"] = lon
                df.at[idx, "geocode_status"] = "geocoded"

    geometry = [
        Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def validate_coordinates(
    gdf: gpd.GeoDataFrame,
    state_boundary: gpd.GeoDataFrame,
    tolerance_m: float = 5000.0,
) -> pd.DataFrame:
    """Flag records whose coordinates fall outside the expected state boundary.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Geocoded facilities.
    state_boundary : gpd.GeoDataFrame
        State polygon used as the bounding reference.
    tolerance_m : float
        Buffer distance in metres around the boundary before flagging.

    Returns
    -------
    pd.DataFrame
        Validation report with columns ``facility_id``, ``is_valid``,
        ``distance_to_boundary_m``.
    """
    if gdf.empty:
        return pd.DataFrame(columns=["facility_id", "is_valid", "distance_to_boundary_m"])

    facilities = gdf.to_crs(3857)
    boundary = state_boundary.to_crs(3857)
    boundary_union = boundary.unary_union
    boundary_buffer = boundary_union.buffer(tolerance_m)

    ids = facilities["facility_id"] if "facility_id" in facilities.columns else facilities.index
    distances = []
    valid_flags = []
    for geom in facilities.geometry:
        if geom is None or geom.is_empty:
            distances.append(float("nan"))
            valid_flags.append(False)
            continue
        inside = geom.within(boundary_buffer)
        valid_flags.append(bool(inside))
        if inside:
            distances.append(0.0)
        else:
            distances.append(float(geom.distance(boundary_union)))

    return pd.DataFrame(
        {
            "facility_id": ids,
            "is_valid": valid_flags,
            "distance_to_boundary_m": distances,
        }
    )


def cross_reference_sources(
    cms_facilities: gpd.GeoDataFrame,
    hrsa_facilities: gpd.GeoDataFrame,
    match_threshold_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Merge CMS and HRSA datasets, de-duplicate by spatial proximity.

    Parameters
    ----------
    cms_facilities : gpd.GeoDataFrame
        CMS Provider of Services records.
    hrsa_facilities : gpd.GeoDataFrame
        HRSA Health Center records.
    match_threshold_m : float
        Maximum distance (metres) to consider two records the same facility.

    Returns
    -------
    gpd.GeoDataFrame
        Unified facility dataset with source provenance column.
    """
    cms = cms_facilities.copy()
    hrsa = hrsa_facilities.copy()

    if cms.empty and hrsa.empty:
        return gpd.GeoDataFrame(columns=["source_provenance", "geometry"], geometry="geometry", crs="EPSG:4326")
    if cms.empty:
        out = hrsa.copy()
        out["source_provenance"] = "hrsa"
        return out
    if hrsa.empty:
        out = cms.copy()
        out["source_provenance"] = "cms"
        return out

    cms = cms.to_crs(4326)
    hrsa = hrsa.to_crs(4326)
    cms["cms_idx"] = cms.index
    hrsa["hrsa_idx"] = hrsa.index

    cms_3857 = cms.to_crs(3857)
    hrsa_3857 = hrsa.to_crs(3857)
    nearest = gpd.sjoin_nearest(
        cms_3857,
        hrsa_3857[["hrsa_idx", "geometry"]],
        how="left",
        max_distance=match_threshold_m,
        distance_col="match_distance_m",
    )

    matched_hrsa_idxs = set(nearest["hrsa_idx"].dropna().astype(int).tolist())
    merged = cms.copy()
    merged["source_provenance"] = "cms"
    merged.loc[nearest["hrsa_idx"].notna().values, "source_provenance"] = "cms+hrsa"

    unmatched_hrsa = hrsa.loc[~hrsa["hrsa_idx"].isin(matched_hrsa_idxs)].copy()
    unmatched_hrsa["source_provenance"] = "hrsa"

    out = pd.concat([merged, unmatched_hrsa], ignore_index=True, sort=False)
    if "cms_idx" in out.columns:
        out = out.drop(columns=["cms_idx"])
    if "hrsa_idx" in out.columns:
        out = out.drop(columns=["hrsa_idx"])
    return gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
