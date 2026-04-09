"""Geocoding utilities for healthcare facility records.

Provides functions to geocode facility addresses to latitude/longitude,
validate coordinate accuracy, and cross-reference across data sources.
"""

from __future__ import annotations

import time
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


def _clean_query_text(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    return cleaned.replace(" AND ", " & ")


def _normalize_city(city: str) -> str:
    normalized = _clean_query_text(city).upper()
    aliases = {
        "PHILA": "PHILADELPHIA",
        "PITTSBURG": "PITTSBURGH",
        "MT GRETNA": "MOUNT GRETNA",
        "ST MARYS": "SAINT MARYS",
    }
    return aliases.get(normalized, normalized)


def _build_query_candidates(
    row: pd.Series,
    address_col: str,
    city_col: str,
    state_col: str,
    zip_col: str,
) -> list[str]:
    facility_name = _clean_query_text(row.get("facility_name", ""))
    address = _clean_query_text(row.get(address_col, ""))
    city = _normalize_city(str(row.get(city_col, "")))
    state = _clean_query_text(row.get(state_col, ""))
    zip_code = _clean_query_text(str(row.get(zip_col, "") or "")[:5])
    address_variants = [address]
    if "&" in address:
        address_variants.append(address.replace("&", "AND"))
    if " ROAD" in address:
        address_variants.append(address.replace(" ROAD", " RD"))
    if " RD" in address:
        address_variants.append(address.replace(" RD", " ROAD"))
    if " BOULEVARD" in address:
        address_variants.append(address.replace(" BOULEVARD", " BLVD"))
    if " BLVD" in address:
        address_variants.append(address.replace(" BLVD", " BOULEVARD"))

    candidates: list[str] = []
    for address_variant in address_variants:
        for parts in [
            [facility_name, address_variant, city, state, zip_code],
            [address_variant, city, state, zip_code],
            [facility_name, city, state, zip_code],
            [facility_name, address_variant, city, state],
            [facility_name, city, state],
            [address_variant, city, state],
        ]:
            query = ", ".join(part for part in parts if part)
            if query and query not in candidates:
                candidates.append(query)
    return candidates


def _nominatim_geocode(query: str, timeout: int = 30) -> tuple[Optional[float], Optional[float]]:
    if not query:
        return None, None
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "us",
                "addressdetails": 1,
            },
            headers={"User-Agent": "DS440-Capstone-Project/1.0"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return None, None
        match = payload[0]
        address = match.get("address", {})
        state_name = str(address.get("state", "") or "")
        if state_name and "pennsylvania" not in state_name.lower():
            return None, None
        return float(match["lat"]), float(match["lon"])
    except Exception:
        return None, None


def _census_batch_geocode(df: pd.DataFrame, address_col: str, city_col: str,
                          state_col: str, zip_col: str,
                          batch_size: int = 1000) -> pd.DataFrame:
    """Batch geocode via the Census Bureau batch endpoint (up to 10K per call).

    Returns a copy of df with 'latitude' and 'longitude' filled where possible.
    """
    import io
    import csv
    import time as _time

    to_geocode = df[df["latitude"].isna() | df["longitude"].isna()].copy()
    if to_geocode.empty:
        return df

    result_df = df.copy()
    url = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"

    for start in range(0, len(to_geocode), batch_size):
        chunk = to_geocode.iloc[start : start + batch_size]
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        for idx, row in chunk.iterrows():
            writer.writerow([
                idx,
                str(row.get(address_col, "") or "").strip(),
                str(row.get(city_col, "") or "").strip(),
                str(row.get(state_col, "") or "").strip(),
                str(row.get(zip_col, "") or "").strip()[:5],
            ])
        csv_bytes = csv_buf.getvalue().encode("utf-8")
        try:
            resp = requests.post(
                url,
                files={"addressFile": ("addresses.csv", csv_bytes, "text/csv")},
                data={"benchmark": "Public_AR_Current"},
                timeout=300,
            )
            resp.raise_for_status()
            reader = csv.reader(io.StringIO(resp.text))
            for row_data in reader:
                if len(row_data) < 6:
                    continue
                row_id = row_data[0].strip()
                if not row_id.isdigit():
                    continue
                row_idx = int(row_id)
                match_status = row_data[2].strip() if len(row_data) > 2 else ""
                if match_status.lower() != "match":
                    continue
                coord_field = row_data[5].strip() if len(row_data) > 5 else ""
                try:
                    lon_str, lat_str = coord_field.split(",", 1)
                    lat = float(lat_str.strip())
                    lon = float(lon_str.strip())
                    result_df.at[row_idx, "latitude"] = lat
                    result_df.at[row_idx, "longitude"] = lon
                    result_df.at[row_idx, "geocode_status"] = "batch_geocoded"
                except (ValueError, IndexError):
                    continue
        except Exception:
            pass
        if start + batch_size < len(to_geocode):
            _time.sleep(1)

    return result_df


def geocode_facilities(
    facilities: pd.DataFrame,
    address_col: str = "address",
    city_col: str = "city",
    state_col: str = "state",
    zip_col: str = "zip",
) -> gpd.GeoDataFrame:
    """Geocode facility addresses to point geometries.

    Uses the Census Bureau batch geocoder for efficiency, then falls back to
    the single-address endpoint for any remaining failures.

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
    if df.empty:
        return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")

    columns = list(df.columns)
    lat_col = _pick_column(columns, ["latitude", "lat", "y"])
    lon_col = _pick_column(columns, ["longitude", "lon", "lng", "x"])

    if lat_col is None:
        df["latitude"] = None
        lat_col = "latitude"
    if lon_col is None:
        df["longitude"] = None
        lon_col = "longitude"
    for required_col in [address_col, city_col, state_col, zip_col]:
        if required_col not in df.columns:
            df[required_col] = ""

    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["geocode_status"] = "missing"
    has_source_coords = df["latitude"].notna() & df["longitude"].notna()
    df.loc[has_source_coords, "geocode_status"] = "provided"

    n_missing = (~has_source_coords).sum()
    if n_missing > 0:
        print(f"  Batch geocoding {n_missing} facilities via Census API...")
        df = _census_batch_geocode(df, address_col, city_col, state_col, zip_col)
        still_missing = df["latitude"].isna() | df["longitude"].isna()
        n_still = still_missing.sum()
        print(f"  Batch geocoded {n_missing - n_still} / {n_missing} facilities")
        if n_still > 0:
            print(f"  Census single-address fallback for {n_still} remaining...")
            census_cache: dict[str, tuple[Optional[float], Optional[float]]] = {}
            before_census = n_still
            for idx in df.index[still_missing]:
                query = ", ".join(
                    str(df.at[idx, col]).strip()
                    for col in [address_col, city_col, state_col, zip_col]
                    if str(df.at[idx, col]).strip()
                )
                if query not in census_cache:
                    census_cache[query] = _census_geocode_oneline(query)
                lat, lon = census_cache[query]
                if lat is not None and lon is not None:
                    df.at[idx, "latitude"] = lat
                    df.at[idx, "longitude"] = lon
                    df.at[idx, "geocode_status"] = "geocoded"

            still_missing = df["latitude"].isna() | df["longitude"].isna()
            n_still = still_missing.sum()
            print(f"  Census fallback geocoded {before_census - n_still} additional facilities")

        if n_still > 0:
            print(f"  Nominatim fallback for {n_still} remaining...")
            nominatim_cache: dict[str, tuple[Optional[float], Optional[float]]] = {}
            for idx in df.index[still_missing]:
                row = df.loc[idx]
                for query in _build_query_candidates(row, address_col, city_col, state_col, zip_col):
                    if query not in nominatim_cache:
                        nominatim_cache[query] = _nominatim_geocode(query)
                        time.sleep(1.0)
                    lat, lon = nominatim_cache[query]
                    if lat is not None and lon is not None:
                        df.at[idx, "latitude"] = lat
                        df.at[idx, "longitude"] = lon
                        df.at[idx, "geocode_status"] = "nominatim_geocoded"
                        break

            still_missing = df["latitude"].isna() | df["longitude"].isna()
            print(f"  Total geocoded after all fallbacks: {(~still_missing).sum()} / {len(df)}")

    geometry = [
        Point(lon, lat) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(df["latitude"], df["longitude"])
    ]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")


def _address_group_key(df: pd.DataFrame) -> pd.Series:
    address = df["address"].fillna("").astype(str).str.upper().str.strip() if "address" in df.columns else ""
    city = df["city"].fillna("").astype(str).str.upper().str.strip() if "city" in df.columns else ""
    state = df["state"].fillna("").astype(str).str.upper().str.strip() if "state" in df.columns else ""
    zip_code = (
        df["zip"].fillna("").astype(str).str.upper().str.strip().str[:5]
        if "zip" in df.columns
        else ""
    )
    if isinstance(address, str):
        return pd.Series(df.index.astype(str), index=df.index)
    key = address + "|" + city + "|" + state + "|" + zip_code
    blank_mask = key.str.replace("|", "", regex=False).eq("")
    if "facility_id" in df.columns:
        key.loc[blank_mask] = "facility_id|" + df.loc[blank_mask, "facility_id"].astype(str)
    else:
        key.loc[blank_mask] = "row|" + df.index[blank_mask].astype(str)
    return key


def _deduplicate_facilities(facilities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if facilities.empty:
        return facilities

    work = facilities.copy()
    if "provider_count" in work.columns:
        work["provider_count"] = pd.to_numeric(work["provider_count"], errors="coerce").fillna(1.0)

    work["_group_key"] = _address_group_key(work)
    rows: list[pd.Series] = []
    for _, group in work.groupby("_group_key", sort=False):
        row = group.iloc[0].copy()
        if "provider_count" in group.columns:
            row["provider_count"] = float(group["provider_count"].sum())
        for col in ["source", "source_provenance"]:
            if col in group.columns:
                values = sorted({str(v).strip() for v in group[col].dropna() if str(v).strip()})
                row[col] = "+".join(values)
        if "geometry" in group.columns:
            valid_geom = group.loc[group.geometry.notna() & ~group.geometry.is_empty, "geometry"]
            row["geometry"] = valid_geom.iloc[0] if not valid_geom.empty else None
        rows.append(row)

    out = pd.DataFrame(rows).drop(columns=["_group_key"], errors="ignore")
    return gpd.GeoDataFrame(out, geometry="geometry", crs=facilities.crs)


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
    boundary_union = boundary.union_all()
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
        return _deduplicate_facilities(out)
    if hrsa.empty:
        out = cms.copy()
        out["source_provenance"] = "cms"
        return _deduplicate_facilities(out)

    cms = cms.to_crs(4326)
    hrsa = hrsa.to_crs(4326)
    cms["cms_idx"] = cms.index
    hrsa["hrsa_idx"] = hrsa.index

    cms_valid = cms[cms.geometry.notna() & ~cms.geometry.is_empty].copy()
    hrsa_valid = hrsa[hrsa.geometry.notna() & ~hrsa.geometry.is_empty].copy()

    matched_hrsa_idxs: set[int] = set()
    if not cms_valid.empty and not hrsa_valid.empty:
        cms_3857 = cms_valid.to_crs(3857)
        hrsa_3857 = hrsa_valid.to_crs(3857)
        nearest = gpd.sjoin_nearest(
            cms_3857,
            hrsa_3857[["hrsa_idx", "geometry"]],
            how="left",
            max_distance=match_threshold_m,
            distance_col="match_distance_m",
        )
        nearest_dedup = nearest[~nearest.index.duplicated(keep="first")]
        matched_hrsa_idxs = set(nearest_dedup["hrsa_idx"].dropna().astype(int).tolist())
        match_mask = nearest_dedup["hrsa_idx"].notna()
        cms.loc[match_mask.index[match_mask], "source_provenance"] = "cms+hrsa"

    merged = cms.copy()
    if "source_provenance" not in merged.columns:
        merged["source_provenance"] = "cms"
    merged["source_provenance"] = merged["source_provenance"].fillna("cms")

    unmatched_hrsa = hrsa.loc[~hrsa["hrsa_idx"].isin(matched_hrsa_idxs)].copy()
    unmatched_hrsa["source_provenance"] = "hrsa"

    out = pd.concat([merged, unmatched_hrsa], ignore_index=True, sort=False)
    for drop_col in ["cms_idx", "hrsa_idx"]:
        if drop_col in out.columns:
            out = out.drop(columns=[drop_col])
    out_gdf = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    return _deduplicate_facilities(out_gdf)
