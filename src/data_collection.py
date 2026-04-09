"""Download and assemble raw datasets for the healthcare access analysis.

Data sources handled by this module:
- CMS Provider of Services (facility locations)
- HRSA Health Center data
- US Census American Community Survey via API
- CDC Social Vulnerability Index
- Census TIGER/Line shapefiles (tracts and roads)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Optional
from urllib.parse import urlencode
import warnings

import geopandas as gpd
import pandas as pd
import requests

import time

from src.config import (
    ACS_VARIABLES,
    CENSUS_API_KEY,
    DATA_RAW,
    DATA_URLS,
    STATE_FIPS,
    STATE_ABBR,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _download_to_path(url: str, output_path: Path, timeout: int = 120) -> Path:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def _load_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".txt", ".tsv"}:
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(f"Unsupported tabular file type: {path}")


def _pick_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _normalise_facility_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    columns = list(df.columns)
    rename_map: dict[str, str] = {}
    mapping = {
        "facility_name": ["facility_name", "name", "provider_name", "site_name"],
        "address": ["address", "street_address", "address_line_1", "address1"],
        "city": ["city", "provider_city"],
        "state": ["state", "provider_state"],
        "zip": ["zip", "zipcode", "postal_code", "provider_zip_code"],
        "latitude": ["latitude", "lat", "y", "provider_latitude"],
        "longitude": ["longitude", "lon", "lng", "x", "provider_longitude"],
        "provider_count": ["provider_count", "capacity", "fte_count"],
    }
    for target, candidates in mapping.items():
        col = _pick_column(columns, candidates)
        if col:
            rename_map[col] = target

    out = df.rename(columns=rename_map).copy()
    for required in ["facility_name", "address", "city", "state", "zip"]:
        if required not in out.columns:
            out[required] = None
    for numeric in ["latitude", "longitude", "provider_count"]:
        if numeric not in out.columns:
            out[numeric] = None
        out[numeric] = pd.to_numeric(out[numeric], errors="coerce")

    out["provider_count"] = out["provider_count"].fillna(1.0)
    out["source"] = source
    out = out.reset_index(drop=True)
    out["facility_id"] = [f"{source}_{idx:06d}" for idx in out.index]
    keep_cols = [
        "facility_id",
        "facility_name",
        "address",
        "city",
        "state",
        "zip",
        "latitude",
        "longitude",
        "provider_count",
        "source",
    ]
    return out[keep_cols]


def _fetch_npi_facilities(
    state: str,
    taxonomy_descriptions: list[str],
    source_label: str,
    limit_per_taxonomy: int = 200,
) -> pd.DataFrame:
    """Fetch healthcare facility records from the NPI Registry API.

    The NPI Registry is a public CMS API that returns provider/organization
    records including addresses.  We query by state and taxonomy description,
    then deduplicate on NPI number.
    """
    api_url = "https://npiregistry.cms.hhs.gov/api/"
    all_records: list[dict] = []
    seen_npi: set[str] = set()

    for taxonomy in taxonomy_descriptions:
        skip = 0
        page_size = 200
        while True:
            params = {
                "version": "2.1",
                "state": state,
                "enumeration_type": "NPI-2",
                "taxonomy_description": taxonomy,
                "limit": str(min(page_size, limit_per_taxonomy - skip)),
                "skip": str(skip),
            }
            try:
                resp = requests.get(api_url, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()
            except Exception:
                break

            results = payload.get("results", [])
            if not results:
                break

            for r in results:
                npi = str(r.get("number", ""))
                if npi in seen_npi:
                    continue
                basic = r.get("basic", {})
                addresses = r.get("addresses", [])
                practice_addr = next(
                    (a for a in addresses if a.get("address_purpose") == "LOCATION"),
                    addresses[0] if addresses else {},
                )
                practice_state = str(practice_addr.get("state", "") or "").strip().upper()
                if practice_state != state.upper():
                    continue
                seen_npi.add(npi)
                all_records.append(
                    {
                        "npi": npi,
                        "facility_name": basic.get("organization_name", "Unknown"),
                        "address": practice_addr.get("address_1", ""),
                        "city": practice_addr.get("city", ""),
                        "state": practice_addr.get("state", state),
                        "zip": practice_addr.get("postal_code", "")[:5],
                        "latitude": None,
                        "longitude": None,
                        "taxonomy": taxonomy,
                        "provider_count": 1.0,
                    }
                )

            skip += len(results)
            if skip >= limit_per_taxonomy or len(results) < page_size:
                break
            time.sleep(0.3)

    df = pd.DataFrame(all_records)
    if df.empty:
        return _normalise_facility_columns(pd.DataFrame(), source=source_label)
    return _normalise_facility_columns(df, source=source_label)


def download_cms_data(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Download hospital facility data via the NPI Registry API.

    Falls back to a local file in ``data/raw/cms/`` if one exists.
    """
    out_dir = output_dir / "cms"
    _ensure_dir(out_dir)
    out_path = out_dir / "cms_facilities_standardized.csv"

    candidates = sorted(
        [f for f in out_dir.glob("*.csv") if f.name != "cms_facilities_standardized.csv"]
        + [*out_dir.glob("*.xlsx"), *out_dir.glob("*.xls")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        cms_df = _load_tabular(candidates[0])
        cms_df = _normalise_facility_columns(cms_df, source="cms")
        cms_df.to_csv(out_path, index=False)
        return cms_df

    print("Fetching PA hospital facilities from NPI Registry...")
    cms_df = _fetch_npi_facilities(
        state=STATE_ABBR,
        taxonomy_descriptions=[
            "General Acute Care Hospital",
            "Critical Access Hospital",
            "Rehabilitation Hospital",
            "Psychiatric Hospital",
            "Long Term Care Hospital",
        ],
        source_label="cms",
        limit_per_taxonomy=200,
    )
    cms_df.to_csv(out_path, index=False)
    print(f"  Saved {len(cms_df)} hospital records to {out_path}")
    return cms_df


def download_hrsa_data(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Download health center / FQHC data via the NPI Registry API.

    Falls back to a local file in ``data/raw/hrsa/`` if one exists.
    """
    out_dir = output_dir / "hrsa"
    _ensure_dir(out_dir)
    out_path = out_dir / "hrsa_facilities_standardized.csv"

    candidates = sorted(
        [f for f in out_dir.glob("*.csv") if f.name != "hrsa_facilities_standardized.csv"]
        + [*out_dir.glob("*.xlsx"), *out_dir.glob("*.xls")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        hrsa_df = _load_tabular(candidates[0])
        hrsa_df = _normalise_facility_columns(hrsa_df, source="hrsa")
        hrsa_df.to_csv(out_path, index=False)
        return hrsa_df

    print("Fetching PA health centers from NPI Registry...")
    hrsa_df = _fetch_npi_facilities(
        state=STATE_ABBR,
        taxonomy_descriptions=[
            "Federally Qualified Health Center",
            "Community Health Center",
            "Clinic/Center",
            "Family Medicine",
            "Internal Medicine",
            "Pediatrics",
        ],
        source_label="hrsa",
        limit_per_taxonomy=200,
    )
    hrsa_df.to_csv(out_path, index=False)
    print(f"  Saved {len(hrsa_df)} health center records to {out_path}")
    return hrsa_df


def fetch_acs_data(
    state_fips: str = STATE_FIPS,
    year: int = 2022,
    variables: Optional[dict[str, str]] = None,
    output_dir: Path = DATA_RAW,
) -> pd.DataFrame:
    """Fetch American Community Survey data from the Census API.

    Parameters
    ----------
    state_fips : str
        Two-digit state FIPS code.
    year : int
        ACS 5-year estimate vintage.
    variables : dict[str, str] | None
        Mapping of ACS variable codes to friendly names.
        Defaults to ``ACS_VARIABLES`` from config.

    Returns
    -------
    pd.DataFrame
        One row per census tract with requested demographic columns.
    """
    if not CENSUS_API_KEY:
        raise ValueError("CENSUS_API_KEY is missing.")

    vars_map = variables or ACS_VARIABLES
    acs_vars = list(vars_map.keys())
    get_vars = ["NAME", *acs_vars]
    query_params = {
        "get": ",".join(get_vars),
        "for": "tract:*",
        "in": f"state:{state_fips} county:*",
        "key": CENSUS_API_KEY,
    }
    endpoint = f"https://api.census.gov/data/{year}/acs/acs5?{urlencode(query_params)}"
    response = requests.get(endpoint, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if not payload or len(payload) < 2:
        raise RuntimeError("ACS response is empty.")

    header = payload[0]
    rows = payload[1:]
    df = pd.DataFrame(rows, columns=header)

    rename_map = {**vars_map}
    df = df.rename(columns=rename_map)
    for var_name in vars_map.values():
        df[var_name] = pd.to_numeric(df[var_name], errors="coerce")

    df["state"] = df["state"].astype(str).str.zfill(2)
    df["county"] = df["county"].astype(str).str.zfill(3)
    df["tract"] = df["tract"].astype(str).str.zfill(6)
    df["geoid"] = df["state"] + df["county"] + df["tract"]

    # Fetch pct_uninsured from the ACS Data Profile endpoint (DP03_0099PE)
    profile_params = {
        "get": "DP03_0099PE",
        "for": "tract:*",
        "in": f"state:{state_fips} county:*",
        "key": CENSUS_API_KEY,
    }
    profile_url = (
        f"https://api.census.gov/data/{year}/acs/acs5/profile"
        f"?{urlencode(profile_params)}"
    )
    try:
        profile_resp = requests.get(profile_url, timeout=120)
        profile_resp.raise_for_status()
        profile_payload = profile_resp.json()
        if profile_payload and len(profile_payload) >= 2:
            p_header = profile_payload[0]
            p_rows = profile_payload[1:]
            profile_df = pd.DataFrame(p_rows, columns=p_header)
            profile_df["state"] = profile_df["state"].astype(str).str.zfill(2)
            profile_df["county"] = profile_df["county"].astype(str).str.zfill(3)
            profile_df["tract"] = profile_df["tract"].astype(str).str.zfill(6)
            profile_df["geoid"] = (
                profile_df["state"] + profile_df["county"] + profile_df["tract"]
            )
            profile_df["pct_uninsured"] = pd.to_numeric(
                profile_df["DP03_0099PE"], errors="coerce"
            )
            df = df.merge(
                profile_df[["geoid", "pct_uninsured"]], on="geoid", how="left"
            )
            print(f"  Fetched pct_uninsured for {profile_df['pct_uninsured'].notna().sum()} tracts")
    except Exception as exc:
        warnings.warn(f"Could not fetch DP03 profile data: {exc}")
        if "pct_uninsured" not in df.columns:
            df["pct_uninsured"] = pd.NA

    out_dir = output_dir / "acs"
    _ensure_dir(out_dir)
    out_path = out_dir / f"acs_{year}_tract_{state_fips}.csv"
    df.to_csv(out_path, index=False)
    return df


def download_svi_data(
    state_fips: str = STATE_FIPS,
    output_dir: Path = DATA_RAW,
) -> pd.DataFrame:
    """Download CDC Social Vulnerability Index for the target state.

    Parameters
    ----------
    state_fips : str
        Two-digit state FIPS code.
    output_dir : Path
        Directory to save the raw download.

    Returns
    -------
    pd.DataFrame
        SVI theme scores per census tract.
    """
    if state_fips != STATE_FIPS:
        raise ValueError(
            f"This function currently supports state_fips={STATE_FIPS} only."
        )

    out_dir = output_dir / "svi"
    _ensure_dir(out_dir)
    out_path = out_dir / "SVI_2022_Pennsylvania.csv"

    base_url = DATA_URLS["svi_tract_layer_2022"]
    page_size = 2000
    offset = 0
    records: list[dict] = []
    while True:
        params = {
            "where": f"ST='{state_fips}'",
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json",
            "resultRecordCount": str(page_size),
            "resultOffset": str(offset),
        }
        response = requests.get(base_url, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        page = [feature["attributes"] for feature in payload.get("features", [])]
        records.extend(page)
        if not payload.get("exceededTransferLimit", False) or not page:
            break
        offset += page_size

    if not records:
        raise RuntimeError("No SVI tract records were returned for Pennsylvania.")

    svi_df = pd.DataFrame(records)
    if "FIPS" in svi_df.columns:
        svi_df["FIPS"] = svi_df["FIPS"].astype(str).str.zfill(11)
    svi_df.to_csv(out_path, index=False)
    return svi_df


def download_tiger_shapefiles(
    state_fips: str = STATE_FIPS,
    output_dir: Path = DATA_RAW,
) -> dict[str, gpd.GeoDataFrame]:
    """Download Census TIGER/Line shapefiles (tracts and roads).

    Parameters
    ----------
    state_fips : str
        Two-digit state FIPS code.
    output_dir : Path
        Directory to save extracted shapefiles.

    Returns
    -------
    dict[str, gpd.GeoDataFrame]
        Keys ``"tracts"`` and ``"roads"`` mapped to GeoDataFrames.
    """
    if state_fips != STATE_FIPS:
        raise ValueError(
            f"Configured TIGER URLs are for state_fips={STATE_FIPS} only."
        )

    out_dir = output_dir / "tiger"
    _ensure_dir(out_dir)

    tract_zip = out_dir / "tl_2024_42_tract.zip"
    _download_to_path(DATA_URLS["tiger_tracts"], tract_zip)

    tracts = gpd.read_file(f"zip://{tract_zip}!tl_2024_42_tract.shp")

    roads_index_url = "https://www2.census.gov/geo/tiger/TIGER2024/ROADS/?C=M;O=A"
    index_html = requests.get(roads_index_url, timeout=120).text
    county_zip_names = sorted(set(re.findall(r"tl_2024_42\d{3}_roads\.zip", index_html)))
    if not county_zip_names:
        raise RuntimeError("No Pennsylvania county TIGER roads files found.")

    roads_frames: list[gpd.GeoDataFrame] = []
    for zip_name in county_zip_names:
        zip_path = out_dir / zip_name
        roads_url = f"https://www2.census.gov/geo/tiger/TIGER2024/ROADS/{zip_name}"
        _download_to_path(roads_url, zip_path)
        shp_name = zip_name.replace(".zip", ".shp")
        roads_frames.append(gpd.read_file(f"zip://{zip_path}!{shp_name}"))

    roads = gpd.GeoDataFrame(
        pd.concat(roads_frames, ignore_index=True), crs=roads_frames[0].crs
    )

    tracts.to_file(out_dir / "tl_2024_42_tract.gpkg", driver="GPKG")
    roads.to_file(out_dir / "tl_2024_42_roads.gpkg", driver="GPKG")

    return {"tracts": tracts, "roads": roads}
