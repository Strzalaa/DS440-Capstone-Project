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

from src.config import (
    ACS_VARIABLES,
    CENSUS_API_KEY,
    DATA_RAW,
    DATA_URLS,
    STATE_FIPS,
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


def download_cms_data(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Download CMS Provider of Services file and return as DataFrame.

    Parameters
    ----------
    output_dir : Path
        Directory to save the raw download.

    Returns
    -------
    pd.DataFrame
        Facility records with name, address, coordinates, and type.
    """
    out_dir = output_dir / "cms"
    _ensure_dir(out_dir)

    candidates = sorted(
        [
            *out_dir.glob("*.csv"),
            *out_dir.glob("*.xlsx"),
            *out_dir.glob("*.xls"),
            *out_dir.glob("*.txt"),
            *out_dir.glob("*.tsv"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        warnings.warn(
            "No CMS raw file found in data/raw/cms. Returning an empty template "
            "DataFrame; place the CMS POS file in this folder to populate it.",
            stacklevel=2,
        )
        template = _normalise_facility_columns(pd.DataFrame(), source="cms")
        template.to_csv(out_dir / "cms_facilities_standardized.csv", index=False)
        return template

    cms_df = _load_tabular(candidates[0])
    cms_df = _normalise_facility_columns(cms_df, source="cms")
    cms_df.to_csv(out_dir / "cms_facilities_standardized.csv", index=False)
    return cms_df


def download_hrsa_data(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Download HRSA Health Center site data.

    Parameters
    ----------
    output_dir : Path
        Directory to save the raw download.

    Returns
    -------
    pd.DataFrame
        Health center records with coordinates.
    """
    out_dir = output_dir / "hrsa"
    _ensure_dir(out_dir)

    candidates = sorted(
        [
            *out_dir.glob("*.csv"),
            *out_dir.glob("*.xlsx"),
            *out_dir.glob("*.xls"),
            *out_dir.glob("*.txt"),
            *out_dir.glob("*.tsv"),
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        warnings.warn(
            "No HRSA raw file found in data/raw/hrsa. Returning an empty template "
            "DataFrame; place the HRSA file in this folder to populate it.",
            stacklevel=2,
        )
        template = _normalise_facility_columns(pd.DataFrame(), source="hrsa")
        template.to_csv(out_dir / "hrsa_facilities_standardized.csv", index=False)
        return template

    hrsa_df = _load_tabular(candidates[0])
    hrsa_df = _normalise_facility_columns(hrsa_df, source="hrsa")
    hrsa_df.to_csv(out_dir / "hrsa_facilities_standardized.csv", index=False)
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
