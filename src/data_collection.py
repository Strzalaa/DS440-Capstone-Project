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
    limit_per_taxonomy: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch healthcare facility records from the NPI Registry API.

    The NPI Registry is a public CMS API that returns provider/organization
    records including addresses.  We query by state and taxonomy description,
    then deduplicate on NPI number.

    When ``limit_per_taxonomy`` is ``None`` the loop paginates until the API
    returns a partial page (``len(results) < page_size``), which means we've
    consumed every available record. This removes the 200-record cap that was
    silently truncating the ingest.
    """
    api_url = "https://npiregistry.cms.hhs.gov/api/"
    all_records: list[dict] = []
    seen_npi: set[str] = set()

    for taxonomy in taxonomy_descriptions:
        skip = 0
        page_size = 200
        while True:
            if limit_per_taxonomy is not None:
                remaining = limit_per_taxonomy - skip
                if remaining <= 0:
                    break
                request_limit = str(min(page_size, remaining))
            else:
                request_limit = str(page_size)
            params = {
                "version": "2.1",
                "state": state,
                "enumeration_type": "NPI-2",
                "taxonomy_description": taxonomy,
                "limit": request_limit,
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
            if len(results) < page_size:
                break
            if limit_per_taxonomy is not None and skip >= limit_per_taxonomy:
                break
            # NPI Registry has a documented skip cap of 1200; stop before
            # hitting the server-side limit to avoid HTTP 400s.
            if skip >= 1200:
                break
            time.sleep(0.3)

    df = pd.DataFrame(all_records)
    if df.empty:
        return _normalise_facility_columns(pd.DataFrame(), source=source_label)
    return _normalise_facility_columns(df, source=source_label)


# ---------------------------------------------------------------------------
# Pre-geocoded facility fetchers (HIFLD / HRSA FeatureServer APIs)
# ---------------------------------------------------------------------------


def _paginate_arcgis_features(
    base_url: str,
    where: str,
    out_fields: str = "*",
    page_size: int = 1000,
    timeout: int = 60,
) -> list[dict]:
    """Paginate an ArcGIS FeatureServer ``query`` endpoint returning GeoJSON.

    Yields the combined list of GeoJSON ``features`` across all pages. Uses
    ``resultOffset`` / ``resultRecordCount`` for pagination, which is the
    documented mechanism for FeatureServer layers served by Esri.
    """
    features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "f": "geojson",
            "resultRecordCount": str(page_size),
            "resultOffset": str(offset),
        }
        try:
            resp = requests.get(base_url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            break
        page = payload.get("features", []) or []
        if not page:
            break
        features.extend(page)
        # exceededTransferLimit is sometimes nested under properties in the
        # GeoJSON response; check both locations.
        exceeded = payload.get("exceededTransferLimit", False) or payload.get(
            "properties", {}
        ).get("exceededTransferLimit", False)
        if len(page) < page_size and not exceeded:
            break
        offset += len(page)
        time.sleep(0.2)
    return features


def _feature_coords(feature: dict) -> tuple[Optional[float], Optional[float]]:
    """Extract (longitude, latitude) from a GeoJSON point feature."""
    geom = feature.get("geometry") or {}
    if geom.get("type") != "Point":
        return (None, None)
    coords = geom.get("coordinates") or []
    if len(coords) < 2:
        return (None, None)
    try:
        lon = float(coords[0])
        lat = float(coords[1])
    except (TypeError, ValueError):
        return (None, None)
    return (lon, lat)


def _first_nonempty(mapping: dict, keys: list[str]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "not available"}:
            return text
    return ""


def _fetch_arcgis_layer(
    url_key: str,
    state: str,
    source_label: str,
    name_keys: list[str],
    address_keys: list[str],
    city_keys: list[str],
    state_keys: list[str],
    zip_keys: list[str],
    provider_count_keys: list[str],
    state_where_fields: list[str],
) -> pd.DataFrame:
    """Generic ArcGIS MapServer/FeatureServer layer loader.

    Tries each candidate in ``state_where_fields`` until one returns rows,
    then normalises each feature into the unified facility schema.
    """
    url = DATA_URLS.get(url_key, "")
    if not url:
        return _normalise_facility_columns(pd.DataFrame(), source=source_label)

    features: list[dict] = []
    for wf in state_where_fields:
        candidate_wheres = [
            f"{wf}='{state.upper()}'",
            f"{wf}='{state.upper()}' AND STATUS='OPEN'",
        ]
        for where in candidate_wheres:
            features = _paginate_arcgis_features(url, where=where)
            if features:
                break
        if features:
            break

    records: list[dict] = []
    for feature in features:
        props = feature.get("properties", {}) or {}
        lon, lat = _feature_coords(feature)
        if lon is None or lat is None:
            lat = (
                props.get("Y")
                or props.get("Latitude")
                or props.get("LATITUDE")
                or props.get("lat")
            )
            lon = (
                props.get("X")
                or props.get("Longitude")
                or props.get("LONGITUDE")
                or props.get("lon")
            )
        provider_count = 1.0
        if provider_count_keys:
            raw = pd.to_numeric(
                pd.Series([_first_nonempty(props, provider_count_keys) or None]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(raw) and raw > 0:
                provider_count = float(raw)
        records.append(
            {
                "facility_name": _first_nonempty(props, name_keys),
                "address": _first_nonempty(props, address_keys),
                "city": _first_nonempty(props, city_keys),
                "state": _first_nonempty(props, state_keys) or state.upper(),
                "zip": _first_nonempty(props, zip_keys)[:5],
                "latitude": lat,
                "longitude": lon,
                "provider_count": provider_count,
            }
        )
    df = pd.DataFrame(records)
    return _normalise_facility_columns(df, source=source_label)


# Field-name candidates shared across HRSA CMS-approved facility layers
# (Hospitals, CAH, RHC, ASC). Also includes the classic HIFLD field spellings
# so the fetchers keep working against legacy HIFLD mirrors when they come
# back online, and so unit tests can mock either schema.
_CMS_APPROVED_NAME_KEYS = ["FACILITY_NM", "NAME", "name", "Facility_Name"]
_CMS_APPROVED_ADDRESS_KEYS = ["CMS_PROVIDER_ADDRESS", "ADDRESS", "address", "Address"]
_CMS_APPROVED_CITY_KEYS = ["CMS_PROVIDER_CITY", "CITY", "city", "City"]
_CMS_APPROVED_STATE_KEYS = [
    "CMS_PROVIDER_STATE_ABBR",
    "STATE",
    "State",
    "state",
]
_CMS_APPROVED_ZIP_KEYS = [
    "CMS_PROVIDER_ZIP_CD",
    "ZIP",
    "zipcode",
    "Zip",
    "POSTAL_CODE",
]
_CMS_APPROVED_BED_KEYS = ["TOT_BED_CT", "CERTIFIED_BED_CT", "BEDS"]
_CMS_APPROVED_STATE_WHERE_FIELDS = [
    "CMS_PROVIDER_STATE_ABBR",
    "STATE",
    "State",
]


def fetch_hifld_hospitals(state: str = STATE_ABBR) -> pd.DataFrame:
    """Fetch the CMS-approved hospitals layer (HRSA gisportal, layer 1).

    Historically this hit the HIFLD Hospitals FeatureServer; that service was
    deactivated in August 2025. HRSA's ``HealthCareFacilities`` MapServer
    hosts the same CMS Provider-of-Services-derived hospital registry with
    rooftop-accurate coordinates, so we've repointed at it. The function
    name is preserved for API stability.
    """
    return _fetch_arcgis_layer(
        url_key="hifld_hospitals",
        state=state,
        source_label="hifld_hospitals",
        name_keys=_CMS_APPROVED_NAME_KEYS,
        address_keys=_CMS_APPROVED_ADDRESS_KEYS,
        city_keys=_CMS_APPROVED_CITY_KEYS,
        state_keys=_CMS_APPROVED_STATE_KEYS,
        zip_keys=_CMS_APPROVED_ZIP_KEYS,
        provider_count_keys=_CMS_APPROVED_BED_KEYS,
        state_where_fields=_CMS_APPROVED_STATE_WHERE_FIELDS,
    )


def fetch_critical_access_hospitals(state: str = STATE_ABBR) -> pd.DataFrame:
    """Fetch the Critical Access Hospitals layer (HRSA gisportal, layer 2)."""
    return _fetch_arcgis_layer(
        url_key="hrsa_critical_access_hospitals",
        state=state,
        source_label="critical_access_hospital",
        name_keys=_CMS_APPROVED_NAME_KEYS,
        address_keys=_CMS_APPROVED_ADDRESS_KEYS,
        city_keys=_CMS_APPROVED_CITY_KEYS,
        state_keys=_CMS_APPROVED_STATE_KEYS,
        zip_keys=_CMS_APPROVED_ZIP_KEYS,
        provider_count_keys=_CMS_APPROVED_BED_KEYS,
        state_where_fields=_CMS_APPROVED_STATE_WHERE_FIELDS,
    )


def fetch_rural_health_clinics(state: str = STATE_ABBR) -> pd.DataFrame:
    """Fetch the Rural Health Clinics layer (HRSA gisportal, layer 4)."""
    return _fetch_arcgis_layer(
        url_key="hrsa_rural_health_clinics",
        state=state,
        source_label="rural_health_clinic",
        name_keys=_CMS_APPROVED_NAME_KEYS,
        address_keys=_CMS_APPROVED_ADDRESS_KEYS,
        city_keys=_CMS_APPROVED_CITY_KEYS,
        state_keys=_CMS_APPROVED_STATE_KEYS,
        zip_keys=_CMS_APPROVED_ZIP_KEYS,
        provider_count_keys=["PHY_FTE_CT", "NURSE_PRACT_FTE_CT"],
        state_where_fields=_CMS_APPROVED_STATE_WHERE_FIELDS,
    )


def fetch_hrsa_health_centers(state: str = STATE_ABBR) -> pd.DataFrame:
    """Fetch HRSA Health Center Service Delivery Sites (layer 18)."""
    return _fetch_arcgis_layer(
        url_key="hrsa_hc_geojson",
        state=state,
        source_label="hrsa_hc",
        name_keys=[
            "SITE_NM",
            "Site_Name",
            "SITE_NAME",
            "Health_Center_Name",
            "NAME",
        ],
        address_keys=[
            "SITE_ADDRESS",
            "Site_Address",
            "ADDRESS",
            "Address",
        ],
        city_keys=["SITE_CITY", "Site_City", "CITY", "City"],
        state_keys=[
            "SITE_STATE_ABBR",
            "Site_State_Abbreviation",
            "STATE",
            "State",
        ],
        zip_keys=[
            "SITE_ZIP_CD",
            "Site_Postal_Code",
            "ZIP",
            "Zip",
        ],
        provider_count_keys=["TOT_OPER_HR_PER_WEEK"],
        state_where_fields=[
            "SITE_STATE_ABBR",
            "Site_State_Abbreviation",
            "STATE",
        ],
    )


def fetch_hifld_urgent_care(state: str = STATE_ABBR) -> pd.DataFrame:
    """Fetch Ambulatory Surgical Centers (HRSA gisportal, layer 7).

    The HIFLD Urgent Care layer is defunct; Ambulatory Surgical Centers is
    the closest CMS-approved analogue that's still publicly served. Function
    name preserved for API stability.
    """
    return _fetch_arcgis_layer(
        url_key="hifld_urgent_care",
        state=state,
        source_label="hifld_urgent_care",
        name_keys=_CMS_APPROVED_NAME_KEYS,
        address_keys=_CMS_APPROVED_ADDRESS_KEYS,
        city_keys=_CMS_APPROVED_CITY_KEYS,
        state_keys=_CMS_APPROVED_STATE_KEYS,
        zip_keys=_CMS_APPROVED_ZIP_KEYS,
        provider_count_keys=["OPERATING_ROOM_CT"],
        state_where_fields=_CMS_APPROVED_STATE_WHERE_FIELDS,
    )


def fetch_cms_pos(state: str = STATE_ABBR) -> pd.DataFrame:
    """Fetch the CMS Provider of Services (POS) CSV and filter to ``state``.

    The POS file is large and CMS rotates the download URL, so the loader
    gracefully returns an empty frame when the download fails; callers are
    expected to fall back to a locally staged CSV when present.
    """
    url = DATA_URLS.get("cms_pos_csv", "")
    if not url:
        return _normalise_facility_columns(pd.DataFrame(), source="cms_pos")

    try:
        resp = requests.get(url, timeout=180, stream=True)
        resp.raise_for_status()
        df = pd.read_csv(pd.io.common.BytesIO(resp.content))
    except Exception:
        return _normalise_facility_columns(pd.DataFrame(), source="cms_pos")

    state_col = _pick_column(list(df.columns), ["state", "provider_state", "STATE_CD"])
    if state_col is not None:
        df = df[df[state_col].astype(str).str.upper().str.strip() == state.upper()]
    return _normalise_facility_columns(df, source="cms_pos")


# ---------------------------------------------------------------------------
# Taxonomy lists used by the NPI fallback for each facility tier.
# Expanded from the original five-entry lists to cover clinics, urgent care,
# ambulatory surgery, dialysis, and rural health clinics that HIFLD / HRSA
# miss.
# ---------------------------------------------------------------------------
HOSPITAL_TAXONOMIES: list[str] = [
    "General Acute Care Hospital",
    "Critical Access Hospital",
    "Rehabilitation Hospital",
    "Psychiatric Hospital",
    "Long Term Care Hospital",
    "Children's Hospital",
    "Chronic Disease Hospital",
    "Military Hospital",
    "Religious Nonmedical Health Care Institution",
]

HEALTH_CENTER_TAXONOMIES: list[str] = [
    "Federally Qualified Health Center (FQHC)",
    "Community Health",
    "Clinic/Center",
    "Rural Health Clinic",
    "Family Medicine",
    "Internal Medicine",
    "Pediatrics",
    "General Practice",
    "Preventive Medicine",
    "Ambulatory Health Care Facilities",
]

URGENT_CARE_TAXONOMIES: list[str] = [
    "Clinic/Center, Urgent Care",
    "Urgent Care",
    "Ambulatory Surgical",
]


def download_cms_data(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Download hospital facility data.

    Prefers the HRSA gisportal HealthCareFacilities service (CMS-approved
    hospitals + critical access hospitals, pre-geocoded). Falls back to a
    local CSV in ``data/raw/cms/`` and finally to the uncapped NPI Registry
    call.
    """
    out_dir = output_dir / "cms"
    _ensure_dir(out_dir)
    out_path = out_dir / "cms_facilities_standardized.csv"

    print("Fetching PA hospitals from HRSA gisportal (CMS hospital layers)...")
    combined_frames: list[pd.DataFrame] = []
    try:
        hospitals_df = fetch_hifld_hospitals(state=STATE_ABBR)
        if not hospitals_df.empty:
            combined_frames.append(hospitals_df)
            print(f"  Hospitals layer returned {len(hospitals_df)} records")
    except Exception as exc:
        warnings.warn(f"Hospitals layer fetch failed: {exc}")

    try:
        cah_df = fetch_critical_access_hospitals(state=STATE_ABBR)
        if not cah_df.empty:
            # Relabel so the merge step can still distinguish provenance.
            cah_df = cah_df.copy()
            cah_df["source"] = "critical_access_hospital"
            cah_df["facility_id"] = [
                f"cah_{idx:06d}" for idx in range(len(cah_df))
            ]
            combined_frames.append(cah_df)
            print(f"  Critical Access Hospitals layer returned {len(cah_df)} records")
    except Exception as exc:
        warnings.warn(f"Critical Access Hospitals fetch failed: {exc}")

    if combined_frames:
        hifld_df = pd.concat(combined_frames, ignore_index=True, sort=False)
        # Relabel the combined frame under the legacy "cms" source name so
        # downstream code stays agnostic of the per-layer breakdown.
        hifld_df.to_csv(out_path, index=False)
        return hifld_df

    candidates = sorted(
        [f for f in out_dir.glob("*.csv") if f.name != "cms_facilities_standardized.csv"]
        + [*out_dir.glob("*.xlsx"), *out_dir.glob("*.xls")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"  HIFLD unavailable; using local file {candidates[0].name}")
        cms_df = _load_tabular(candidates[0])
        cms_df = _normalise_facility_columns(cms_df, source="cms")
        cms_df.to_csv(out_path, index=False)
        return cms_df

    print("  HIFLD + local file unavailable; falling back to uncapped NPI Registry")
    cms_df = _fetch_npi_facilities(
        state=STATE_ABBR,
        taxonomy_descriptions=HOSPITAL_TAXONOMIES,
        source_label="cms",
        limit_per_taxonomy=None,
    )
    cms_df.to_csv(out_path, index=False)
    print(f"  Saved {len(cms_df)} hospital records to {out_path}")
    return cms_df


def download_hrsa_data(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Download health center / FQHC data.

    Prefers the HRSA Health Center Service Delivery Sites layer (pre-geocoded).
    Falls back to a local file in ``data/raw/hrsa/`` and finally to the
    uncapped NPI Registry call.
    """
    out_dir = output_dir / "hrsa"
    _ensure_dir(out_dir)
    out_path = out_dir / "hrsa_facilities_standardized.csv"

    print("Fetching PA health centers from HRSA service delivery sites layer...")
    try:
        hrsa_df = fetch_hrsa_health_centers(state=STATE_ABBR)
    except Exception as exc:
        warnings.warn(f"HRSA fetch failed: {exc}")
        hrsa_df = pd.DataFrame()

    if not hrsa_df.empty:
        print(f"  HRSA returned {len(hrsa_df)} health center records")
        hrsa_df.to_csv(out_path, index=False)
        return hrsa_df

    candidates = sorted(
        [f for f in out_dir.glob("*.csv") if f.name != "hrsa_facilities_standardized.csv"]
        + [*out_dir.glob("*.xlsx"), *out_dir.glob("*.xls")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"  HRSA API unavailable; using local file {candidates[0].name}")
        hrsa_df = _load_tabular(candidates[0])
        hrsa_df = _normalise_facility_columns(hrsa_df, source="hrsa")
        hrsa_df.to_csv(out_path, index=False)
        return hrsa_df

    print("  HRSA API + local file unavailable; falling back to uncapped NPI Registry")
    hrsa_df = _fetch_npi_facilities(
        state=STATE_ABBR,
        taxonomy_descriptions=HEALTH_CENTER_TAXONOMIES,
        source_label="hrsa",
        limit_per_taxonomy=None,
    )
    hrsa_df.to_csv(out_path, index=False)
    print(f"  Saved {len(hrsa_df)} health center records to {out_path}")
    return hrsa_df


def download_urgent_care_data(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Download urgent care / ambulatory care facility data.

    Pulls the HRSA gisportal Ambulatory Surgical Centers layer (CMS-approved,
    pre-geocoded) and the Rural Health Clinics layer and concatenates them.
    Falls back to a local CSV and finally to the uncapped NPI Registry for
    urgent-care / ambulatory-surgery taxonomies.
    """
    out_dir = output_dir / "urgent_care"
    _ensure_dir(out_dir)
    out_path = out_dir / "urgent_care_facilities_standardized.csv"

    print("Fetching PA urgent/ambulatory care facilities from HRSA gisportal...")
    combined_frames: list[pd.DataFrame] = []
    try:
        asc_df = fetch_hifld_urgent_care(state=STATE_ABBR)
        if not asc_df.empty:
            combined_frames.append(asc_df)
            print(f"  Ambulatory Surgical Centers layer returned {len(asc_df)} records")
    except Exception as exc:
        warnings.warn(f"ASC fetch failed: {exc}")

    try:
        rhc_df = fetch_rural_health_clinics(state=STATE_ABBR)
        if not rhc_df.empty:
            rhc_df = rhc_df.copy()
            rhc_df["source"] = "rural_health_clinic"
            rhc_df["facility_id"] = [
                f"rhc_{idx:06d}" for idx in range(len(rhc_df))
            ]
            combined_frames.append(rhc_df)
            print(f"  Rural Health Clinics layer returned {len(rhc_df)} records")
    except Exception as exc:
        warnings.warn(f"RHC fetch failed: {exc}")

    if combined_frames:
        urgent_df = pd.concat(combined_frames, ignore_index=True, sort=False)
        urgent_df.to_csv(out_path, index=False)
        return urgent_df

    candidates = sorted(
        [f for f in out_dir.glob("*.csv") if f.name != "urgent_care_facilities_standardized.csv"]
        + [*out_dir.glob("*.xlsx"), *out_dir.glob("*.xls")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"  HIFLD unavailable; using local file {candidates[0].name}")
        urgent_df = _load_tabular(candidates[0])
        urgent_df = _normalise_facility_columns(urgent_df, source="urgent_care")
        urgent_df.to_csv(out_path, index=False)
        return urgent_df

    print("  HIFLD + local file unavailable; falling back to uncapped NPI Registry")
    urgent_df = _fetch_npi_facilities(
        state=STATE_ABBR,
        taxonomy_descriptions=URGENT_CARE_TAXONOMIES,
        source_label="urgent_care",
        limit_per_taxonomy=None,
    )
    urgent_df.to_csv(out_path, index=False)
    print(f"  Saved {len(urgent_df)} urgent care records to {out_path}")
    return urgent_df


def download_cms_pos_data(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Download the CMS Provider of Services (POS) file.

    Used as a gap-filler for CMS-certified facilities (rural health clinics,
    ambulatory surgery, dialysis) not present in HIFLD / HRSA. Prefers a
    locally staged CSV; otherwise tries the direct download URL; otherwise
    returns an empty frame and lets the pipeline continue.
    """
    out_dir = output_dir / "cms_pos"
    _ensure_dir(out_dir)
    out_path = out_dir / "cms_pos_standardized.csv"

    candidates = sorted(
        [f for f in out_dir.glob("*.csv") if f.name != "cms_pos_standardized.csv"]
        + [*out_dir.glob("*.xlsx"), *out_dir.glob("*.xls")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        print(f"Loading CMS POS from local file {candidates[0].name}")
        pos_df = _load_tabular(candidates[0])
        pos_df = _normalise_facility_columns(pos_df, source="cms_pos")
        pos_df.to_csv(out_path, index=False)
        return pos_df

    print("Attempting to download CMS POS file...")
    try:
        pos_df = fetch_cms_pos(state=STATE_ABBR)
    except Exception as exc:
        warnings.warn(f"CMS POS fetch failed: {exc}")
        pos_df = pd.DataFrame()

    if pos_df.empty:
        print("  CMS POS unavailable; continuing without POS gap-fill")
        return _normalise_facility_columns(pd.DataFrame(), source="cms_pos")

    print(f"  Saved {len(pos_df)} POS records to {out_path}")
    pos_df.to_csv(out_path, index=False)
    return pos_df


def download_npi_gapfill(output_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Pull the full uncapped NPI Registry snapshot for gap-filling.

    This runs all three taxonomy families through the uncapped NPI fetcher
    and saves a single standardized CSV. It is the fallback of last resort
    after HIFLD/HRSA/POS have been merged.
    """
    out_dir = output_dir / "npi"
    _ensure_dir(out_dir)
    out_path = out_dir / "npi_facilities_standardized.csv"

    print("Fetching uncapped NPI Registry snapshot (gap-fill tier)...")
    npi_df = _fetch_npi_facilities(
        state=STATE_ABBR,
        taxonomy_descriptions=HOSPITAL_TAXONOMIES
        + HEALTH_CENTER_TAXONOMIES
        + URGENT_CARE_TAXONOMIES,
        source_label="npi",
        limit_per_taxonomy=None,
    )
    npi_df.to_csv(out_path, index=False)
    print(f"  Saved {len(npi_df)} NPI records to {out_path}")
    return npi_df


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

    # Idempotent shortcut: if both GeoPackages are already present, just
    # load them. TIGER downloads are large and flaky; this avoids re-fetching
    # several hundred MB on every notebook run.
    tracts_gpkg = out_dir / "tl_2024_42_tract.gpkg"
    roads_gpkg = out_dir / "tl_2024_42_roads.gpkg"
    if tracts_gpkg.exists() and roads_gpkg.exists():
        print("TIGER/Line GeoPackages already present; skipping download")
        return {
            "tracts": gpd.read_file(tracts_gpkg),
            "roads": gpd.read_file(roads_gpkg),
        }

    tract_zip = out_dir / "tl_2024_42_tract.zip"
    if not tract_zip.exists():
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
        if not zip_path.exists():
            roads_url = f"https://www2.census.gov/geo/tiger/TIGER2024/ROADS/{zip_name}"
            _download_to_path(roads_url, zip_path)
        shp_name = zip_name.replace(".zip", ".shp")
        roads_frames.append(gpd.read_file(f"zip://{zip_path}!{shp_name}"))

    roads = gpd.GeoDataFrame(
        pd.concat(roads_frames, ignore_index=True), crs=roads_frames[0].crs
    )

    tracts.to_file(tracts_gpkg, driver="GPKG")
    roads.to_file(roads_gpkg, driver="GPKG")

    return {"tracts": tracts, "roads": roads}
