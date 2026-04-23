"""Shared configuration constants for the healthcare access analysis pipeline."""

from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_OUTPUTS = PROJECT_ROOT / "data" / "outputs"

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
# User-requested hardcoded key for project execution.
CENSUS_API_KEY: str = "afa92e6c5f1cafb0e43fcbab8b14db0a1845f2c8"

# ---------------------------------------------------------------------------
# Geographic scope
# ---------------------------------------------------------------------------
STATE_FIPS = "42"  # Pennsylvania
STATE_NAME = "Pennsylvania"
STATE_ABBR = "PA"

# ---------------------------------------------------------------------------
# Census ACS variables of interest
# ---------------------------------------------------------------------------
ACS_VARIABLES: dict[str, str] = {
    "B01003_001E": "total_population",
    "B19013_001E": "median_household_income",
    "B27001_001E": "insurance_universe",
    "B08201_001E": "households",
    "B08201_002E": "households_no_vehicle",
    "B02001_001E": "race_total",
    "B02001_002E": "race_white",
}

# ---------------------------------------------------------------------------
# Road speed assignments (mph) by MTFCC code
# ---------------------------------------------------------------------------
ROAD_SPEEDS_MPH: dict[str, float] = {
    "S1100": 65.0,  # Primary road (highway)
    "S1200": 45.0,  # Secondary road (major)
    "S1400": 25.0,  # Local neighborhood road
    "S1500": 25.0,  # Vehicular trail
    "S1630": 15.0,  # Ramp
    "S1640": 25.0,  # Service drive
    "S1740": 25.0,  # Internal census use
}

# ---------------------------------------------------------------------------
# Catchment thresholds (minutes) by urbanicity
# ---------------------------------------------------------------------------
CATCHMENT_THRESHOLDS: dict[str, float] = {
    "urban": 10.0,       # population density > 2000 / sq mi
    "suburban": 20.0,    # population density 500–2000 / sq mi
    "rural": 30.0,       # population density < 500 / sq mi
}

# Density breakpoints (persons per square mile)
URBAN_DENSITY_THRESHOLD = 2000.0
SUBURBAN_DENSITY_THRESHOLD = 500.0

# ---------------------------------------------------------------------------
# E2SFCA distance-decay parameters
# ---------------------------------------------------------------------------
DECAY_FUNCTION = "gaussian"  # Options: "gaussian", "linear", "exponential"

# ---------------------------------------------------------------------------
# Database (optional — used when PostGIS backend is enabled)
# ---------------------------------------------------------------------------
DB_CONFIG: dict[str, str] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME", "healthcare_access"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# ---------------------------------------------------------------------------
# Data source URLs
# ---------------------------------------------------------------------------
DATA_URLS: dict[str, str] = {
    "tiger_tracts": (
        "https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_42_tract.zip"
    ),
    "tiger_roads": (
        "https://www2.census.gov/geo/tiger/TIGER2024/ROADS/tl_2024_42_roads.zip"
    ),
    "svi_pa": (
        "https://svi.cdc.gov/Documents/Data/2022/csv/"
        "SVI_2022_Pennsylvania.csv"
    ),
    "svi_tract_layer_2022": (
        "https://onemap.cdc.gov/onemapservices/rest/services/SVI/"
        "CDC_ATSDR_Social_Vulnerability_Index_2022_USA/FeatureServer/2/query"
    ),
    "hrsa_health_centers": "https://data.hrsa.gov/data/download",
    "cms_pos": "https://data.cms.gov/",
    # HRSA's gisportal hosts the canonical CMS-approved facility registry.
    # The HIFLD hosted feature layer was deactivated in August 2025, so we
    # repoint the hospital / urgent-care tiers at HRSA which sources the
    # same CMS Provider of Services data at the layer level.
    #
    # Layer 1  = Hospitals (CMS approved, short-term acute & specialty)
    # Layer 2  = Critical Access Hospitals
    # Layer 4  = Rural Health Clinics
    # Layer 7  = Ambulatory Surgical Centers
    # Layer 18 = Health Center Service Delivery Sites (FQHC / Look-Alike)
    # Layer 19 = Health Center Look-Alike Sites
    "hifld_hospitals": (
        "https://gisportal.hrsa.gov/server/rest/services/HealthCareFacilities/"
        "HealthCareFacilities/MapServer/1/query"
    ),
    "hrsa_critical_access_hospitals": (
        "https://gisportal.hrsa.gov/server/rest/services/HealthCareFacilities/"
        "HealthCareFacilities/MapServer/2/query"
    ),
    "hrsa_rural_health_clinics": (
        "https://gisportal.hrsa.gov/server/rest/services/HealthCareFacilities/"
        "HealthCareFacilities/MapServer/4/query"
    ),
    "hifld_urgent_care": (
        "https://gisportal.hrsa.gov/server/rest/services/HealthCareFacilities/"
        "HealthCareFacilities/MapServer/7/query"
    ),
    "hrsa_hc_geojson": (
        "https://gisportal.hrsa.gov/server/rest/services/HealthCareFacilities/"
        "HealthCareFacilities/MapServer/18/query"
    ),
    "hrsa_lookalike_sites": (
        "https://gisportal.hrsa.gov/server/rest/services/HealthCareFacilities/"
        "HealthCareFacilities/MapServer/19/query"
    ),
    # CMS Provider of Services file (hospital + non-hospital facilities, CSV).
    # Endpoint intentionally points at a placeholder; the loader falls back to
    # a locally staged CSV when the direct download URL is unavailable.
    "cms_pos_csv": (
        "https://data.cms.gov/provider-of-services/provider-of-services-file-"
        "hospital-non-hospital-facilities/data.csv"
    ),
}
