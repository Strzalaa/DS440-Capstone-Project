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
CENSUS_API_KEY: str = os.getenv("CENSUS_API_KEY", "")

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
    "svi_pa": (
        "https://svi.cdc.gov/Documents/Data/2022/csv/"
        "SVI_2022_Pennsylvania.csv"
    ),
    "hrsa_health_centers": "https://data.hrsa.gov/data/download",
    "cms_pos": "https://data.cms.gov/",
}
