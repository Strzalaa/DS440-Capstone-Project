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
from typing import Optional

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
    raise NotImplementedError


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
    raise NotImplementedError


def fetch_acs_data(
    state_fips: str = STATE_FIPS,
    year: int = 2022,
    variables: Optional[dict[str, str]] = None,
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
