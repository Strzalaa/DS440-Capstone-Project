"""Spatial accessibility analysis using the Enhanced Two-Step Floating Catchment Area method.

Implements network-based drive-time calculations, distance-decay functions,
the E2SFCA algorithm, and sensitivity analysis utilities.
"""

from __future__ import annotations

from typing import Literal, Optional

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from src.config import (
    CATCHMENT_THRESHOLDS,
    DECAY_FUNCTION,
    ROAD_SPEEDS_MPH,
    SUBURBAN_DENSITY_THRESHOLD,
    URBAN_DENSITY_THRESHOLD,
)


def build_road_network(
    roads_gdf: gpd.GeoDataFrame,
    speed_map: dict[str, float] = ROAD_SPEEDS_MPH,
) -> nx.DiGraph:
    """Construct a routable network graph from TIGER/Line road geometries.

    Edge weights are travel-time in minutes derived from segment length and
    the speed assignment for the road's MTFCC classification.

    Parameters
    ----------
    roads_gdf : gpd.GeoDataFrame
        TIGER/Line roads with ``MTFCC`` and ``geometry`` columns.
    speed_map : dict[str, float]
        MTFCC code -> speed in mph.

    Returns
    -------
    nx.DiGraph
        Directed graph with travel-time edge weights.
    """
    raise NotImplementedError


def compute_drive_times(
    graph: nx.DiGraph,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    max_minutes: float = 30.0,
) -> pd.DataFrame:
    """Compute shortest-path drive times between origins and destinations.

    Parameters
    ----------
    graph : nx.DiGraph
        Road network graph from :func:`build_road_network`.
    origins : gpd.GeoDataFrame
        Census tract centroids (or population-weighted centroids).
    destinations : gpd.GeoDataFrame
        Healthcare facility locations.
    max_minutes : float
        Cutoff beyond which pairs are excluded.

    Returns
    -------
    pd.DataFrame
        Columns ``origin_id``, ``destination_id``, ``drive_time_min``.
    """
    raise NotImplementedError


def gaussian_decay(distance: np.ndarray, threshold: float) -> np.ndarray:
    """Apply Gaussian distance-decay weighting.

    Parameters
    ----------
    distance : np.ndarray
        Travel times (minutes).
    threshold : float
        Catchment threshold (minutes) controlling decay width.

    Returns
    -------
    np.ndarray
        Weights in [0, 1].
    """
    return np.exp(-((distance / threshold) ** 2))


def linear_decay(distance: np.ndarray, threshold: float) -> np.ndarray:
    """Linear distance-decay: weight decreases linearly to zero at threshold."""
    weights = 1.0 - (distance / threshold)
    return np.clip(weights, 0.0, 1.0)


def exponential_decay(distance: np.ndarray, threshold: float) -> np.ndarray:
    """Exponential distance-decay function."""
    beta = np.log(2) / (threshold / 2)
    return np.exp(-beta * distance)


DECAY_FUNCTIONS = {
    "gaussian": gaussian_decay,
    "linear": linear_decay,
    "exponential": exponential_decay,
}


def classify_urbanicity(
    tracts: gpd.GeoDataFrame,
    pop_density_col: str = "pop_density_sq_mi",
) -> pd.Series:
    """Classify each tract as urban, suburban, or rural by population density.

    Parameters
    ----------
    tracts : gpd.GeoDataFrame
        Census tracts with a population-density column.
    pop_density_col : str
        Column name containing persons per square mile.

    Returns
    -------
    pd.Series
        Categorical series with values ``"urban"``, ``"suburban"``, ``"rural"``.
    """
    density = tracts[pop_density_col]
    return pd.cut(
        density,
        bins=[-np.inf, SUBURBAN_DENSITY_THRESHOLD, URBAN_DENSITY_THRESHOLD, np.inf],
        labels=["rural", "suburban", "urban"],
    )


def e2sfca(
    tracts: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    drive_times: pd.DataFrame,
    population_col: str = "total_population",
    provider_col: str = "provider_count",
    decay: Literal["gaussian", "linear", "exponential"] = DECAY_FUNCTION,
) -> gpd.GeoDataFrame:
    """Run the Enhanced Two-Step Floating Catchment Area analysis.

    Step 1 — For each facility *j*, compute the provider-to-population ratio
    weighted by distance decay across all tracts within the catchment.

    Step 2 — For each tract *i*, sum the ratios of all reachable facilities
    weighted by distance decay.

    Parameters
    ----------
    tracts : gpd.GeoDataFrame
        Census tracts with population and urbanicity columns.
    facilities : gpd.GeoDataFrame
        Healthcare facilities with provider counts.
    drive_times : pd.DataFrame
        Pairwise drive times from :func:`compute_drive_times`.
    population_col : str
        Column in *tracts* with total population.
    provider_col : str
        Column in *facilities* with provider / capacity count.
    decay : str
        Decay function name (key into ``DECAY_FUNCTIONS``).

    Returns
    -------
    gpd.GeoDataFrame
        *tracts* augmented with columns:
        ``accessibility_score``, ``nearest_facility_min``,
        ``facilities_in_catchment``, ``provider_pop_ratio``.
    """
    raise NotImplementedError


def sensitivity_analysis(
    tracts: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    drive_times: pd.DataFrame,
    threshold_offsets: list[float] = [-5.0, 0.0, 5.0],
    decay_functions: list[str] = ["gaussian", "linear", "exponential"],
) -> pd.DataFrame:
    """Run E2SFCA under multiple parameter configurations.

    Parameters
    ----------
    tracts : gpd.GeoDataFrame
        Census tracts.
    facilities : gpd.GeoDataFrame
        Healthcare facilities.
    drive_times : pd.DataFrame
        Pairwise drive times.
    threshold_offsets : list[float]
        Minutes to add/subtract from base catchment thresholds.
    decay_functions : list[str]
        Decay function names to test.

    Returns
    -------
    pd.DataFrame
        Accessibility scores for every (tract, parameter-set) combination
        with columns ``tract_id``, ``threshold_offset``, ``decay``,
        ``accessibility_score``.
    """
    raise NotImplementedError
