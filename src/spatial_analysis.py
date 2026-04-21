"""Spatial accessibility analysis using the Enhanced Two-Step Floating Catchment Area method.

Implements drive-time estimation, distance-decay functions,
the E2SFCA algorithm, and sensitivity analysis utilities.

Drive-time estimation uses a Euclidean distance + circuity factor approach
rather than full road-network routing, trading a small accuracy penalty for
orders-of-magnitude faster execution (~seconds vs. hours on 596K road segments).
"""

from __future__ import annotations

from typing import Literal, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.config import (
    CATCHMENT_THRESHOLDS,
    DECAY_FUNCTION,
    ROAD_SPEEDS_MPH,
    SUBURBAN_DENSITY_THRESHOLD,
    URBAN_DENSITY_THRESHOLD,
)

CIRCUITY_FACTORS: dict[str, float] = {
    "urban": 1.20,
    "suburban": 1.30,
    "rural": 1.40,
}
AVG_ROAD_SPEED_MPH: float = 35.0


def _project_points(gdf: gpd.GeoDataFrame, epsg: int) -> gpd.GeoDataFrame:
    """Project geometries safely, converting polygons to centroids in projected CRS."""
    work = gdf.copy()
    work = work.to_crs(epsg=epsg)
    if not all(work.geometry.geom_type == "Point"):
        work["geometry"] = work.geometry.centroid
    return work


def _nearest_drive_time_series(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    urbanicity: pd.Series | None = None,
) -> pd.Series:
    """Compute nearest-facility drive time for every origin without a cutoff."""
    if origins.empty or destinations.empty:
        return pd.Series(dtype=float)

    origin_id_col = "geoid" if "geoid" in origins.columns else None
    o = _project_points(origins, epsg=32617)
    d = _project_points(destinations, epsg=32617)
    origin_ids = o[origin_id_col].astype(str).values if origin_id_col else o.index.astype(str).values

    if urbanicity is not None:
        urban_arr = urbanicity.astype(str).values
    else:
        urban_arr = np.full(len(o), "suburban")
    circ_arr = np.array([CIRCUITY_FACTORS.get(str(u), 1.30) for u in urban_arr])

    origin_xy = np.column_stack([o.geometry.x.values, o.geometry.y.values])
    dest_xy = np.column_stack([d.geometry.x.values, d.geometry.y.values])
    dest_tree = cKDTree(dest_xy)
    dists_m, _ = dest_tree.query(origin_xy, k=1)
    road_miles = (dists_m * circ_arr) / 1609.344
    travel_min = (road_miles / AVG_ROAD_SPEED_MPH) * 60.0
    return pd.Series(travel_min, index=origin_ids, dtype=float)


def build_road_network(
    roads_gdf: gpd.GeoDataFrame | None = None,
    speed_map: dict[str, float] | None = None,
) -> dict:
    """Prepare spatial index data for fast drive-time estimation.

    Instead of building a full NetworkX graph from 596K road segments,
    this returns a lightweight metadata dict used by :func:`compute_drive_times`.
    The function signature is preserved for backward compatibility.
    """
    return {"method": "euclidean_circuity"}


def compute_drive_times(
    network_info: dict | object,
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    max_minutes: float = 30.0,
    urbanicity: pd.Series | None = None,
) -> pd.DataFrame:
    """Estimate drive times using Euclidean distance with circuity adjustment.

    For each origin-destination pair within the catchment, straight-line
    distance is multiplied by a circuity factor (1.2x urban, 1.3x suburban,
    1.4x rural) then divided by average road speed to yield travel time.

    Parameters
    ----------
    network_info : dict | object
        Output of :func:`build_road_network` (unused, kept for API compat).
    origins : gpd.GeoDataFrame
        Census tract centroids with ``geoid`` column.
    destinations : gpd.GeoDataFrame
        Healthcare facility locations with ``facility_id`` column.
    max_minutes : float
        Cutoff beyond which pairs are excluded.
    urbanicity : pd.Series | None
        Series aligned with *origins* containing 'urban'/'suburban'/'rural'.
        Falls back to 'suburban' if not provided.

    Returns
    -------
    pd.DataFrame
        Columns ``origin_id``, ``destination_id``, ``drive_time_min``.
    """
    if origins.empty or destinations.empty:
        return pd.DataFrame(columns=["origin_id", "destination_id", "drive_time_min"])

    origin_id_col = "geoid" if "geoid" in origins.columns else None
    dest_id_col = "facility_id" if "facility_id" in destinations.columns else None

    o = _project_points(origins, epsg=32617)
    d = _project_points(destinations, epsg=32617)

    origin_ids = o[origin_id_col].astype(str).values if origin_id_col else o.index.astype(str).values
    dest_ids = d[dest_id_col].astype(str).values if dest_id_col else d.index.astype(str).values

    if urbanicity is not None:
        urban_arr = urbanicity.values
    else:
        urban_arr = np.full(len(o), "suburban")

    circ_arr = np.array([CIRCUITY_FACTORS.get(str(u), 1.30) for u in urban_arr])

    max_straight_miles = (max_minutes / 60.0) * AVG_ROAD_SPEED_MPH / min(CIRCUITY_FACTORS.values())
    max_straight_m = max_straight_miles * 1609.344

    origin_xy = np.column_stack([o.geometry.x.values, o.geometry.y.values])
    dest_xy = np.column_stack([d.geometry.x.values, d.geometry.y.values])

    dest_tree = cKDTree(dest_xy)

    results: list[dict] = []
    for i, (ox, oy) in enumerate(origin_xy):
        nearby_idx = dest_tree.query_ball_point([ox, oy], r=max_straight_m)
        if not nearby_idx:
            continue
        near_xy = dest_xy[nearby_idx]
        dists_m = np.sqrt((near_xy[:, 0] - ox) ** 2 + (near_xy[:, 1] - oy) ** 2)
        road_miles = (dists_m * circ_arr[i]) / 1609.344
        travel_min = (road_miles / AVG_ROAD_SPEED_MPH) * 60.0

        mask = travel_min <= max_minutes
        for j_local in np.where(mask)[0]:
            j_global = nearby_idx[j_local]
            results.append(
                {
                    "origin_id": str(origin_ids[i]),
                    "destination_id": str(dest_ids[j_global]),
                    "drive_time_min": float(travel_min[j_local]),
                }
            )

    return pd.DataFrame(results)


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
    catchment_thresholds: Optional[dict[str, float]] = None,
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
    if decay not in DECAY_FUNCTIONS:
        raise ValueError(f"Unsupported decay function: {decay}")

    tracts_out = tracts.copy()
    if "urbanicity" not in tracts_out.columns:
        if "pop_density_sq_mi" in tracts_out.columns:
            tracts_out["urbanicity"] = classify_urbanicity(tracts_out).astype(str)
        else:
            tracts_out["urbanicity"] = "suburban"

    tract_id_col = "geoid" if "geoid" in tracts_out.columns else None
    facility_id_col = "facility_id" if "facility_id" in facilities.columns else None

    tract_ids = tracts_out[tract_id_col].astype(str) if tract_id_col else tracts_out.index.astype(str)
    facility_ids = facilities[facility_id_col].astype(str) if facility_id_col else facilities.index.astype(str)
    tracts_out["_tract_id"] = tract_ids.values
    facilities_work = facilities.copy()
    facilities_work["_facility_id"] = facility_ids.values

    thresholds = catchment_thresholds or CATCHMENT_THRESHOLDS
    decay_fn = DECAY_FUNCTIONS[decay]

    pairs = drive_times.copy()
    if pairs.empty:
        tracts_out["accessibility_score"] = 0.0
        nearest_all = _nearest_drive_time_series(
            tracts_out,
            facilities_work,
            tracts_out["urbanicity"],
        )
        tracts_out["nearest_facility_min"] = tracts_out["_tract_id"].map(nearest_all)
        tracts_out["facilities_in_catchment"] = 0
        tracts_out["provider_pop_ratio"] = 0.0
        return tracts_out.drop(columns=["_tract_id"])

    pairs["origin_id"] = pairs["origin_id"].astype(str)
    pairs["destination_id"] = pairs["destination_id"].astype(str)

    tract_meta = tracts_out[["_tract_id", population_col, "urbanicity"]].rename(columns={"_tract_id": "origin_id"})
    fac_meta = facilities_work[["_facility_id", provider_col]].rename(columns={"_facility_id": "destination_id"})
    pairs = pairs.merge(tract_meta, on="origin_id", how="inner")
    pairs = pairs.merge(fac_meta, on="destination_id", how="inner")

    pairs["threshold"] = pairs["urbanicity"].map(thresholds).fillna(thresholds.get("suburban", 20.0))
    pairs = pairs[pairs["drive_time_min"] <= pairs["threshold"]].copy()
    nearest_all = _nearest_drive_time_series(
        tracts_out,
        facilities_work,
        tracts_out["urbanicity"],
    )
    if pairs.empty:
        tracts_out["accessibility_score"] = 0.0
        tracts_out["nearest_facility_min"] = tracts_out["_tract_id"].map(nearest_all)
        tracts_out["facilities_in_catchment"] = 0
        tracts_out["provider_pop_ratio"] = 0.0
        return tracts_out.drop(columns=["_tract_id"])

    pairs["weight"] = decay_fn(pairs["drive_time_min"].to_numpy(dtype=float), pairs["threshold"].to_numpy(dtype=float))
    pairs["weighted_pop"] = pairs[population_col] * pairs["weight"]

    denom = pairs.groupby("destination_id", as_index=True)["weighted_pop"].sum().replace(0.0, np.nan)
    providers = facilities_work.set_index("_facility_id")[provider_col].astype(float)
    facility_ratio = (providers / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pairs["facility_ratio"] = pairs["destination_id"].map(facility_ratio).fillna(0.0)
    pairs["ratio_weighted"] = pairs["facility_ratio"] * pairs["weight"]

    scores = pairs.groupby("origin_id", as_index=True)["ratio_weighted"].sum()
    facility_counts = pairs.groupby("origin_id", as_index=True)["destination_id"].nunique()
    providers_in_range = pairs.groupby("origin_id", as_index=True)[provider_col].sum()
    populations = tracts_out.set_index("_tract_id")[population_col].replace(0.0, np.nan)
    provider_pop_ratio = ((providers_in_range / populations) * 1000.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tracts_out["accessibility_score"] = tracts_out["_tract_id"].map(scores).fillna(0.0)
    tracts_out["nearest_facility_min"] = tracts_out["_tract_id"].map(nearest_all)
    tracts_out["facilities_in_catchment"] = tracts_out["_tract_id"].map(facility_counts).fillna(0).astype(int)
    tracts_out["provider_pop_ratio"] = tracts_out["_tract_id"].map(provider_pop_ratio).fillna(0.0)
    return tracts_out.drop(columns=["_tract_id"])


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
    results: list[pd.DataFrame] = []
    base_thresholds = CATCHMENT_THRESHOLDS.copy()

    tract_id_col = "geoid" if "geoid" in tracts.columns else None
    tract_ids = tracts[tract_id_col].astype(str) if tract_id_col else tracts.index.astype(str)

    for offset in threshold_offsets:
        adjusted = {
            k: max(1.0, float(v + offset))
            for k, v in base_thresholds.items()
        }
        for decay_name in decay_functions:
            out = e2sfca(
                tracts=tracts,
                facilities=facilities,
                drive_times=drive_times,
                decay=decay_name,  # type: ignore[arg-type]
                catchment_thresholds=adjusted,
            )
            run_df = pd.DataFrame(
                {
                    "tract_id": tract_ids.values,
                    "threshold_offset": offset,
                    "decay": decay_name,
                    "accessibility_score": out["accessibility_score"].values,
                }
            )
            results.append(run_df)

    if not results:
        return pd.DataFrame(columns=["tract_id", "threshold_offset", "decay", "accessibility_score"])
    return pd.concat(results, ignore_index=True)
