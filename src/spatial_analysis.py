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
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString

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
    roads = roads_gdf.to_crs(3857)
    graph = nx.DiGraph()

    for _, row in roads.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        mtfcc = str(row.get("MTFCC", "S1400"))
        speed_mph = float(speed_map.get(mtfcc, speed_map.get("S1400", 25.0)))
        if speed_mph <= 0:
            speed_mph = 25.0

        segments = [geom] if isinstance(geom, LineString) else list(geom.geoms) if isinstance(geom, MultiLineString) else []
        for segment in segments:
            coords = list(segment.coords)
            for start, end in zip(coords[:-1], coords[1:]):
                sx, sy = start
                ex, ey = end
                length_m = float(LineString([start, end]).length)
                if length_m <= 0:
                    continue
                miles = length_m / 1609.344
                travel_time_min = (miles / speed_mph) * 60.0

                start_node = (sx, sy)
                end_node = (ex, ey)
                graph.add_edge(start_node, end_node, travel_time_min=travel_time_min, length_m=length_m, speed_mph=speed_mph)
                graph.add_edge(end_node, start_node, travel_time_min=travel_time_min, length_m=length_m, speed_mph=speed_mph)

    return graph


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
    if graph.number_of_nodes() == 0 or origins.empty or destinations.empty:
        return pd.DataFrame(columns=["origin_id", "destination_id", "drive_time_min"])

    origin_id_col = "geoid" if "geoid" in origins.columns else None
    dest_id_col = "facility_id" if "facility_id" in destinations.columns else None

    o = origins.copy()
    d = destinations.copy()
    if not all(o.geometry.geom_type == "Point"):
        o["geometry"] = o.geometry.centroid
    if not all(d.geometry.geom_type == "Point"):
        d["geometry"] = d.geometry.centroid
    o = o.to_crs(3857)
    d = d.to_crs(3857)

    node_coords = np.array(list(graph.nodes()))
    tree = cKDTree(node_coords)

    origin_xy = np.array([(geom.x, geom.y) for geom in o.geometry])
    dest_xy = np.array([(geom.x, geom.y) for geom in d.geometry])
    _, origin_node_idx = tree.query(origin_xy, k=1)
    _, dest_node_idx = tree.query(dest_xy, k=1)
    origin_nodes = [tuple(node_coords[idx]) for idx in origin_node_idx]
    dest_nodes = [tuple(node_coords[idx]) for idx in dest_node_idx]

    origin_ids = o[origin_id_col].astype(str).tolist() if origin_id_col else o.index.astype(str).tolist()
    dest_ids = d[dest_id_col].astype(str).tolist() if dest_id_col else d.index.astype(str).tolist()
    dest_lookup = pd.DataFrame({"destination_id": dest_ids, "dest_node": dest_nodes})

    results: list[dict[str, object]] = []
    for origin_id, origin_node in zip(origin_ids, origin_nodes):
        path_lengths = nx.single_source_dijkstra_path_length(
            graph,
            origin_node,
            cutoff=max_minutes,
            weight="travel_time_min",
        )
        reachable = dest_lookup[dest_lookup["dest_node"].isin(path_lengths.keys())]
        for _, row in reachable.iterrows():
            drive_time = float(path_lengths[row["dest_node"]])
            results.append(
                {
                    "origin_id": origin_id,
                    "destination_id": str(row["destination_id"]),
                    "drive_time_min": drive_time,
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
        tracts_out["nearest_facility_min"] = np.nan
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
    if pairs.empty:
        tracts_out["accessibility_score"] = 0.0
        tracts_out["nearest_facility_min"] = np.nan
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
    nearest = drive_times.groupby("origin_id", as_index=True)["drive_time_min"].min()
    facility_counts = pairs.groupby("origin_id", as_index=True)["destination_id"].nunique()
    providers_in_range = pairs.groupby("origin_id", as_index=True)[provider_col].sum()
    populations = tracts_out.set_index("_tract_id")[population_col].replace(0.0, np.nan)
    provider_pop_ratio = ((providers_in_range / populations) * 1000.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tracts_out["accessibility_score"] = tracts_out["_tract_id"].map(scores).fillna(0.0)
    tracts_out["nearest_facility_min"] = tracts_out["_tract_id"].map(nearest)
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
