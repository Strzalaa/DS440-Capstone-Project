"""Mapping and charting utilities for healthcare access visualisations.

Provides reusable helpers for choropleth maps, demographic overlays,
facility point maps, and cluster displays using Folium and Plotly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import folium
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go


def choropleth_map(
    tracts: gpd.GeoDataFrame,
    value_col: str = "accessibility_score",
    title: str = "Healthcare Accessibility Score",
    cmap: str = "RdYlGn",
    classification: str = "quantiles",
    k: int = 5,
) -> folium.Map:
    """Create a Folium choropleth map of tract-level values.

    Parameters
    ----------
    tracts : gpd.GeoDataFrame
        Census tracts with the value column.
    value_col : str
        Numeric column to visualise.
    title : str
        Map legend title.
    cmap : str
        Colour palette name.
    classification : str
        Classification scheme (``"quantiles"``, ``"natural_breaks"``,
        ``"equal_interval"``).
    k : int
        Number of classes.

    Returns
    -------
    folium.Map
    """
    raise NotImplementedError


def demographic_overlay(
    base_map: folium.Map,
    tracts: gpd.GeoDataFrame,
    demographic_col: str,
    label: str,
    cmap: str = "YlOrRd",
    opacity: float = 0.5,
) -> folium.Map:
    """Add a toggle-able demographic overlay to an existing Folium map.

    Parameters
    ----------
    base_map : folium.Map
        Existing map to add the layer to.
    tracts : gpd.GeoDataFrame
        Census tracts with the demographic column.
    demographic_col : str
        Column to overlay.
    label : str
        Layer name shown in the layer control.
    cmap : str
        Colour palette.
    opacity : float
        Layer opacity.

    Returns
    -------
    folium.Map
        Map with the new overlay added.
    """
    raise NotImplementedError


def facility_map(
    facilities: gpd.GeoDataFrame,
    base_map: Optional[folium.Map] = None,
    popup_cols: Optional[list[str]] = None,
) -> folium.Map:
    """Plot healthcare facilities as interactive point markers.

    Parameters
    ----------
    facilities : gpd.GeoDataFrame
        Facility points.
    base_map : folium.Map | None
        If provided, facilities are added as a layer; otherwise a new map
        is created.
    popup_cols : list[str] | None
        Columns to include in click popups.

    Returns
    -------
    folium.Map
    """
    raise NotImplementedError


def cluster_map(
    tracts: gpd.GeoDataFrame,
    cluster_col: str = "cluster",
    title: str = "Community Typology Clusters",
) -> folium.Map:
    """Colour-coded map of cluster membership.

    Parameters
    ----------
    tracts : gpd.GeoDataFrame
        Tracts with cluster labels.
    cluster_col : str
        Column containing integer or categorical cluster labels.
    title : str
        Map title.

    Returns
    -------
    folium.Map
    """
    raise NotImplementedError


def plot_elbow(
    inertias: list[float],
    silhouettes: list[float],
    k_range: range,
) -> go.Figure:
    """Plot the elbow curve and silhouette scores for k-means selection.

    Parameters
    ----------
    inertias : list[float]
        Within-cluster sum of squares for each k.
    silhouettes : list[float]
        Silhouette scores for each k.
    k_range : range
        Candidate k values.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    raise NotImplementedError


def save_map(fmap: folium.Map, path: Path) -> None:
    """Save a Folium map to an HTML file.

    Parameters
    ----------
    fmap : folium.Map
        Map object.
    path : Path
        Output file path.
    """
    fmap.save(str(path))
