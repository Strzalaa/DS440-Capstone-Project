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
    if tracts.empty:
        return folium.Map(location=[40.9, -77.8], zoom_start=7, tiles="cartodbpositron")

    work = tracts.to_crs(4326).copy()
    work = work.reset_index(drop=True)
    work["_row_id"] = work.index.astype(str)
    center = work.geometry.union_all().centroid
    fmap = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=work.__geo_interface__,
        data=work,
        columns=["_row_id", value_col],
        key_on="feature.properties._row_id",
        fill_color=cmap,
        fill_opacity=0.8,
        line_opacity=0.2,
        legend_name=title,
        nan_fill_color="lightgray",
    ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    return fmap


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
    work = tracts.to_crs(4326).copy().reset_index(drop=True)
    work["_row_id"] = work.index.astype(str)
    layer = folium.FeatureGroup(name=label, show=False)
    choropleth = folium.Choropleth(
        geo_data=work.__geo_interface__,
        data=work,
        columns=["_row_id", demographic_col],
        key_on="feature.properties._row_id",
        fill_color=cmap,
        fill_opacity=opacity,
        line_opacity=0.1,
        legend_name=label,
        nan_fill_color="lightgray",
    )
    choropleth.add_to(layer)
    layer.add_to(base_map)
    folium.LayerControl().add_to(base_map)
    return base_map


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
    fac = facilities.to_crs(4326).copy()
    if fac.empty:
        return base_map or folium.Map(location=[40.9, -77.8], zoom_start=7, tiles="cartodbpositron")

    if base_map is None:
        center = fac.geometry.union_all().centroid
        base_map = folium.Map(location=[center.y, center.x], zoom_start=7, tiles="cartodbpositron")

    popup_cols = popup_cols or [c for c in ["facility_name", "source", "provider_count"] if c in fac.columns]
    for _, row in fac.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        popup_html = "<br>".join(f"{col}: {row.get(col, '')}" for col in popup_cols)
        folium.CircleMarker(
            location=[geom.y, geom.x],
            radius=3,
            color="#2b8cbe",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(base_map)
    return base_map


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
    work = tracts.to_crs(4326).copy().reset_index(drop=True)
    if work.empty:
        return folium.Map(location=[40.9, -77.8], zoom_start=7, tiles="cartodbpositron")

    work[cluster_col] = work[cluster_col].astype("category")
    work["_cluster_code"] = work[cluster_col].cat.codes
    cmap_name = "Set1" if work["_cluster_code"].nunique() <= 9 else "tab20"
    return choropleth_map(
        tracts=work,
        value_col="_cluster_code",
        title=title,
        cmap=cmap_name,
    )


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
    k_values = list(k_range)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=inertias,
            mode="lines+markers",
            name="Inertia",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=silhouettes,
            mode="lines+markers",
            name="Silhouette",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="K-Means Model Selection",
        xaxis=dict(title="k (number of clusters)"),
        yaxis=dict(title="Inertia"),
        yaxis2=dict(title="Silhouette", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        template="plotly_white",
    )
    return fig


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
