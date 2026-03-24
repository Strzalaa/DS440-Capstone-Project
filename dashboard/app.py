"""Plotly Dash dashboard for exploring healthcare access disparities.

Run with:
    python dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.express as px
import plotly.graph_objects as go

DATA_OUTPUTS = Path(__file__).resolve().parent.parent / "data" / "outputs"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def _load_data():
    clusters_path = DATA_OUTPUTS / "pa_accessibility_clusters.gpkg"
    facilities_path = DATA_PROCESSED / "pa_facilities.gpkg"

    gdf = gpd.read_file(clusters_path).to_crs(4326)
    if "geoid" not in gdf.columns and "GEOID" in gdf.columns:
        gdf["geoid"] = gdf["GEOID"].astype(str).str.zfill(11)

    numeric_cols = [
        "accessibility_score", "nearest_facility_min", "provider_pop_ratio",
        "median_household_income", "pct_uninsured", "pct_no_vehicle",
        "svi_overall", "pop_density_sq_mi", "facilities_in_catchment",
        "total_population",
    ]
    for col in numeric_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    # Census uses -666666666 as a sentinel for missing income data
    for col in ["median_household_income"]:
        if col in gdf.columns:
            gdf.loc[gdf[col] < 0, col] = np.nan

    # Simplify geometries to reduce browser payload (~35 MB -> ~5 MB)
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.005, preserve_topology=True)

    fac = gpd.read_file(facilities_path).to_crs(4326)
    fac = fac[fac.geometry.notna() & ~fac.geometry.is_empty].copy()
    return gdf, fac


tracts_gdf, facilities_gdf = _load_data()
geojson_data = json.loads(tracts_gdf.to_json())

app = Dash(__name__, title="Healthcare Access Dashboard")

METRIC_OPTIONS = [
    {"label": "Accessibility Score (E2SFCA)", "value": "accessibility_score"},
    {"label": "Drive Time to Nearest Facility (min)", "value": "nearest_facility_min"},
    {"label": "Provider-to-Population Ratio (per 1K)", "value": "provider_pop_ratio"},
    {"label": "Cluster Assignment", "value": "cluster"},
]
OVERLAY_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "Median Household Income", "value": "median_household_income"},
    {"label": "% Uninsured", "value": "pct_uninsured"},
    {"label": "% Without Vehicle", "value": "pct_no_vehicle"},
    {"label": "SVI Overall Score", "value": "svi_overall"},
    {"label": "Population Density (per sq mi)", "value": "pop_density_sq_mi"},
]

FRIENDLY_NAMES = {
    "accessibility_score": "Accessibility Score",
    "nearest_facility_min": "Drive Time (min)",
    "provider_pop_ratio": "Providers per 1K Pop",
    "cluster": "Cluster",
    "median_household_income": "Median Income ($)",
    "pct_uninsured": "% Uninsured",
    "pct_no_vehicle": "% No Vehicle",
    "svi_overall": "SVI Score",
    "pop_density_sq_mi": "Pop Density (per sq mi)",
}

app.layout = html.Div(
    [
        html.H1(
            "Healthcare Access Disparities — Pennsylvania",
            style={"textAlign": "center", "marginBottom": "0.2rem"},
        ),
        html.P(
            "Interactive explorer for E2SFCA spatial accessibility scores, "
            "demographic overlays, and community typology clusters.",
            style={"textAlign": "center", "color": "#666"},
        ),
        html.Hr(),
        html.Div(
            [
                # Left sidebar
                html.Div(
                    [
                        html.H3("Controls"),
                        html.Label("Accessibility metric:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=METRIC_OPTIONS,
                            value="accessibility_score",
                            clearable=False,
                        ),
                        html.Br(),
                        html.Label("Demographic overlay:", style={"fontWeight": "bold"}),
                        dcc.Dropdown(
                            id="overlay-dropdown",
                            options=OVERLAY_OPTIONS,
                            value="none",
                            clearable=False,
                        ),
                        html.Br(),
                        html.Label("Show facilities:", style={"fontWeight": "bold"}),
                        dcc.Checklist(
                            id="show-facilities",
                            options=[{"label": " Facility locations", "value": "show"}],
                            value=["show"],
                        ),
                        html.Hr(),
                        html.Div(id="summary-stats", style={"fontSize": "0.85rem"}),
                    ],
                    style={
                        "width": "22%", "display": "inline-block",
                        "verticalAlign": "top", "padding": "1rem",
                        "backgroundColor": "#f9f9f9", "borderRadius": "8px",
                    },
                ),
                # Main map area
                html.Div(
                    [dcc.Graph(id="main-map", style={"height": "80vh"})],
                    style={"width": "76%", "display": "inline-block", "verticalAlign": "top"},
                ),
            ]
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "margin": "1rem"},
)


def _build_hover_text(gdf: gpd.GeoDataFrame) -> list[str]:
    """Build a rich hover tooltip for each tract."""
    texts = []
    for _, row in gdf.iterrows():
        parts = []
        if "geoid" in gdf.columns:
            parts.append(f"Tract: {row['geoid']}")
        if "accessibility_score" in gdf.columns:
            parts.append(f"Access Score: {row['accessibility_score']:.6f}")
        if "nearest_facility_min" in gdf.columns:
            val = row["nearest_facility_min"]
            parts.append(f"Nearest Facility: {val:.1f} min" if pd.notna(val) else "Nearest Facility: N/A")
        if "facilities_in_catchment" in gdf.columns:
            parts.append(f"Facilities in Catchment: {int(row['facilities_in_catchment'])}")
        if "total_population" in gdf.columns and pd.notna(row.get("total_population")):
            parts.append(f"Population: {int(row['total_population']):,}")
        if "median_household_income" in gdf.columns and pd.notna(row.get("median_household_income")):
            parts.append(f"Median Income: ${int(row['median_household_income']):,}")
        texts.append("<br>".join(parts))
    return texts


hover_texts = _build_hover_text(tracts_gdf)


@app.callback(
    Output("main-map", "figure"),
    Output("summary-stats", "children"),
    Input("metric-dropdown", "value"),
    Input("overlay-dropdown", "value"),
    Input("show-facilities", "value"),
)
def update_map(metric: str, overlay: str, show_fac: list[str]):
    is_categorical = metric == "cluster"

    fig = go.Figure()

    if is_categorical:
        cluster_vals = tracts_gdf["cluster"].astype(str)
        unique_clusters = sorted(cluster_vals.unique())
        color_map = px.colors.qualitative.Set2
        cluster_color = [color_map[int(c) % len(color_map)] for c in cluster_vals]

        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=tracts_gdf.index,
            z=tracts_gdf["cluster"].astype(float),
            colorscale=[[i / max(1, len(unique_clusters) - 1), color_map[i % len(color_map)]]
                        for i in range(len(unique_clusters))],
            marker_opacity=0.7,
            marker_line_width=0.3,
            marker_line_color="white",
            text=hover_texts,
            hoverinfo="text",
            colorbar=dict(title="Cluster", tickvals=list(range(len(unique_clusters))),
                          ticktext=unique_clusters),
            name="Clusters",
        ))
    else:
        color_vals = tracts_gdf[metric].copy()
        reverse = metric == "nearest_facility_min"
        colorscale = "RdYlGn_r" if reverse else "RdYlGn"

        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=tracts_gdf.index,
            z=color_vals,
            colorscale=colorscale,
            marker_opacity=0.7,
            marker_line_width=0.3,
            marker_line_color="white",
            text=hover_texts,
            hoverinfo="text",
            colorbar=dict(title=FRIENDLY_NAMES.get(metric, metric)),
            name=FRIENDLY_NAMES.get(metric, metric),
            zmin=float(color_vals.quantile(0.02)) if color_vals.notna().any() else 0,
            zmax=float(color_vals.quantile(0.98)) if color_vals.notna().any() else 1,
        ))

    # Demographic overlay as a second semi-transparent layer
    if overlay != "none" and overlay in tracts_gdf.columns and not is_categorical:
        overlay_vals = tracts_gdf[overlay].copy()
        fig.add_trace(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=tracts_gdf.index,
            z=overlay_vals,
            colorscale="YlOrRd",
            marker_opacity=0.35,
            marker_line_width=0,
            text=[f"{FRIENDLY_NAMES.get(overlay, overlay)}: {v:.2f}" if pd.notna(v) else "N/A"
                  for v in overlay_vals],
            hoverinfo="text",
            colorbar=dict(title=FRIENDLY_NAMES.get(overlay, overlay), x=1.08),
            name=FRIENDLY_NAMES.get(overlay, overlay),
            zmin=float(overlay_vals.quantile(0.02)) if overlay_vals.notna().any() else 0,
            zmax=float(overlay_vals.quantile(0.98)) if overlay_vals.notna().any() else 1,
        ))

    # Facility point markers
    if show_fac and "show" in show_fac and not facilities_gdf.empty:
        fac_lat = facilities_gdf.geometry.y.tolist()
        fac_lon = facilities_gdf.geometry.x.tolist()
        fac_text = (
            facilities_gdf["facility_name"].tolist()
            if "facility_name" in facilities_gdf.columns
            else [""] * len(facilities_gdf)
        )
        fig.add_trace(go.Scattermapbox(
            lat=fac_lat,
            lon=fac_lon,
            mode="markers",
            marker=dict(size=4, color="#2b8cbe", opacity=0.6),
            text=fac_text,
            name="Facilities",
            hoverinfo="text",
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=40.9, lon=-77.8),
            zoom=6.3,
        ),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title=f"PA Census Tracts — {FRIENDLY_NAMES.get(metric, metric)}",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
    )

    # Summary statistics panel
    col_data = tracts_gdf[metric] if metric in tracts_gdf.columns and not is_categorical else None
    stat_parts = []
    if col_data is not None:
        stat_parts.extend([
            html.Strong(FRIENDLY_NAMES.get(metric, metric)),
            html.Br(),
            html.Span(f"Mean: {col_data.mean():.4f}"),
            html.Br(),
            html.Span(f"Median: {col_data.median():.4f}"),
            html.Br(),
            html.Span(f"Min: {col_data.min():.4f}"),
            html.Br(),
            html.Span(f"Max: {col_data.max():.4f}"),
            html.Br(),
            html.Span(f"Std Dev: {col_data.std():.4f}"),
            html.Br(),
            html.Span(f"Zero-score tracts: {int((col_data == 0).sum())}"),
            html.Br(), html.Br(),
        ])
    elif is_categorical and "cluster" in tracts_gdf.columns:
        vc = tracts_gdf["cluster"].value_counts().sort_index()
        stat_parts.append(html.Strong("Cluster Breakdown"))
        stat_parts.append(html.Br())
        for cluster_id, count in vc.items():
            stat_parts.append(html.Span(f"Cluster {cluster_id}: {count} tracts"))
            stat_parts.append(html.Br())
        stat_parts.append(html.Br())

    if overlay != "none" and overlay in tracts_gdf.columns:
        ov_data = tracts_gdf[overlay]
        stat_parts.extend([
            html.Strong(f"Overlay: {FRIENDLY_NAMES.get(overlay, overlay)}"),
            html.Br(),
            html.Span(f"Mean: {ov_data.mean():.2f}"),
            html.Br(),
            html.Span(f"Median: {ov_data.median():.2f}"),
            html.Br(), html.Br(),
        ])

    stat_parts.extend([
        html.Span(f"Total tracts: {len(tracts_gdf):,}"),
        html.Br(),
        html.Span(f"Geocoded facilities: {len(facilities_gdf):,}"),
    ])

    return fig, html.Div(stat_parts)


if __name__ == "__main__":
    app.run(debug=True)
