"""Plotly Dash dashboard for exploring healthcare access disparities.

Run with:
    python dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.express as px

DATA_OUTPUTS = Path(__file__).resolve().parent.parent / "data" / "outputs"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

def _load_data():
    clusters_path = DATA_OUTPUTS / "pa_accessibility_clusters.gpkg"
    facilities_path = DATA_PROCESSED / "pa_facilities.gpkg"

    gdf = gpd.read_file(clusters_path).to_crs(4326)
    if "geoid" not in gdf.columns and "GEOID" in gdf.columns:
        gdf["geoid"] = gdf["GEOID"].astype(str).str.zfill(11)

    for col in ["accessibility_score", "nearest_facility_min", "provider_pop_ratio",
                 "median_household_income", "pct_no_vehicle", "svi_overall", "pop_density_sq_mi"]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

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
    {"label": "% Without Vehicle", "value": "pct_no_vehicle"},
    {"label": "SVI Overall Score", "value": "svi_overall"},
    {"label": "Population Density (per sq mi)", "value": "pop_density_sq_mi"},
]

app.layout = html.Div(
    [
        html.H1("Healthcare Access Disparities — Pennsylvania",
                style={"textAlign": "center", "marginBottom": "0.2rem"}),
        html.P(
            "Interactive explorer for E2SFCA spatial accessibility scores, "
            "demographic overlays, and community typology clusters.",
            style={"textAlign": "center", "color": "#666"},
        ),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Controls"),
                        html.Label("Accessibility metric:"),
                        dcc.Dropdown(id="metric-dropdown", options=METRIC_OPTIONS,
                                     value="accessibility_score", clearable=False),
                        html.Br(),
                        html.Label("Demographic overlay:"),
                        dcc.Dropdown(id="overlay-dropdown", options=OVERLAY_OPTIONS,
                                     value="none", clearable=False),
                        html.Br(),
                        html.Label("Show facilities:"),
                        dcc.Checklist(id="show-facilities",
                                      options=[{"label": " Facility locations", "value": "show"}],
                                      value=["show"]),
                        html.Hr(),
                        html.Div(id="summary-stats", style={"fontSize": "0.9rem"}),
                    ],
                    style={"width": "22%", "display": "inline-block",
                           "verticalAlign": "top", "padding": "1rem",
                           "backgroundColor": "#f9f9f9", "borderRadius": "8px"},
                ),
                html.Div(
                    [dcc.Graph(id="main-map", style={"height": "78vh"})],
                    style={"width": "76%", "display": "inline-block", "verticalAlign": "top"},
                ),
            ]
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "margin": "1rem"},
)


@app.callback(
    Output("main-map", "figure"),
    Output("summary-stats", "children"),
    Input("metric-dropdown", "value"),
    Input("overlay-dropdown", "value"),
    Input("show-facilities", "value"),
)
def update_map(metric: str, overlay: str, show_fac: list[str]):
    color_col = metric if metric != "cluster" else "cluster"
    is_categorical = metric == "cluster"

    hover_data = {
        "geoid": True,
        "accessibility_score": ":.6f",
        "nearest_facility_min": ":.1f",
        "facilities_in_catchment": True,
    }

    if is_categorical:
        fig = px.choropleth_mapbox(
            tracts_gdf, geojson=geojson_data,
            locations=tracts_gdf.index, color=color_col,
            mapbox_style="carto-positron",
            center={"lat": 40.9, "lon": -77.8}, zoom=6.3,
            opacity=0.7,
            hover_data=hover_data,
            title=f"PA Census Tracts — {metric}",
        )
    else:
        fig = px.choropleth_mapbox(
            tracts_gdf, geojson=geojson_data,
            locations=tracts_gdf.index, color=color_col,
            color_continuous_scale="RdYlGn" if metric != "nearest_facility_min" else "RdYlGn_r",
            mapbox_style="carto-positron",
            center={"lat": 40.9, "lon": -77.8}, zoom=6.3,
            opacity=0.7,
            hover_data=hover_data,
            title=f"PA Census Tracts — {metric}",
        )

    if overlay != "none" and overlay in tracts_gdf.columns and not is_categorical:
        pass

    if show_fac and "show" in show_fac and not facilities_gdf.empty:
        fac_lat = facilities_gdf.geometry.y
        fac_lon = facilities_gdf.geometry.x
        fac_name = facilities_gdf["facility_name"] if "facility_name" in facilities_gdf.columns else ""
        fig.add_scattermapbox(
            lat=fac_lat, lon=fac_lon,
            mode="markers",
            marker=dict(size=4, color="#2b8cbe", opacity=0.6),
            text=fac_name,
            name="Facilities",
            hoverinfo="text",
        )

    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

    col_data = tracts_gdf[metric] if metric in tracts_gdf.columns and not is_categorical else None
    if col_data is not None:
        stats = html.Div([
            html.Strong(f"Summary: {metric}"),
            html.Br(),
            html.Span(f"Mean: {col_data.mean():.4f}"),
            html.Br(),
            html.Span(f"Median: {col_data.median():.4f}"),
            html.Br(),
            html.Span(f"Min: {col_data.min():.4f}"),
            html.Br(),
            html.Span(f"Max: {col_data.max():.4f}"),
            html.Br(),
            html.Span(f"Std: {col_data.std():.4f}"),
            html.Br(), html.Br(),
            html.Span(f"Tracts: {len(tracts_gdf)}"),
            html.Br(),
            html.Span(f"Facilities: {len(facilities_gdf)}"),
        ])
    else:
        stats = html.Div([
            html.Span(f"Tracts: {len(tracts_gdf)}"),
            html.Br(),
            html.Span(f"Facilities: {len(facilities_gdf)}"),
        ])

    return fig, stats


if __name__ == "__main__":
    app.run(debug=True)
