"""Plotly Dash dashboard for exploring healthcare access disparities.

Run with:
    python dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from dash import Dash, Input, Output, dcc, html
import plotly.express as px
import plotly.graph_objects as go

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from src.svi import mask_svi_percentile
from src.clustering import characterize_clusters

DATA_OUTPUTS = Path(__file__).resolve().parent.parent / "data" / "outputs"
DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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

    if "median_household_income" in gdf.columns:
        gdf.loc[gdf["median_household_income"] < 0, "median_household_income"] = np.nan
    for col in ["pct_uninsured", "pct_no_vehicle"]:
        if col in gdf.columns:
            gdf.loc[~gdf[col].between(0.0, 100.0, inclusive="both"), col] = np.nan

    if "svi_overall" in gdf.columns:
        gdf["svi_overall"] = mask_svi_percentile(gdf["svi_overall"])

    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.005, preserve_topology=True)

    fac = gpd.read_file(facilities_path).to_crs(4326)
    fac = fac[fac.geometry.notna() & ~fac.geometry.is_empty].copy()
    return gdf, fac


tracts_gdf, facilities_gdf = _load_data()
geojson_data = json.loads(tracts_gdf.to_json())

# Typology table for sidebar when the cluster map is not shown (dominant cluster)
CLUSTER_CHOROPLETH_MAX_LARGEST_FRAC = 0.90
try:
    CLUSTER_PROFILES: pd.DataFrame | None = (
        characterize_clusters(tracts_gdf, label_col="cluster")
        if "cluster" in tracts_gdf.columns
        else None
    )
except ValueError:
    CLUSTER_PROFILES = None


def _cluster_choropleth_worthwhile() -> bool:
    """False when a cluster choropleth would look like a single class (e.g. ≥90% in one)."""
    if "cluster" not in tracts_gdf.columns:
        return False
    s = tracts_gdf["cluster"].dropna()
    if len(s) < 2:
        return False
    vc = s.value_counts()
    if len(vc) < 2:
        return False
    largest = float(vc.max() / len(s))
    return largest < CLUSTER_CHOROPLETH_MAX_LARGEST_FRAC


def _cluster_profile_table_block():
    if CLUSTER_PROFILES is None or CLUSTER_PROFILES.empty:
        return html.Div("Cluster profile table is not available for this file.", style={"fontSize": "0.8rem"})
    df = CLUSTER_PROFILES
    ths = [html.Th(" ".join(c.replace("_", " ").split()[:4])) for c in df.columns]
    rows = []
    for _, r in df.iterrows():
        tds = []
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                tds.append(html.Td("—"))
            elif c == "cluster":
                tds.append(html.Td(str(int(v))))
            elif c in ("n_tracts", "total_population"):
                tds.append(html.Td(f"{int(v):,}"))
            else:
                tds.append(html.Td(f"{float(v):.4g}"))
        rows.append(html.Tr(tds))
    return html.Div([
        html.Div(
            "Cluster typology profiles (means; map hidden when one group dominates the state)",
            style={"fontWeight": "700", "fontSize": "0.82rem", "marginBottom": "0.35rem", "lineHeight": "1.3"},
        ),
        html.Div(
            html.Table(
                [html.Thead(html.Tr(ths)), html.Tbody(rows)],
                style={"borderCollapse": "collapse", "fontSize": "0.68rem", "width": "100%"},
            ),
            style={"overflowX": "auto", "maxWidth": "100%"},
        ),
    ])


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

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

FRIENDLY = {
    "accessibility_score": "Accessibility Score",
    "nearest_facility_min": "Drive Time (min)",
    "provider_pop_ratio": "Providers per 1K Pop",
    "cluster": "Cluster",
    "median_household_income": "Median Income ($)",
    "pct_uninsured": "% Uninsured",
    "pct_no_vehicle": "% No Vehicle",
    "svi_overall": "SVI Score",
    "pop_density_sq_mi": "Pop Density (/sq mi)",
    "none": "None",
}

METRIC_COLORSCALES = {
    "accessibility_score": "Greens",
    "nearest_facility_min": "OrRd",
    "provider_pop_ratio": "Blues",
}

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div([
    html.Div([
        html.H1("Healthcare Access Disparities — Pennsylvania",
                style={"margin": "0", "fontSize": "1.6rem"}),
        html.P(
            "Interactive explorer for E2SFCA spatial accessibility scores, "
            "demographic overlays, and community typology clusters.",
            style={"margin": "0.25rem 0 0 0", "color": "#666", "fontSize": "0.9rem"},
        ),
    ], style={"textAlign": "center", "padding": "0.75rem 0"}),

    html.Hr(style={"margin": "0 0 0.75rem 0"}),

    html.Div([
        # Sidebar
        html.Div([
            html.H3("Controls", style={"marginTop": "0"}),

            html.Label("Accessibility Metric", style={"fontWeight": "600", "fontSize": "0.85rem"}),
            dcc.Dropdown(id="metric-dropdown", options=METRIC_OPTIONS,
                         value="accessibility_score", clearable=False,
                         style={"marginBottom": "0.75rem"}),

            html.Label("Demographic Overlay", style={"fontWeight": "600", "fontSize": "0.85rem"}),
            dcc.Dropdown(id="overlay-dropdown", options=OVERLAY_OPTIONS,
                         value="none", clearable=False,
                         style={"marginBottom": "0.75rem"}),

            dcc.Checklist(
                id="show-facilities",
                options=[{"label": " Show facility locations", "value": "show"}],
                value=["show"],
                style={"marginBottom": "0.75rem"},
            ),

            html.Hr(),

            html.Div(id="summary-stats", style={"fontSize": "0.82rem", "lineHeight": "1.5"}),

            html.Div(id="scatter-container", style={"marginTop": "0.75rem"}),
        ], style={
            "width": "280px", "minWidth": "280px", "padding": "1rem",
            "backgroundColor": "#f7f8fa", "borderRadius": "8px",
            "overflowY": "auto", "maxHeight": "88vh",
        }),

        # Map
        html.Div([
            dcc.Loading(
                dcc.Graph(id="main-map", style={"height": "85vh"},
                          config={"scrollZoom": True}),
                type="circle",
            ),
        ], style={"flex": "1", "minWidth": "0"}),
    ], style={
        "display": "flex", "gap": "0.75rem", "alignItems": "flex-start",
    }),
], style={"fontFamily": "'Segoe UI', Arial, sans-serif", "margin": "0.75rem 1rem"})


# ---------------------------------------------------------------------------
# Hover text (precomputed)
# ---------------------------------------------------------------------------

def _build_hover(gdf, overlay_col=None):
    texts = []
    for _, r in gdf.iterrows():
        p = []
        if "geoid" in gdf.columns:
            p.append(f"<b>Tract {r['geoid']}</b>")
        if "accessibility_score" in gdf.columns:
            p.append(f"Access Score: {r['accessibility_score']:.6f}")
        if "nearest_facility_min" in gdf.columns:
            v = r["nearest_facility_min"]
            p.append(f"Nearest Facility: {v:.1f} min" if pd.notna(v) else "Nearest Facility: N/A")
        if "facilities_in_catchment" in gdf.columns and pd.notna(r.get("facilities_in_catchment")):
            p.append(
                "Facilities in Catchment: "
                f"{int(r['facilities_in_catchment'])} "
                "(within drive-time radius; may be outside tract)"
            )
        if "total_population" in gdf.columns and pd.notna(r.get("total_population")):
            p.append(f"Population: {int(r['total_population']):,}")
        if overlay_col and overlay_col in gdf.columns:
            ov = r.get(overlay_col)
            label = FRIENDLY.get(overlay_col, overlay_col)
            p.append(f"<b>{label}: {ov:.2f}</b>" if pd.notna(ov) else f"{label}: N/A")
        texts.append("<br>".join(p))
    return texts


_hover_cache: dict[str, list[str]] = {}


def _get_hover(overlay_col):
    key = overlay_col or "none"
    if key not in _hover_cache:
        _hover_cache[key] = _build_hover(tracts_gdf, overlay_col if overlay_col != "none" else None)
    return _hover_cache[key]


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------

@app.callback(
    Output("main-map", "figure"),
    Output("summary-stats", "children"),
    Output("scatter-container", "children"),
    Input("metric-dropdown", "value"),
    Input("overlay-dropdown", "value"),
    Input("show-facilities", "value"),
)
def update_map(metric: str, overlay: str, show_fac: list[str] | None):
    is_categorical = metric == "cluster"
    show_map_clusters = is_categorical and _cluster_choropleth_worthwhile()
    show_fac = show_fac or []
    hover = _get_hover(overlay)

    fig = go.Figure()

    # -- Main choropleth ------------------------------------------------
    loc_ids = tracts_gdf["geoid"].astype(str) if "geoid" in tracts_gdf.columns else tracts_gdf.index.astype(str)

    if is_categorical and show_map_clusters:
        cluster_vals = tracts_gdf["cluster"].astype(int)
        unique_k = sorted(cluster_vals.unique())
        palette = px.colors.qualitative.Set2
        n = len(unique_k)
        cscale = [[i / max(1, n - 1), palette[i % len(palette)]] for i in range(n)]

        fig.add_trace(go.Choroplethmap(
            geojson=geojson_data,
            featureidkey="properties.geoid" if "geoid" in tracts_gdf.columns else "id",
            locations=loc_ids,
            z=cluster_vals.astype(float),
            colorscale=cscale,
            marker_opacity=0.75,
            marker_line_width=0.3,
            marker_line_color="#ccc",
            text=hover, hoverinfo="text",
            colorbar=dict(
                title=dict(text="Cluster"), len=0.5, y=0.5,
                tickvals=list(range(n)), ticktext=[str(c) for c in unique_k],
            ),
        ))
    elif is_categorical and not show_map_clusters:
        fig.add_annotation(
            text=(
                "Cluster map hidden: the largest group holds a large share of tracts, "
                "so a choropleth would look like a single color. See the <b>typology table</b> "
                "below, or re-run the clustering step after refreshing tract features."
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.55,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=12, family="Segoe UI, Arial, sans-serif"),
        )
    elif not is_categorical:
        vals = tracts_gdf[metric].copy()
        cscale = METRIC_COLORSCALES.get(metric, "Greens")

        fig.add_trace(go.Choroplethmap(
            geojson=geojson_data,
            featureidkey="properties.geoid" if "geoid" in tracts_gdf.columns else "id",
            locations=loc_ids,
            z=vals,
            colorscale=cscale,
            marker_opacity=0.75,
            marker_line_width=0.3,
            marker_line_color="#ccc",
            text=hover, hoverinfo="text",
            colorbar=dict(
                title=dict(text=FRIENDLY.get(metric, metric)),
                len=0.5, y=0.5,
            ),
            zmin=float(vals.quantile(0.02)) if vals.notna().any() else 0,
            zmax=float(vals.quantile(0.98)) if vals.notna().any() else 1,
        ))

    # -- Facility markers -----------------------------------------------
    if "show" in show_fac and not facilities_gdf.empty:
        fac_names = (
            facilities_gdf["facility_name"].tolist()
            if "facility_name" in facilities_gdf.columns
            else [""] * len(facilities_gdf)
        )
        fig.add_trace(go.Scattermap(
            lat=facilities_gdf.geometry.y.tolist(),
            lon=facilities_gdf.geometry.x.tolist(),
            mode="markers",
            marker=dict(size=5, color="#e31a1c", opacity=0.7),
            text=fac_names,
            name="Facilities",
            hoverinfo="text",
        ))

    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=40.9, lon=-77.8),
            zoom=6.3,
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        showlegend=False,
        dragmode="pan",
    )

    # -- Sidebar stats & cluster profile / scatter -----------------------
    stats = _build_stats(metric, overlay, is_categorical, show_map_clusters)
    if is_categorical and not show_map_clusters:
        scatter = _cluster_profile_table_block()
    else:
        scatter = _build_scatter(metric, overlay, is_categorical)

    return fig, stats, scatter


def _build_stats(metric, overlay, is_categorical, show_cluster_map: bool = True):
    parts = []

    if not is_categorical and metric in tracts_gdf.columns:
        d = tracts_gdf[metric].dropna()
        parts.extend([
            html.Div(FRIENDLY.get(metric, metric),
                     style={"fontWeight": "700", "marginBottom": "0.25rem"}),
            _stat_line("Mean", f"{d.mean():.4f}"),
            _stat_line("Median", f"{d.median():.4f}"),
            _stat_line("Min", f"{d.min():.4f}"),
            _stat_line("Max", f"{d.max():.4f}"),
            _stat_line("Std Dev", f"{d.std():.4f}"),
        ])
        if metric == "accessibility_score":
            parts.append(_stat_line("Zero-score tracts", f"{int((d == 0).sum())}"))
        parts.append(html.Br())

    elif is_categorical and "cluster" in tracts_gdf.columns:
        vc = tracts_gdf["cluster"].value_counts().sort_index()
        parts.append(html.Div("Cluster Breakdown",
                              style={"fontWeight": "700", "marginBottom": "0.25rem"}))
        for cid, cnt in vc.items():
            parts.append(_stat_line(f"Cluster {cid}", f"{cnt} tracts"))
        if not show_cluster_map:
            parts.append(html.P(
                "Statewide map is off while one cluster is dominant. Use the profile table in the next panel or re-run k-means with the updated selection rule.",
                style={"color": "#666", "fontSize": "0.78rem", "margin": "0.4rem 0 0 0", "lineHeight": "1.35"},
            ))
        parts.append(html.Br())

    if overlay != "none" and overlay in tracts_gdf.columns:
        od = tracts_gdf[overlay].dropna()
        parts.extend([
            html.Div(f"Overlay: {FRIENDLY.get(overlay, overlay)}",
                     style={"fontWeight": "700", "marginBottom": "0.25rem"}),
            _stat_line("Mean", f"{od.mean():.2f}"),
            _stat_line("Median", f"{od.median():.2f}"),
            _stat_line("Min", f"{od.min():.2f}"),
            _stat_line("Max", f"{od.max():.2f}"),
            html.Br(),
        ])

    parts.extend([
        _stat_line("Total tracts", f"{len(tracts_gdf):,}"),
        _stat_line("Geocoded facilities", f"{len(facilities_gdf):,}"),
    ])
    return html.Div(parts)


def _stat_line(label, value):
    return html.Div([
        html.Span(f"{label}: ", style={"color": "#555"}),
        html.Span(value, style={"fontWeight": "600"}),
    ])


def _build_scatter(metric, overlay, is_categorical):
    if overlay == "none" or is_categorical:
        return html.Div()
    if metric not in tracts_gdf.columns or overlay not in tracts_gdf.columns:
        return html.Div()

    df = tracts_gdf[[metric, overlay]].dropna()
    if len(df) < 10:
        return html.Div()

    corr = df[metric].corr(df[overlay])
    fig = px.scatter(
        df, x=overlay, y=metric, opacity=0.3,
        labels={overlay: FRIENDLY.get(overlay, overlay),
                metric: FRIENDLY.get(metric, metric)},
        trendline="ols",
    )
    fig.update_traces(marker=dict(size=3, color="#4575b4"))
    fig.update_layout(
        height=220,
        margin=dict(l=40, r=10, t=30, b=40),
        title=dict(text=f"r = {corr:.3f}", x=0.5, font=dict(size=12)),
        template="plotly_white",
        font=dict(size=10),
    )
    return html.Div([
        html.Div("Metric vs Overlay",
                 style={"fontWeight": "700", "fontSize": "0.82rem", "marginBottom": "0.25rem"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"height": "220px"}),
    ])


if __name__ == "__main__":
    app.run(debug=True)
