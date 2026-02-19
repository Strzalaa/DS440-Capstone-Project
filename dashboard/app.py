"""Plotly Dash dashboard for exploring healthcare access disparities.

Run with:
    python dashboard/app.py
"""

from __future__ import annotations

from dash import Dash, dcc, html
import plotly.express as px

app = Dash(__name__, title="Healthcare Access Dashboard")

app.layout = html.Div(
    [
        html.H1("Healthcare Access Disparities — Pennsylvania"),
        html.P(
            "Interactive explorer for spatial accessibility scores, "
            "demographic overlays, and community typology clusters."
        ),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Filters"),
                        html.Label("Accessibility metric:"),
                        dcc.Dropdown(
                            id="metric-dropdown",
                            options=[
                                {"label": "Accessibility Score", "value": "accessibility_score"},
                                {"label": "Drive Time to Nearest Facility", "value": "nearest_facility_min"},
                                {"label": "Provider-to-Population Ratio", "value": "provider_pop_ratio"},
                            ],
                            value="accessibility_score",
                        ),
                        html.Br(),
                        html.Label("Demographic overlay:"),
                        dcc.Dropdown(
                            id="overlay-dropdown",
                            options=[
                                {"label": "None", "value": "none"},
                                {"label": "Median Income", "value": "median_household_income"},
                                {"label": "% Uninsured", "value": "pct_uninsured"},
                                {"label": "% Without Vehicle", "value": "pct_no_vehicle"},
                                {"label": "SVI Score", "value": "svi_overall"},
                            ],
                            value="none",
                        ),
                    ],
                    style={"width": "25%", "display": "inline-block", "verticalAlign": "top", "padding": "1rem"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="main-map", style={"height": "70vh"}),
                    ],
                    style={"width": "73%", "display": "inline-block", "verticalAlign": "top"},
                ),
            ]
        ),
    ],
    style={"fontFamily": "Arial, sans-serif", "margin": "1rem"},
)


if __name__ == "__main__":
    app.run(debug=True)
