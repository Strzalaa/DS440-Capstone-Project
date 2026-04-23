"""Tests for OLS/GWR helpers (SVI masking, GWR geometry order)."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, box

from src.statistics import run_ols_regression, run_gwr


def _minimal_tract_gdf(n: int = 10) -> gpd.GeoDataFrame:
    rng = np.random.default_rng(0)
    # Distinct tract shapes so projected centroids differ (GWR needs varying coordinates).
    geoms = [
        box(-77.0 + i * 0.01, 40.0, -76.99 + i * 0.01, 40.01) for i in range(n)
    ]
    return gpd.GeoDataFrame(
        {
            "accessibility_score": rng.uniform(1e-5, 2e-4, n),
            "median_household_income": rng.uniform(30000, 90000, n),
            "pct_no_vehicle": rng.uniform(0, 25, n),
            "svi_overall": rng.uniform(0.05, 0.95, n),
            "pop_density_sq_mi": rng.uniform(100, 5000, n),
        },
        geometry=geoms,
        crs="EPSG:4326",
    )


def test_ols_drops_rows_with_svi_sentinel() -> None:
    gdf = _minimal_tract_gdf(12)
    gdf.loc[gdf.index[:3], "svi_overall"] = -999.0
    out = run_ols_regression(gdf)
    assert out["model"].nobs == len(gdf) - 3


def test_gwr_runs_after_polygon_centroid_in_projected_crs() -> None:
    pytest.importorskip("mgwr")
    gdf = _minimal_tract_gdf(50)
    # Fixed kernel is more stable for small synthetic samples than adaptive k-NN.
    out = run_gwr(gdf, kernel="fixed")
    assert "aic" in out
    assert out["local_coefficients"].shape[0] == 50
