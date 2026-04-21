"""Tests for the clustering module."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from src.clustering import (
    prepare_features,
    kmeans_optimal_k,
    hierarchical_clustering,
    characterize_clusters,
)


def _make_test_gdf(n: int = 50) -> gpd.GeoDataFrame:
    """Generate a small synthetic tract-like GeoDataFrame for testing."""
    rng = np.random.default_rng(42)
    return gpd.GeoDataFrame(
        {
            "accessibility_score": rng.uniform(0, 0.01, n),
            "nearest_facility_min": rng.uniform(1, 30, n),
            "provider_pop_ratio": rng.uniform(0, 5, n),
            "median_household_income": rng.uniform(20000, 120000, n),
            "pct_no_vehicle": rng.uniform(0, 30, n),
            "svi_overall": rng.uniform(0, 1, n),
        },
        geometry=[Point(-76 + rng.uniform(-1, 1), 40 + rng.uniform(-1, 1)) for _ in range(n)],
        crs="EPSG:4326",
    )


class TestPrepareFeatures:
    def test_returns_numpy_array(self):
        gdf = _make_test_gdf()
        X, cols = prepare_features(gdf)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == len(gdf)

    def test_column_selection(self):
        gdf = _make_test_gdf()
        X, cols = prepare_features(gdf, feature_cols=["accessibility_score", "svi_overall"])
        assert len(cols) == 2
        assert X.shape[1] == 2

    def test_scaling_normalizes(self):
        gdf = _make_test_gdf()
        X, _ = prepare_features(gdf, scale=True)
        assert abs(X.mean(axis=0)).max() < 0.1

    def test_no_nans_in_output(self):
        gdf = _make_test_gdf()
        gdf.loc[0, "accessibility_score"] = np.nan
        X, _ = prepare_features(gdf)
        assert not np.isnan(X).any()

    def test_svi_sentinel_treated_as_missing(self):
        gdf = _make_test_gdf(60)
        gdf.loc[:10, "svi_overall"] = -999.0
        X, cols = prepare_features(gdf)
        svi_i = cols.index("svi_overall")
        assert np.isfinite(X[:, svi_i]).all()
        assert np.abs(X[:, svi_i]).max() < 5.0

    def test_extreme_outlier_is_winsorized_before_scaling(self):
        gdf = _make_test_gdf(100)
        gdf["provider_pop_ratio"] = 1.0
        gdf.loc[99, "provider_pop_ratio"] = 10_000.0
        X, cols = prepare_features(gdf, scale=False)
        ratio_i = cols.index("provider_pop_ratio")
        assert X[:, ratio_i].max() < 10_000.0

    def test_raises_on_no_columns(self):
        gdf = gpd.GeoDataFrame({"x": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
        with pytest.raises(ValueError):
            prepare_features(gdf)


class TestKmeansOptimalK:
    def test_returns_valid_k(self):
        gdf = _make_test_gdf(100)
        X, _ = prepare_features(gdf)
        result = kmeans_optimal_k(X)
        # Default min_k=3: typology uses at least 3 groups when n is large
        assert 3 <= result["optimal_k"] <= 10
        assert len(result["max_largest_cluster_fracs"]) == len(result["valid_ks"])
        assert "balance_preference_met" in result

    def test_silhouette_scores_in_range(self):
        gdf = _make_test_gdf(100)
        X, _ = prepare_features(gdf)
        result = kmeans_optimal_k(X)
        for s in result["silhouette_scores"]:
            assert -1.0 <= s <= 1.0

    def test_labels_match_sample_count(self):
        gdf = _make_test_gdf(50)
        X, _ = prepare_features(gdf)
        result = kmeans_optimal_k(X)
        assert len(result["labels"]) == 50

    def test_inertias_decreasing(self):
        gdf = _make_test_gdf(100)
        X, _ = prepare_features(gdf)
        result = kmeans_optimal_k(X)
        inertias = result["inertias"]
        assert all(inertias[i] >= inertias[i + 1] for i in range(len(inertias) - 1))

    def test_fixed_k_respected(self):
        gdf = _make_test_gdf(80)
        X, _ = prepare_features(gdf)
        result = kmeans_optimal_k(X, fixed_k=4)
        assert result["optimal_k"] == 4
        assert len(np.unique(result["labels"])) == 4


class TestHierarchicalClustering:
    def test_returns_correct_number_of_clusters(self):
        gdf = _make_test_gdf(50)
        X, _ = prepare_features(gdf)
        result = hierarchical_clustering(X, n_clusters=3)
        assert len(set(result["labels"])) == 3

    def test_labels_match_sample_count(self):
        gdf = _make_test_gdf(50)
        X, _ = prepare_features(gdf)
        result = hierarchical_clustering(X, n_clusters=4)
        assert len(result["labels"]) == 50


class TestCharacterizeClusters:
    def test_one_row_per_cluster(self):
        gdf = _make_test_gdf(50)
        gdf["cluster"] = np.repeat([0, 1, 2, 3, 4], 10)
        result = characterize_clusters(gdf, label_col="cluster")
        assert len(result) == 5

    def test_includes_n_tracts_column(self):
        gdf = _make_test_gdf(30)
        gdf["cluster"] = np.repeat([0, 1, 2], 10)
        result = characterize_clusters(gdf, label_col="cluster")
        assert "n_tracts" in result.columns
        assert result["n_tracts"].sum() == 30

    def test_raises_on_missing_label(self):
        gdf = _make_test_gdf(10)
        with pytest.raises(ValueError):
            characterize_clusters(gdf, label_col="nonexistent")
