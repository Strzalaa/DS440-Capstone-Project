"""Unsupervised clustering to identify community typologies by healthcare access profile.

Implements feature engineering, k-means with optimal-k selection,
hierarchical clustering, and spatial clustering (LISA-based).
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.statistics import lisa_analysis


def prepare_features(
    data: gpd.GeoDataFrame,
    feature_cols: Optional[list[str]] = None,
    scale: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Standardise and prepare the feature matrix for clustering.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Tract-level data containing accessibility and demographic columns.
    feature_cols : list[str] | None
        Columns to include.  Defaults to a curated set if *None*.
    scale : bool
        If True, apply z-score standardisation.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (feature_matrix, column_names).
    """
    default_cols = [
        "accessibility_score",
        "nearest_facility_min",
        "provider_pop_ratio",
        "median_household_income",
        "pct_uninsured",
        "pct_no_vehicle",
        "svi_overall",
    ]
    cols = feature_cols or [c for c in default_cols if c in data.columns]
    if not cols:
        raise ValueError("No usable feature columns found for clustering.")

    matrix = data[cols].replace([np.inf, -np.inf], np.nan).fillna(data[cols].median(numeric_only=True))
    X = matrix.to_numpy(dtype=float)
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, cols


def kmeans_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> dict:
    """Run k-means for a range of *k* and select the optimal number of clusters.

    Uses the elbow method (inertia) and silhouette scores.

    Parameters
    ----------
    X : np.ndarray
        Standardised feature matrix.
    k_range : range
        Candidate cluster counts.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Keys: ``"optimal_k"`` (int), ``"labels"`` (np.ndarray for best k),
        ``"inertias"`` (list[float]), ``"silhouette_scores"`` (list[float]),
        ``"models"`` (dict mapping k -> fitted KMeans).
    """
    if X.shape[0] < 3:
        raise ValueError("At least 3 rows are required for k-means evaluation.")

    inertias: list[float] = []
    silhouettes: list[float] = []
    models: dict[int, KMeans] = {}

    valid_ks = [k for k in k_range if k >= 2 and k < X.shape[0]]
    if not valid_ks:
        raise ValueError("No valid k values in k_range for this dataset size.")

    for k in valid_ks:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X)
        models[k] = model
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(X, labels)))

    best_idx = int(np.argmax(silhouettes))
    optimal_k = valid_ks[best_idx]
    best_model = models[optimal_k]
    return {
        "optimal_k": int(optimal_k),
        "labels": best_model.labels_,
        "inertias": inertias,
        "silhouette_scores": silhouettes,
        "models": models,
    }


def hierarchical_clustering(
    X: np.ndarray,
    n_clusters: int = 4,
    method: str = "ward",
) -> dict:
    """Agglomerative hierarchical clustering with dendrogram data.

    Parameters
    ----------
    X : np.ndarray
        Standardised feature matrix.
    n_clusters : int
        Number of clusters to produce.
    method : str
        Linkage method (``"ward"``, ``"complete"``, ``"average"``).

    Returns
    -------
    dict
        Keys: ``"labels"`` (np.ndarray), ``"linkage_matrix"`` (np.ndarray),
        ``"model"`` (fitted AgglomerativeClustering).
    """
    if X.shape[0] < n_clusters:
        raise ValueError("n_clusters cannot exceed number of samples.")

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = model.fit_predict(X)
    linkage_matrix = linkage(X, method=method)
    return {
        "labels": labels,
        "linkage_matrix": linkage_matrix,
        "model": model,
    }


def spatial_clustering(
    data: gpd.GeoDataFrame,
    variable: str = "accessibility_score",
    significance: float = 0.05,
) -> gpd.GeoDataFrame:
    """LISA-based spatial clustering identifying High-High and Low-Low groups.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Tract-level data with the target variable.
    variable : str
        Column to cluster on.
    significance : float
        p-value cutoff for labelling significant clusters.

    Returns
    -------
    gpd.GeoDataFrame
        Input data with ``spatial_cluster`` column.
    """
    out = lisa_analysis(data, variable=variable, significance=significance)
    mapping = {
        "High-High": "High Access Cluster",
        "Low-Low": "Low Access Cluster",
        "High-Low": "Spatial Outlier (High)",
        "Low-High": "Spatial Outlier (Low)",
    }
    out["spatial_cluster"] = out["lisa_cluster"].map(mapping).fillna("Not Significant")
    return out


def characterize_clusters(
    data: gpd.GeoDataFrame,
    label_col: str = "cluster",
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Generate summary profiles for each cluster.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Tract-level data with cluster labels.
    label_col : str
        Column containing cluster assignments.
    feature_cols : list[str] | None
        Columns to summarise per cluster.

    Returns
    -------
    pd.DataFrame
        Mean, median, count, and total population per cluster.
    """
    if label_col not in data.columns:
        raise ValueError(f"Cluster label column '{label_col}' not found.")

    default_features = [
        "accessibility_score",
        "nearest_facility_min",
        "provider_pop_ratio",
        "median_household_income",
        "pct_uninsured",
        "pct_no_vehicle",
        "svi_overall",
    ]
    cols = feature_cols or [c for c in default_features if c in data.columns]
    if not cols:
        raise ValueError("No feature columns available to characterize clusters.")

    grouped = data.groupby(label_col, dropna=False)
    summary = grouped[cols].agg(["mean", "median"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary["n_tracts"] = grouped.size()
    if "total_population" in data.columns:
        summary["total_population"] = grouped["total_population"].sum()
    return summary.reset_index()
