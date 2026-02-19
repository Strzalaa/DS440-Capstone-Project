"""Unsupervised clustering to identify community typologies by healthcare access profile.

Implements feature engineering, k-means with optimal-k selection,
hierarchical clustering, and spatial clustering (LISA-based).
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
