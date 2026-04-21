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
from src.svi import mask_svi_percentile


def _sanitize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if "median_household_income" in out.columns:
        out.loc[out["median_household_income"] < 0, "median_household_income"] = np.nan
    for c in ["pct_uninsured", "pct_no_vehicle"]:
        if c in out.columns:
            out.loc[~out[c].between(0.0, 100.0, inclusive="both"), c] = np.nan
    if "nearest_facility_min" in out.columns:
        out.loc[out["nearest_facility_min"] < 0, "nearest_facility_min"] = np.nan
    if "provider_pop_ratio" in out.columns:
        out.loc[out["provider_pop_ratio"] < 0, "provider_pop_ratio"] = np.nan
    if "accessibility_score" in out.columns:
        out.loc[out["accessibility_score"] < 0, "accessibility_score"] = np.nan
    if "svi_overall" in out.columns:
        out["svi_overall"] = mask_svi_percentile(out["svi_overall"])
    return out


def _winsorize_feature_frame(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.DataFrame:
    """Clip extreme values so tiny-population outliers do not dominate K-means."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        valid = s.dropna()
        if valid.empty:
            continue
        lo = float(valid.quantile(lower_q))
        hi = float(valid.quantile(upper_q))
        out[c] = s.clip(lower=lo, upper=hi)
    return out


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

    matrix = _sanitize_feature_frame(data[cols].copy())
    matrix = _winsorize_feature_frame(matrix)
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(matrix.median(numeric_only=True)).fillna(0.0)
    X = matrix.to_numpy(dtype=float)
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, cols


def kmeans_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
    min_k: int = 3,
    max_largest_cluster_frac: float = 0.90,
    fixed_k: int | None = None,
) -> dict:
    """Run k-means for a range of *k* and select the optimal number of clusters.

    For each *k*, the largest cluster’s share of tracts is recorded.  By default
    the chosen *k* is the silhouette-best among those with
    *k* ≥ *min_k* and no group holding more than *max_largest_cluster_frac* of
    points—so binary splits that place ~all points in one cluster are avoided
    for typology mapping.  If no *k* passes that filter, the best silhouette
    among *k* ≥ *min_k* is used; if only *k* = 2 is in range, that is used.

    Parameters
    ----------
    X : np.ndarray
        Standardised feature matrix.
    k_range : range
        Candidate cluster counts.
    random_state : int
        Seed for reproducibility.
    min_k : int
        Minimum cluster count to prefer for typology (default 3).
    max_largest_cluster_frac : float
        Reject *k* (when selecting among balanced options) if any cluster has
        more than this fraction of samples.
    fixed_k : int | None
        If set, skip selection and return labels for this *k* only (inertias and
        silhouettes are still computed for the full *k_range* for diagnostics).

    Returns
    -------
    dict
        Keys: ``"optimal_k"`` (int), ``"labels"`` (np.ndarray for best k),
        ``"inertias"`` (list[float]), ``"silhouette_scores"`` (list[float]),
        ``"models"`` (dict mapping k -> fitted KMeans),
        ``"max_largest_cluster_fracs"`` (list[float], same order as *valid_ks*),
        ``"balance_preference_met"`` (bool; True if a balanced *k* was chosen).
    """
    if X.shape[0] < 3:
        raise ValueError("At least 3 rows are required for k-means evaluation.")

    inertias: list[float] = []
    silhouettes: list[float] = []
    max_largest: list[float] = []
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
        counts = np.bincount(labels, minlength=k)
        max_largest.append(float(counts.max() / X.shape[0]))

    n = len(valid_ks)

    def _argmax_sil(idxs: list[int]) -> int:
        if not idxs:
            return 0
        return int(idxs[int(np.argmax([silhouettes[i] for i in idxs]))])

    if fixed_k is not None:
        if fixed_k not in valid_ks:
            raise ValueError(f"fixed_k={fixed_k} is not in valid_ks {valid_ks}.")
        best_local = valid_ks.index(fixed_k)
        optimal_k = fixed_k
        balance_met = min_k <= fixed_k and max_largest[best_local] <= max_largest_cluster_frac
    else:
        balanced = [
            i
            for i in range(n)
            if valid_ks[i] >= min_k and max_largest[i] <= max_largest_cluster_frac
        ]
        if balanced:
            best_local = _argmax_sil(balanced)
            balance_met = True
        else:
            ge_min = [i for i in range(n) if valid_ks[i] >= min_k]
            if ge_min:
                best_local = _argmax_sil(ge_min)
                balance_met = False
            else:
                best_local = int(np.argmax(silhouettes))
                balance_met = False
        optimal_k = valid_ks[best_local]

    best_model = models[optimal_k]
    return {
        "optimal_k": int(optimal_k),
        "labels": best_model.labels_,
        "inertias": inertias,
        "silhouette_scores": silhouettes,
        "models": models,
        "valid_ks": valid_ks,
        "max_largest_cluster_fracs": max_largest,
        "balance_preference_met": bool(balance_met),
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

    work = data.copy()
    work[cols] = _sanitize_feature_frame(work[cols])
    grouped = work.groupby(label_col, dropna=False)
    summary = grouped[cols].agg(["mean", "median"])
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    summary["n_tracts"] = grouped.size()
    if "total_population" in data.columns:
        summary["total_population"] = grouped["total_population"].sum()
    return summary.reset_index()
