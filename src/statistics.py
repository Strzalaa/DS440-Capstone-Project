"""Statistical modelling of demographic disparities in healthcare access.

Includes OLS regression, geographically weighted regression (GWR),
and spatial autocorrelation measures (Moran's I, LISA).
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda import Moran, Moran_Local
from libpysal.weights import Queen


def run_ols_regression(
    data: gpd.GeoDataFrame,
    dependent_var: str = "accessibility_score",
    independent_vars: Optional[list[str]] = None,
) -> dict:
    """Fit an OLS regression of accessibility on demographic predictors.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Tract-level data with accessibility scores and demographics.
    dependent_var : str
        Column name for the response variable.
    independent_vars : list[str] | None
        Predictor column names.  Defaults to a standard set of
        demographic variables if *None*.

    Returns
    -------
    dict
        Keys: ``"model"`` (fitted statsmodels object), ``"summary"`` (str),
        ``"coefficients"`` (pd.DataFrame), ``"r_squared"`` (float),
        ``"aic"`` (float).
    """
    default_vars = [
        "median_household_income",
        "pct_uninsured",
        "pct_no_vehicle",
        "svi_overall",
        "population_density_sq_mi",
    ]
    predictors = independent_vars or [col for col in default_vars if col in data.columns]
    if not predictors:
        raise ValueError("No independent variables available for OLS.")

    cols = [dependent_var, *predictors]
    clean = data[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("No valid rows remain after dropping missing values.")

    X = sm.add_constant(clean[predictors])
    y = clean[dependent_var]
    model = sm.OLS(y, X).fit()
    coef_df = (
        pd.DataFrame(
            {
                "term": model.params.index,
                "coefficient": model.params.values,
                "p_value": model.pvalues.values,
                "std_error": model.bse.values,
            }
        )
        .sort_values("p_value")
        .reset_index(drop=True)
    )
    return {
        "model": model,
        "summary": model.summary().as_text(),
        "coefficients": coef_df,
        "r_squared": float(model.rsquared),
        "aic": float(model.aic),
    }


def run_gwr(
    data: gpd.GeoDataFrame,
    dependent_var: str = "accessibility_score",
    independent_vars: Optional[list[str]] = None,
    kernel: str = "adaptive",
) -> dict:
    """Fit a Geographically Weighted Regression model.

    Allows regression coefficients to vary across space, revealing
    where specific demographic factors most strongly predict poor access.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Tract-level data (must have point geometry, e.g. centroids).
    dependent_var : str
        Response variable column.
    independent_vars : list[str] | None
        Predictor columns.
    kernel : str
        GWR kernel type (``"adaptive"`` or ``"fixed"``).

    Returns
    -------
    dict
        Keys: ``"model"`` (fitted mgwr object), ``"local_r_squared"`` (array),
        ``"local_coefficients"`` (pd.DataFrame), ``"aic"`` (float),
        ``"bandwidth"`` (float).
    """
    try:
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW
    except ImportError as exc:
        raise ImportError("mgwr is required for run_gwr(). Install from requirements.txt.") from exc

    default_vars = [
        "median_household_income",
        "pct_uninsured",
        "pct_no_vehicle",
        "svi_overall",
        "population_density_sq_mi",
    ]
    predictors = independent_vars or [col for col in default_vars if col in data.columns]
    if not predictors:
        raise ValueError("No independent variables available for GWR.")

    work = data.copy()
    if not all(work.geometry.geom_type == "Point"):
        work["geometry"] = work.geometry.centroid
    work = work.to_crs(3857)
    cols = [dependent_var, *predictors, "geometry"]
    work = work[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise ValueError("No valid rows remain for GWR.")

    coords = np.column_stack([work.geometry.x.values, work.geometry.y.values])
    y = work[dependent_var].to_numpy(dtype=float).reshape((-1, 1))
    X = work[predictors].to_numpy(dtype=float)

    fixed = kernel == "fixed"
    bw_selector = Sel_BW(coords, y, X, fixed=fixed)
    bw = bw_selector.search()

    gwr_model = GWR(coords, y, X, bw, fixed=fixed).fit()
    coef_cols = ["const", *predictors]
    coef_df = pd.DataFrame(gwr_model.params, columns=coef_cols, index=work.index)
    return {
        "model": gwr_model,
        "local_r_squared": gwr_model.localR2,
        "local_coefficients": coef_df,
        "aic": float(gwr_model.aic),
        "bandwidth": float(bw),
    }


def compute_morans_i(
    data: gpd.GeoDataFrame,
    variable: str = "accessibility_score",
) -> dict:
    """Compute Global Moran's I for spatial autocorrelation.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Tract-level data with the target variable.
    variable : str
        Column to test for spatial clustering.

    Returns
    -------
    dict
        Keys: ``"I"`` (float), ``"p_value"`` (float),
        ``"z_score"`` (float), ``"expected_I"`` (float).
    """
    clean = data[[variable, "geometry"]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("No valid rows available for Moran's I.")

    w = Queen.from_dataframe(clean)
    w.transform = "r"
    moran = Moran(clean[variable].to_numpy(dtype=float), w)
    return {
        "I": float(moran.I),
        "p_value": float(moran.p_sim),
        "z_score": float(moran.z_sim),
        "expected_I": float(moran.EI),
    }


def lisa_analysis(
    data: gpd.GeoDataFrame,
    variable: str = "accessibility_score",
    significance: float = 0.05,
) -> gpd.GeoDataFrame:
    """Local Indicators of Spatial Association (LISA) analysis.

    Identifies statistically significant local clusters and outliers:
    High-High, Low-Low, High-Low, Low-High.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Tract-level data.
    variable : str
        Column to analyse.
    significance : float
        p-value threshold for significance.

    Returns
    -------
    gpd.GeoDataFrame
        Input data augmented with ``lisa_cluster`` (category),
        ``lisa_p_value``, and ``lisa_z_score`` columns.
    """
    out = data.copy()
    clean = out[[variable, "geometry"]].replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        out["lisa_cluster"] = "Not Significant"
        out["lisa_p_value"] = np.nan
        out["lisa_z_score"] = np.nan
        return out

    w = Queen.from_dataframe(clean)
    w.transform = "r"
    values = clean[variable].to_numpy(dtype=float)
    m_local = Moran_Local(values, w)

    cluster_map = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    sig = m_local.p_sim < significance
    labels = np.where(sig, [cluster_map.get(int(q), "Not Significant") for q in m_local.q], "Not Significant")

    out["lisa_cluster"] = "Not Significant"
    out["lisa_p_value"] = np.nan
    out["lisa_z_score"] = np.nan
    out.loc[clean.index, "lisa_cluster"] = labels
    out.loc[clean.index, "lisa_p_value"] = m_local.p_sim
    out.loc[clean.index, "lisa_z_score"] = m_local.z_sim
    return out
