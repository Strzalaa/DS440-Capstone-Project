"""Statistical modelling of demographic disparities in healthcare access.

Includes OLS regression, geographically weighted regression (GWR),
and spatial autocorrelation measures (Moran's I, LISA).
"""

from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
