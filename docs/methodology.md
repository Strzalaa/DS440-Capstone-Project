# Methodology Report

This report summarizes the Pennsylvania healthcare access disparities pipeline implemented in this repository (DS 440 capstone).

## 1. Data Sources and Integration

- **Facilities:** CMS NPI Registry–based hospital and HRSA-related facility lists, geocoded to points (see `src/geocoding.py`, `src/data_collection.py`).
- **Census tracts:** TIGER/Line shapefiles; tract identifiers joined to ACS 5-year estimates and CDC SVI tract attributes.
- **Demographics:** American Community Survey variables (income, vehicles, insurance where available) merged by `geoid`.
- **SVI:** CDC Social Vulnerability Index tract fields; **RPL percentile fields** (`RPL_THEMES`, theme ranks) use documented sentinels **-999 / -888** for missing or unreliable values. These are masked to missing before analysis (`src/svi.py`, `mask_svi_percentile`), including in notebook `02_data_cleaning.ipynb`, clustering, the dashboard, and OLS/GWR in `src/statistics.py`.

## 2. Enhanced Two-Step Floating Catchment Area (E2SFCA)

Pairwise origin–destination travel times are estimated with a **Euclidean distance + circuity factor** approximation and urbanicity-based catchment thresholds, then fed into Gaussian (or configured) distance decay and the two-step ratio construction (`src/spatial_analysis.py`). Outputs include tract-level `accessibility_score`, `nearest_facility_min`, and `provider_pop_ratio`.

## 3. Statistical Analysis (OLS, GWR, Moran's I, LISA)

- **OLS:** Baseline global regression of `accessibility_score` on selected demographic predictors (`run_ols_regression` in `src/statistics.py`). `svi_overall` is SVI-masked before fitting.
- **GWR:** Local regression with coordinates from tract geometries: polygons are **reprojected to EPSG:3857, then centroid**, so centroids are computed in a projected CRS (`run_gwr`). `svi_overall` is masked before fitting.
- **Moran's I / LISA:** Global and local spatial autocorrelation using queen contiguity weights (`compute_morans_i`, `lisa_analysis`).

## 4. Unsupervised Clustering

K-means (and related methods in `src/clustering.py`) on standardized, winsorized features, with SVI and percentage fields sanitized via `_sanitize_feature_frame` (including `mask_svi_percentile` for `svi_overall`).

## 5. Interactive Visualisation

Plotly Dash app (`dashboard/app.py`) choropleth map of tract metrics, optional demographic overlays, facility points from `data/processed/pa_facilities.gpkg`, and summary statistics. SVI overlay values use the same percentile masking as analysis.

## 6. Validation

Unit tests under `tests/` cover SVI masking, spatial helpers, geocoding utilities, clustering, and OLS/GWR wiring. Re-run `pytest` after environment or data changes.

## 7. Limitations

- Drive times are approximated (not full network routing); results are suitable for comparative access patterns, not clinical navigation.
- Facility lists depend on source taxonomies and geocoding match quality; deduplication of CMS/HRSA pairs changes row counts versus raw NPI pulls.
- Small-area rates (uninsured, no vehicle) can be unstable for very small tracts; the pipeline caps invalid percentages and masks SVI sentinels.
