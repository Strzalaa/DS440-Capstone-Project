# Project Progress Tracker

**DS 440 Capstone — Mapping Healthcare Access Disparities (Group 24)**

Last updated: 2026-03-05

---

## Pipeline Overview

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Collection (ACS, SVI, TIGER, CMS, HRSA) | Complete |
| 2 | Data Cleaning & Geocoding | Complete |
| 3 | E2SFCA Spatial Accessibility Analysis | Complete |
| 4 | Statistical Analysis (OLS, GWR, Moran's I, LISA) | Complete |
| 5 | Clustering (K-means, Hierarchical, Spatial) | Complete |
| 6 | Visualization & Dashboard | Complete |
| 7 | Testing | Complete |

---

## Detailed Task Checklist

### Phase 1 — Data Collection

- [x] Census ACS demographics downloaded (`data/raw/acs/acs_2022_tract_42.csv`)
- [x] CDC SVI downloaded (`data/raw/svi/SVI_2022_Pennsylvania.csv`)
- [x] TIGER/Line tracts shapefile (`data/raw/tiger/tl_2024_42_tract.gpkg`)
- [x] TIGER/Line roads — all 67 PA counties (`data/raw/tiger/tl_2024_42_roads.gpkg`, 325 MB, 596K segments)
- [x] Hospital facility data via NPI Registry API (639 records)
- [x] Health center / FQHC data via NPI Registry API (923 records)
- [x] `src/data_collection.py` — all functions implemented (including NPI-based fetch)
- [x] Notebook `01_data_collection.ipynb` — executed with saved outputs

### Phase 2 — Data Cleaning & Geocoding

- [x] `src/geocoding.py` — all functions implemented (batch + single-address geocoding)
- [x] Notebook `02_data_cleaning.ipynb` — executed with saved outputs
- [x] Enriched tracts GeoPackage (`data/processed/pa_tracts_enriched.gpkg`, 3,446 tracts, 36 columns)
- [x] Populated facilities GeoPackage (`data/processed/pa_facilities.gpkg` — 1,514 facilities, 1,341 geocoded)
- [x] Missing data report generated

### Phase 3 — E2SFCA Analysis

- [x] `src/spatial_analysis.py` — optimized with Euclidean + circuity approach
- [x] Notebook `03_e2sfca_analysis.ipynb` — executed with saved outputs
- [x] `data/outputs/pa_accessibility_scores.gpkg` — 3,446 tracts with accessibility scores
- [x] `data/outputs/pa_drive_times.csv` — 281,142 origin-destination pairs
- [x] `data/outputs/pa_e2sfca_sensitivity.csv` — 31,014 sensitivity analysis rows
- [x] `data/outputs/pa_accessibility_map.html` — interactive map

### Phase 4 — Statistical Analysis

- [x] `src/statistics.py` — all functions implemented
- [x] Notebook `04_statistical_analysis.ipynb` — executed with saved outputs
- [x] OLS regression, Moran's I, GWR, LISA all computed

### Phase 5 — Clustering

- [x] `src/clustering.py` — all functions implemented
- [x] Notebook `05_clustering.ipynb` — executed with saved outputs
- [x] `data/outputs/pa_accessibility_clusters.gpkg` generated
- [x] `data/outputs/pa_cluster_profiles.csv` generated
- [x] `data/outputs/pa_cluster_map.html` generated

### Phase 6 — Visualization & Dashboard

- [x] `src/visualization.py` — all functions implemented
- [x] Notebook `06_visualization.ipynb` — executed with saved outputs
- [x] `dashboard/app.py` — wired with Dash callbacks for interactive choropleth map
- [x] HTML map exports in `data/outputs/`

### Phase 7 — Testing

- [x] `tests/test_spatial_analysis.py` — 9 tests (decay functions)
- [x] `tests/test_geocoding.py` — 8 tests (geocoding, validation, cross-reference)
- [x] `tests/test_clustering.py` — 14 tests (prepare, k-means, hierarchical, characterize)
- [x] `tests/test_data_collection.py` — 9 tests (pick_column, normalise_facility_columns)
- [x] **42 total tests — all passing**

---

## Key Design Decisions

1. **NPI Registry** used as facility data source (CMS POS and HRSA APIs were unreliable/deprecated).
2. **Euclidean + circuity factor** approach for drive-time estimation instead of full road network routing (596K segments were computationally infeasible).
3. **Census Bureau batch geocoder** for efficient facility geocoding (89% success rate).
4. **GWR** (`mgwr`) wrapped in try/except for environments where it's unavailable.

---

## Environment

- Python 3.12.9
- Virtual environment: `.venv/`
- Jupyter kernel: `ds440-venv`
- All packages from `requirements.txt` installed
- 42 unit tests passing
