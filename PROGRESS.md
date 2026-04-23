# Project Progress Tracker

**DS 440 Capstone — Mapping Healthcare Access Disparities (Group 24)**

Last updated: 2026-04-23

---

## Pipeline Overview

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Collection (ACS, SVI, TIGER, HIFLD, HRSA, ASC/RHC, NPI) | Complete |
| 2 | Data Cleaning, Geocoding & N-way Source Merge | Complete |
| 3 | E2SFCA Spatial Accessibility Analysis | Complete |
| 4 | Statistical Analysis (OLS, GWR, Moran's I, LISA) | Complete |
| 5 | Clustering (K-means, Hierarchical, Spatial) | Complete |
| 6 | Visualization & Dashboard | Complete |
| 7 | Testing | Complete |

---

## Detailed Task Checklist

### Phase 1 — Data Collection

- [x] Census ACS demographics (`data/raw/acs/acs_2022_tract_42.csv`)
- [x] CDC SVI (`data/raw/svi/SVI_2022_Pennsylvania.csv`)
- [x] TIGER/Line tracts (`data/raw/tiger/tl_2024_42_tract.gpkg`, 3,446 tracts)
- [x] TIGER/Line roads — all 67 PA counties (`tl_2024_42_roads.gpkg`, 325 MB, 596K segments)
- [x] **Hospitals via HRSA gisportal layer 1** (`data/raw/cms/cms_facilities_standardized.csv`, 252 pre-geocoded records)
- [x] **Critical Access Hospitals via HRSA gisportal layer 2** (folded into CMS file)
- [x] **Rural Health Clinics + ASCs via HRSA gisportal layers 4 & 7** (`data/raw/urgent_care/urgent_care_facilities_standardized.csv`, 310 pre-geocoded records)
- [x] **HRSA health center service delivery sites via gisportal layer 18** (`data/raw/hrsa/hrsa_facilities_standardized.csv`, 378 pre-geocoded records)
- [x] **NPI Registry gap-fill (uncapped)** (`data/raw/npi/npi_facilities_standardized.csv`, 7,128 records)
- [x] `src/data_collection.py` — ArcGIS pagination helpers, HRSA/HIFLD fetchers, uncapped NPI
- [x] Notebook `01_data_collection.ipynb` — executed with saved outputs

### Phase 2 — Data Cleaning & Geocoding

- [x] `src/geocoding.py` — refactored to N-way `cross_reference_sources(*source_gdfs, priority=[...])` with 250 m spatial match + 1500 m fuzzy-name match (≥ 0.85 token-set ratio) and `batch_only=True` mode for large gap-fill sources
- [x] Notebook `02_data_cleaning.ipynb` — executed with saved outputs (≈95 s runtime)
- [x] Enriched tracts GeoPackage (`data/processed/pa_tracts_enriched.gpkg`, 3,446 tracts, ACS + SVI + density)
- [x] **Populated facilities GeoPackage (`data/processed/pa_facilities.gpkg`, 4,638 unique facilities — up from NPI-capped 1,514)**
- [x] Mount Nittany Medical Center confirmed present (HIFLD + NPI provenance)
- [x] Missing data report generated

### Phase 3 — E2SFCA Analysis

- [x] `src/spatial_analysis.py` — Euclidean + circuity approximation
- [x] Notebook `03_e2sfca_analysis.ipynb` — executed with saved outputs
- [x] `data/outputs/pa_accessibility_scores.gpkg` — 3,446 tracts; 3,439 non-zero scores (99.8 % coverage)
- [x] `data/outputs/pa_drive_times.csv` — 47 MB origin-destination matrix
- [x] `data/outputs/pa_e2sfca_sensitivity.csv` — multi-threshold sensitivity analysis
- [x] `data/outputs/pa_accessibility_map.html` — interactive map (36 MB)

### Phase 4 — Statistical Analysis

- [x] `src/statistics.py` — all functions implemented
- [x] Notebook `04_statistical_analysis.ipynb` — executed with saved outputs
- [x] OLS regression, Moran's I, GWR, LISA all computed

### Phase 5 — Clustering

- [x] `src/clustering.py` — all functions implemented
- [x] Notebook `05_clustering.ipynb` — executed with saved outputs
- [x] `data/outputs/pa_accessibility_clusters.gpkg` — 3,446 tracts with `cluster`, `cluster_hier`, `spatial_cluster`, `lisa_cluster`
- [x] `data/outputs/pa_cluster_profiles.csv` — 4 k-means clusters (n=1040/1596/471/339 tracts)
- [x] `data/outputs/pa_cluster_map.html` — interactive map

### Phase 6 — Visualization & Dashboard

- [x] `src/visualization.py` — all functions implemented
- [x] Notebook `06_visualization.ipynb` — executed with saved outputs (large interactive HTML outputs stripped from the notebook but preserved as stand-alone HTML in `data/outputs/`)
- [x] `dashboard/app.py` — Dash callbacks wired for choropleth + demographic overlays
- [x] HTML map exports: `pa_accessibility_map.html`, `pa_cluster_map.html`, `dashboard_accessibility_overlay.html`, `dashboard_clusters.html`

### Phase 7 — Testing

- [x] `tests/test_spatial_analysis.py` — 9 tests (decay functions)
- [x] `tests/test_geocoding.py` — 18 tests (batch/single-address geocoding, validation, N-way cross-reference, fuzzy-name merge)
- [x] `tests/test_clustering.py` — 17 tests
- [x] `tests/test_data_collection.py` — 17 tests (HIFLD/HRSA/ASC/RHC/CMS POS fetchers, NPI uncapped, Mount Nittany regression)
- [x] `tests/test_statistics.py` — 2 tests
- [x] `tests/test_svi.py` — 3 tests
- [x] **66 total tests — all passing (31.89 s)**

---

## Key Design Decisions

1. **Hybrid facility pipeline** — HIFLD/HRSA/ASC/RHC canonical registries from HRSA's gisportal (pre-geocoded, rooftop-accurate), supplemented by uncapped NPI Registry gap-fill. This replaced the former NPI-only pipeline which was capped at 200 records per taxonomy and missed hospitals like Mount Nittany Medical Center.
2. **HRSA gisportal over HIFLD Open Data** — HIFLD Open Data was deactivated in Aug 2025; HRSA's gisportal canonically hosts the same CMS-approved registry data. Field names changed (`CMS_PROVIDER_ADDRESS` rather than `ADDRESS`), but unit tests retain HIFLD-style fallbacks so mocks still work.
3. **N-way spatial + fuzzy-name merge** — `cross_reference_sources(*gdfs, priority=[...])` with 250 m spatial radius and a 1500 m fallback where ≥ 0.85 token-set name similarity counts as a match.
4. **`batch_only` geocoder mode** — added for large gap-fill sources (e.g. NPI's 7,128 records) to skip the slow per-address Census single-line and Nominatim fallbacks (~hours). The bulk Census batch geocoder handles what it can; remaining un-geocoded records are dropped at merge time rather than blocking the pipeline.
5. **Euclidean + circuity factor** for drive times — 596K-segment full-network routing was computationally infeasible.
6. **Census Bureau batch geocoder** as primary geocoding service (high success rate in batch mode).
7. **GWR** (`mgwr`) wrapped in try/except for environments where it's unavailable.

---

## Live Facility Counts (Pennsylvania)

| Source layer                  | Pre-geocoded records |
|-------------------------------|---------------------:|
| HIFLD Hospitals (HRSA layer 1)| 235                  |
| Critical Access (layer 2)     | 17                   |
| Rural Health Clinics (layer 4)| 75                   |
| ASCs (layer 7)                | 235                  |
| HRSA Health Centers (layer 18)| 378                  |
| **Pre-geocoded subtotal**     | **940**              |
| NPI gap-fill (uncapped)       | 7,128                |
| **Merged unique facilities**  | **4,638**            |

---

## Environment

- Python 3.12.9
- Virtual environment: `.venv/`
- Jupyter kernel: `ds440-venv`
- All packages from `requirements.txt` installed
- **66 unit tests passing**
