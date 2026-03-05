# Project Progress Tracker

**DS 440 Capstone — Mapping Healthcare Access Disparities (Group 24)**

Last updated: 2026-02-26

---

## Pipeline Overview

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data Collection (ACS, SVI, TIGER, CMS, HRSA) | In Progress |
| 2 | Data Cleaning & Geocoding | In Progress |
| 3 | E2SFCA Spatial Accessibility Analysis | Not Started |
| 4 | Statistical Analysis (OLS, GWR, Moran's I, LISA) | Not Started |
| 5 | Clustering (K-means, Hierarchical, Spatial) | Not Started |
| 6 | Visualization & Dashboard | Not Started |
| 7 | Testing | Not Started |

---

## Detailed Task Checklist

### Phase 1 — Data Collection

- [x] Census ACS demographics downloaded (`data/raw/acs/acs_2022_tract_42.csv`)
- [x] CDC SVI downloaded (`data/raw/svi/SVI_2022_Pennsylvania.csv`)
- [x] TIGER/Line tracts shapefile (`data/raw/tiger/tl_2024_42_tract.gpkg`)
- [x] TIGER/Line roads — all 67 PA counties (`data/raw/tiger/tl_2024_42_roads.gpkg`, 325 MB, 596K segments)
- [ ] HRSA Health Center facility data (via API)
- [ ] CMS / HIFLD hospital facility data
- [x] `src/data_collection.py` — all functions implemented
- [x] Notebook `01_data_collection.ipynb` — cells populated and executed

### Phase 2 — Data Cleaning & Geocoding

- [x] `src/geocoding.py` — all functions implemented
- [x] Notebook `02_data_cleaning.ipynb` — cells populated and executed
- [x] Enriched tracts GeoPackage (`data/processed/pa_tracts_enriched.gpkg`, 23.5 MB)
- [ ] Populated facilities GeoPackage (`data/processed/pa_facilities.gpkg` — currently 0 rows)
- [x] Missing data report generated

### Phase 3 — E2SFCA Analysis

- [x] `src/spatial_analysis.py` — all functions implemented
- [ ] Optimize `build_road_network` / `compute_drive_times` for feasible runtime
- [ ] Notebook `03_e2sfca_analysis.ipynb` — executed with saved outputs
- [ ] `data/outputs/pa_accessibility_scores.gpkg` generated
- [ ] `data/outputs/pa_drive_times.csv` generated
- [ ] `data/outputs/pa_e2sfca_sensitivity.csv` generated

### Phase 4 — Statistical Analysis

- [x] `src/statistics.py` — all functions implemented
- [ ] Notebook `04_statistical_analysis.ipynb` — executed with saved outputs

### Phase 5 — Clustering

- [x] `src/clustering.py` — all functions implemented
- [ ] Notebook `05_clustering.ipynb` — executed with saved outputs
- [ ] `data/outputs/pa_accessibility_clusters.gpkg` generated
- [ ] `data/outputs/pa_cluster_profiles.csv` generated

### Phase 6 — Visualization & Dashboard

- [x] `src/visualization.py` — all functions implemented
- [ ] Notebook `06_visualization.ipynb` — executed with saved outputs
- [ ] `dashboard/app.py` — wire callbacks for interactive map
- [ ] HTML map exports in `data/outputs/`

### Phase 7 — Testing

- [x] `tests/test_spatial_analysis.py` — 9 real tests (decay functions)
- [ ] `tests/test_geocoding.py` — implement real tests
- [ ] `tests/test_clustering.py` — implement real tests
- [ ] `tests/test_data_collection.py` — implement real tests

---

## Known Issues / Notes

1. **CMS/HRSA raw files** must be acquired via API; manual download URLs are not stable.
2. **Road network performance**: 596K TIGER segments make row-by-row NetworkX graph construction infeasible. Solution: Euclidean distance with circuity factors.
3. **GWR** (`mgwr`) may fail on some environments; wrapped in try/except.
4. **Jupyter kernel**: Use `ds440-venv` kernel (registered from `.venv`) for notebook execution.
5. **`.gitignore`** excludes all data files (`.csv`, `.gpkg`, etc.); only code and notebooks (with outputs) are tracked in Git.

---

## Environment

- Python 3.12.9
- Virtual environment: `.venv/`
- Jupyter kernel: `ds440-venv`
- All packages from `requirements.txt` installed
