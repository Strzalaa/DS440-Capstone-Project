# Comprehensive Project Summary

**DS 440 Capstone — Mapping Healthcare Access Disparities (Group 24)**  
**Geography:** Pennsylvania, census-tract level (~3,446 tracts)  
**Last updated:** April 2026

This document consolidates the project’s purpose, data pipeline, analytical results, and how the interactive dashboard translates findings for exploration. For day-to-day task status, see [PROGRESS.md](PROGRESS.md). For setup and run instructions, see [README.md](README.md).

---

## 1. Project purpose and research question

**Problem.** Geographic, economic, and transportation barriers create inequities in who can reach care within reasonable time. Rural areas, low-income neighborhoods, and populations with limited vehicle access often face long travel times to facilities.

**Goal.** Build a **reproducible, tract-level** framework to:

- **Quantify** healthcare spatial accessibility using the **Enhanced Two-Step Floating Catchment Area (E2SFCA)** method with distance decay.
- **Relate** accessibility to **demographics** and **social vulnerability (SVI)**.
- **Detect** **spatial patterns** (global and local) and **community typologies** (clustering).
- **Deliver** an **interactive dashboard** so patterns can be explored without re-running the full pipeline.

**Outcome.** The project produces enriched tract layers, a merged facility file, E2SFCA scores, statistical models, cluster assignments, and exportable HTML maps—plus a Dash app for policy-oriented exploration.

---

## 2. End-to-end pipeline (what runs where)

| Step | Notebook / artifact | What it does |
|------|---------------------|--------------|
| 1 | `notebooks/01_data_collection.ipynb` | Downloads or fetches raw inputs: ACS, SVI, TIGER, hybrid facility sources (see §3), optional NPI gap-fill. |
| 2 | `notebooks/02_data_cleaning.ipynb` | Geocodes as needed, **N-way merge** of facility sources, builds `pa_tracts_enriched.gpkg` and `pa_facilities.gpkg`. |
| 3 | `notebooks/03_e2sfca_analysis.ipynb` | Computes catchment-based accessibility, drive-time approximations, sensitivity analysis; writes scores and O–D tables. |
| 4 | `notebooks/04_statistical_analysis.ipynb` | OLS, Moran’s I, GWR (when available), LISA-style local indicators. |
| 5 | `notebooks/05_clustering.ipynb` | K-means, hierarchical, and spatially informed labels; cluster GeoPackage and profiles. |
| 6 | `notebooks/06_visualization.ipynb` | Stand-alone Plotly/Folium HTML maps in `data/outputs/`. |
| — | `dashboard/app.py` | Reads processed outputs; interactive **choropleth + overlays + optional facility points**. |

**Source code modules** under `src/` implement configuration, collection, geocoding, E2SFCA and drive-time logic, statistics, clustering, and visualization helpers. **66 automated tests** cover core behavior (`pytest`).

---

## 3. Data sources and the hybrid facility strategy

### 3.1 Context: why the pipeline is “hybrid”

An earlier **NPI-only** design hit the NPI Registry’s **~200 records per taxonomy** cap, which **undercounted** hospitals and other sites (e.g. **Mount Nittany Medical Center** was missing from capped extracts).

The current design uses **canonical, pre-geocoded** facility layers (same CMS-style registry, exposed via **HRSA’s gisportal** after **HIFLD Open Data** was discontinued in 2025) and adds **uncapped NPI** as a **gap-fill** layer, then **deduplicates** in space and by fuzzy facility name (see §3.3).

### 3.2 Major layers (illustrative counts)

| Role | Source / layer | Notes |
|------|------------------|--------|
| Tract boundaries + roads | TIGER/Line (PA) | Tracts for analysis; full road file available but routing was replaced by Euclidean + circuity (§4). |
| Demographics | Census ACS 2022 (tract) | Income, insurance proxies, vehicle access, population, etc. |
| Vulnerability | CDC SVI 2022 | Thematic percentiles; **SVI “overall”** is masked/interpreted with project helpers. |
| Hospitals | HRSA gisportal hospital layer | Pre-coordinates, CMS-aligned fields. |
| CAH, RHC, ASC, HRSA HCs | Additional gisportal layers | Critical Access, Rural Health Clinics, Ambulatory Surgery Centers, Health Center sites. |
| Gap-fill | NPI Registry (uncapped) | Address-level records; **batch Census geocoding** with optional `batch_only` mode to avoid hours of per-address fallbacks. |

**Approximate row counts** from the last documented full run: **~940** pre-geocoded registry rows across those layers, **~7,100+** NPI rows for PA, and **~4,600+** **unique** facilities after merge and geometry filtering (see [PROGRESS.md](PROGRESS.md) for the exact snapshot).

### 3.3 Merging and deduplication (`cross_reference_sources`)

- **Priority order** among sources (hospital/HRSA/urgent/POS/NPI) is explicit so “gold” registries win over NPI duplicates.
- **250 m** spatial match (rooftop-accurate layers).
- Wider **~1,500 m** window with **≥ 0.85** **token-set** name similarity (RapidFuzz, difflib fallback) to catch same facility at slightly different coordinates or labels.
- **Provenance** is tracked (e.g. `hifld_hospitals+npi`) for transparency.

### 3.4 Geocoding design note

- **Census batch** geocoder is the workhorse.
- For very large **address-only** tables (NPI), **`batch_only=True`** can **skip** slow single-line / Nominatim fallback loops; rows that never get coordinates are **dropped** before saving if they have no valid geometry, since they cannot enter spatial analysis.

---

## 4. Accessibility model (E2SFCA) and drive times

- **E2SFCA** assigns **supply** at facilities, applies **decay** by distance to demand (tract population), and aggregates to a tract-level **accessibility score**.
- **Drive times** in this project use a **practical compromise**: **Euclidean distance** with **circuity** (urban vs rural factors) and implied speeds—not full 596K-segment **NetworkX** routing on all TIGER roads, which was **infeasible** at this scale on a single machine.
- **Sensitivity** outputs (e.g. threshold sweeps) are written for robustness checks.

**Empirical score distribution (last run):** about **3,446** tracts with scores; **3,439** tracts with **non-zero** scores (~**99.8%**), mean score on the order of **0.0055** on the E2SFCA scale used in the pipeline (see notebook 03 and `data/outputs/pa_accessibility_scores.gpkg` for exact field definitions).

---

## 5. Statistical results (high level)

All numbers below come from the executed **`04_statistical_analysis.ipynb`** outputs unless noted.

### 5.1 Global OLS (example specification)

- **R² ≈ 0.068** (adjusted **~0.066**): a modest share of variance in **tract accessibility** explained by a simple linear combination of sociodemographic predictors—**expected** when accessibility is spatially smooth and multicausal.
- Coefficient patterns in the saved output include, for example:
  - **Higher % households without a vehicle** associated with **slightly higher** tract accessibility in the baseline OLS (coefficient small but **highly significant** in the reported table)—interpretable in urban cores with many providers but also high need.
  - **Higher SVI overall** associated with **lower** accessibility score (**negative** coefficient, `p < 0.001` in the run shown)—consistent with “vulnerability” aligning with **worse** measured access in that specification.

*Interpretation caveat:* OLS is **aspatial**; **Moran’s I** and **GWR** address spatial structure.

### 5.2 Global Moran’s I (accessibility score)

- **I ≈ 0.63**, with **p ≈ 0.001** in the reported run.
- **Meaning:** accessibility scores are **strongly positively spatially autocorrelated**—similar values cluster in space (e.g. swaths of high or low access), which validates using **local** and **spatial** methods and justifies the cluster narrative.

### 5.3 GWR and LISA

- **GWR** is **optional** in code (environment / `mgwr` availability). If GWR does not run, the notebook can skip local coefficient maps; when it runs, it **localizes** relationships between accessibility and predictors.
- **LISA**-style local clustering highlights **hot/cold** spots; example output rows show tracts tagged **High–Low** vs **not significant** with local **p**-values for exploration.

For **full** coefficient tables, diagnostic plots, and maps, open the executed notebook 04 in Jupyter.

---

## 6. Clustering results (community typologies)

**Method:** K-means (and related labels in the GeoPackage) on tract feature vectors that include accessibility, distance, provider–population ratios, and sociodemographic fields.

**Cluster sizes** (last `pa_cluster_profiles.csv`): **1,040** | **1,596** | **471** | **339** tracts (total **3,446**).

**Mean profiles (abbreviated; see CSV for full columns):**

| Cluster (id) | Tracts (n) | Pop (approx.) | Accessibility (mean) | Nearest facility time (min, mean) | SVI overall (mean) | Median income (mean $) | Story (qualitative) |
|--------------|------------|---------------|----------------------|-----------------------------------|----------------------|-------------------------|------------------------|
| **0** | 1,040 | ~3.94M | ~0.0046 | ~1.4 | **~0.70** (high vulnerability) | ~52,535 | **Higher need / vulnerability**, relatively **short** median drive to nearest site in the aggregate—often **denser** or **mixed** contexts with more facilities but stressed populations. |
| **1** | 1,596 | ~6.47M | ~0.0052 | ~2.5 | **~0.23** (low vulnerability) | **~98,843** | **Largest** cluster: **affluent** tracts, **low SVI**, **favorable** access metrics on average. |
| **2** | 471 | ~1.68M | ~0.0043 | **~13.0** | ~0.36 | ~70,777 | **Longer** travel to nearest facility on average—**rural / peripheral** typology. |
| **3** | 339 | ~0.91M | **~0.0113** (highest) | **~0.8** | ~0.52 | ~65,017 | **Very high** measured accessibility; **extreme** mean provider–population ratio in the file suggests **major hospital / academic** anchoring in some tracts (check for lever tracts in maps). **High** % no-vehicle in the mean profile (~29%)—urban core or safety-net **service–dense** areas with transit-dependent populations. |

These labels are **descriptive** cluster means, not causal claims. Use the **maps and dashboard** to see *where* each cluster lies.

---

## 7. Dashboard: features and “insights”

**Run:** `python dashboard/app.py` (default: local Dash server, e.g. port **8050**).

**Data loaded at startup:**

- `data/outputs/pa_accessibility_clusters.gpkg` (tract polygons + analysis fields).
- `data/processed/pa_facilities.gpkg` (point overlay).

**What users can do:**

1. **Metric dropdown** — Choropleth by:
   - **Accessibility score (E2SFCA)** (default, green scale).
   - **Drive time to nearest facility (minutes)** (orange–red).
   - **Provider-to-population ratio (per 1K)** (blue).
   - **Cluster ID** (categorical / qualitative palette).

2. **Demographic overlay** — Color on hover and **sidebar summary** for:
   - *None*
   - Median household income  
   - % Uninsured  
   - % No vehicle  
   - SVI overall  
   - Population density  

3. **Facility toggle** — Optional **red scatter** points for **every geocoded facility** in the merged file, so users can **see colocation** of tracts and physical sites.

4. **Sidebar “summary stats”** — For the selected **metric** and **overlay**:
   - **Mean, median, min, max, std** for continuous metrics.  
   - For **accessibility score**, also **count of zero-score tracts** (if any).  
   - For **cluster** view, **tract counts per cluster**.  
   - **Global** totals: total tracts, **geocoded facility count**.

5. **Metric vs overlay scatter (small multiple)** — When an overlay is active and the metric is **not** the cluster view, a **bivariate scatter** with **OLS trendline** and **Pearson r** in the title shows the **strength and direction** of the association (e.g. accessibility vs SVI) **within the current selections**.

6. **Cluster view safeguard** — If **one cluster** would dominate a choropleth (e.g. **≥ 90%** of tracts in a single class), the app **hides the cluster map** and instead shows a **typology table** built from `characterize_clusters` so the UI does not imply false geographic diversity. With **balanced** four-cluster output, the **cluster map** is **shown** and remains interpretable.

7. **Hover text** — Tract **GEOID**, accessibility, nearest-facility time, **facilities in catchment** (with caveat on geometry), **population**, and **overlay** value for quick audit.

**Practical “insights” the dashboard is built to support:**

- **Where** accessibility is high vs low **spatially** (E2SFCA and drive time).  
- **How** those patterns align with **income, insurance, vehicle access, SVI, and density** (overlay + scatter **r**).  
- **Which community typology** a tract belongs to and **how** clusters differ in mean need vs mean access (sidebar + table fallback).  
- **Whether** “good access” in raw metrics still coincides with **vulnerable** populations (e.g. **cluster 3** high access + high no-vehicle in means—policy-relevant).  
- **Ground-truth** sense-check: **facility dots** on the basemap vs tract coloring.

---

## 8. Outputs and reproducibility (where files live)

- **Raw / processed / outputs** paths are configured in `src/config.py`.  
- **Large** binaries and CSVs are typically **gitignored** (see `.gitignore`); the **repository** carries **code, notebooks, tests, and documentation**.  
- After a full run, expect outputs such as:  
  - `data/processed/pa_tracts_enriched.gpkg`, `pa_facilities.gpkg`  
  - `data/outputs/pa_accessibility_scores.gpkg`, `pa_accessibility_clusters.gpkg`  
  - `data/outputs/pa_drive_times.csv`, `pa_e2sfca_sensitivity.csv`, `pa_cluster_profiles.csv`  
  - HTML: `pa_accessibility_map.html`, `pa_cluster_map.html`, `dashboard_*.html`  
- **Notebook bloat** from huge Plotly embeds can be **stripped** with `_clear_outputs.py` while keeping small text outputs; **full** interactive HTML remains in `data/outputs/` when generated.

---

## 9. Limitations (explicit)

- **E2SFCA and drive times** are **model-based**; Euclidean + circuity is an **approximation**, not a turn-by-turn route engine.  
- **Facility completeness** still depends on **what each registry** lists and how **merging** deduplicates; NPI gap-fill can add **noise** (duplicates, non-clinical sites) mitigated by **priority and fuzzy matching**.  
- **Ecological** analysis at **tract** level: relationships are **not** individual-level causal effects.  
- **GWR** and some heavy plots may be **environment-dependent** in notebook 04.

---

## 10. Suggested reading order

1. This file (**big picture**).  
2. [README.md](README.md) — setup.  
3. [PROGRESS.md](PROGRESS.md) — current checklist and numbers.  
4. Executed notebooks **03 → 06** for **maps and tables**.  
5. `dashboard/app.py` — **interactive** exploration.

---

*Academic use: Penn State DS 440, Spring 2026. Group 24.*
