# Mapping Healthcare Access Disparities

**Using Geospatial Analysis to Identify Underserved Communities**

DS 440 Capstone Project — Group 24

| Member | Role |
|---|---|
| Nathan Kizlinski | Team Member |
| Eric Strzalkowski | Team Member |
| Cameron Moore | Team Member |
| Rishi Nair | Team Member |
| Zack Ranjan | Team Member |

---

## Problem Statement

Geographic and demographic barriers create significant inequities in healthcare
access across American communities.  Rural areas, low-income neighbourhoods, and
populations with limited transportation face excessive travel times — often
exceeding 30 minutes — to reach the nearest healthcare facility.  This project
develops a comprehensive geospatial analysis framework that quantifies these
disparities at the census-tract level for **Pennsylvania** (~3,200 tracts).

## Methodology

1. **Data Integration** — CMS Provider of Services, HRSA Health Centers, Census
   ACS demographics, CDC Social Vulnerability Index, TIGER/Line shapefiles.
2. **E2SFCA Analysis** — Enhanced Two-Step Floating Catchment Area method with
   Gaussian distance decay and variable urban/suburban/rural catchment sizes.
3. **Statistical Modelling** — OLS and Geographically Weighted Regression (GWR),
   Moran's I, LISA analysis.
4. **Clustering** — K-means and hierarchical clustering to identify community
   typologies with shared access challenges.
5. **Interactive Dashboard** — Plotly Dash web app for policymaker exploration.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Strzalaa/DS440-Capstone-Project.git
cd DS440-Capstone-Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate healthcare_access
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your Census API key
```

Request a free Census API key at <https://api.census.gov/data/key_signup.html>.

### 5. Run the notebooks

Open Jupyter and work through notebooks `01` → `06` in order:

```bash
jupyter notebook notebooks/
```

### 6. Launch the dashboard

```bash
python dashboard/app.py
```

## Project Structure

```
DS440-Capstone-Project/
├── data/
│   ├── raw/              # Downloaded source data
│   ├── processed/        # Cleaned, merged, geocoded data
│   └── outputs/          # Analysis results
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_e2sfca_analysis.ipynb
│   ├── 04_statistical_analysis.ipynb
│   ├── 05_clustering.ipynb
│   └── 06_visualization.ipynb
├── src/                  # Reusable Python modules
│   ├── config.py
│   ├── data_collection.py
│   ├── geocoding.py
│   ├── spatial_analysis.py
│   ├── statistics.py
│   ├── clustering.py
│   └── visualization.py
├── dashboard/            # Plotly Dash web app
│   └── app.py
├── tests/                # Pytest test suite
├── docs/                 # Methodology report & user guide
├── requirements.txt
├── environment.yml
├── .env.example
└── .gitignore
```

## Data Sources

| Source | URL |
|---|---|
| CMS Provider of Services | <https://data.cms.gov/> |
| HRSA Health Centers | <https://data.hrsa.gov/data/download> |
| Census ACS (API) | <https://api.census.gov/> |
| CDC Social Vulnerability Index | <https://svi.cdc.gov/> |
| Census TIGER/Line Shapefiles | <https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html> |

## Key Dependencies

GeoPandas, OSMnx, NetworkX, scikit-learn, statsmodels, mgwr (GWR),
libpysal + esda (spatial statistics), Plotly/Dash, Folium.

See `requirements.txt` for the full list.

## License

This project is for academic purposes (Penn State DS 440, Spring 2026).
