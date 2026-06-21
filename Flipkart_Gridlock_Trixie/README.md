# Bengaluru Parking Intelligence Platform

An AI-powered dashboard that forecasts, explains, simulates, and recommends interventions for illegal parking congestion across Bengaluru.

## Features

| Feature | Description |
|---------|-------------|
| **Historical Heatmaps** | Interactive Folium maps showing violation density, traffic impact severity, and temporal patterns |
| **Traffic Impact Analysis** | Greenshields speed-density model quantifying speed drops, vehicle-hours lost, and capacity reduction per hotspot |
| **Tomorrow's Forecast** | LightGBM + XGBoost ensemble predicting next-day violations with confidence intervals |
| **Priority Scoring** | Weighted composite score (0-100) based on frequency, impact, urgency, and criticality |
| **Parking Risk Index** | Proprietary metric: `0.4×Illegal Parking + 0.3×Density + 0.2×Road Importance + 0.1×Event Score` |
| **Explainable AI** | Root cause breakdown per hotspot into percentage contributors (Illegal Parking, Road Width, Density, etc.) |
| **What-If Simulator** | Interactive toggles to simulate removing vehicles, changing weather, or day type — see instant impact |
| **Congestion Propagation** | Timeline showing how an incident cascades to neighboring hotspots over 15-120 minutes |
| **Early Warning System** | Micro-forecasts for the next 15, 30, and 60 minutes with threat levels |
| **Action Recommendations** | Transforms data into dispatch decisions: deploy officers, expected delay reduction, timing |

## Project Structure

```
THEME_1_New/
├── app.py                     # Streamlit dashboard (entry point)
├── config.py                  # All constants, weights, paths
├── requirements.txt           # Python dependencies
├── .streamlit/config.toml     # Dark theme config
├── data/
│   └── jan to may police violation_anonymized791b166.csv
├── src/
│   ├── data_pipeline.py       # Load, clean, preprocess (298K → 115K rows)
│   ├── clustering.py          # HDBSCAN spatial clustering (~1,000+ hotspots)
│   ├── traffic_impact.py      # Greenshields model + vectorized hourly impact
│   ├── scoring.py             # Priority scorer + Parking Risk Index
│   ├── predictive_model.py    # LightGBM/XGBoost ensemble with Optuna tuning
│   ├── analytics.py           # XAI + propagation + recommendations + early warning
│   ├── scenario_simulator.py  # What-if engine
│   ├── visualization.py       # Folium maps + matplotlib/seaborn charts
│   └── utils.py               # Haversine, road classification, helpers
├── dashboard/
│   ├── tab_main.py            # Overview + heatmaps + forecast + dispatch
│   ├── tab_scenario.py        # What-if simulator UI
│   ├── tab_propagation.py     # Congestion cascade map
│   ├── tab_insights.py        # XAI breakdown + PRI ranking
│   └── tab_actions.py         # Early warnings + action recommendations
└── output/
    ├── heatmaps/              # Generated HTML maps
    ├── reports/               # Dispatch reports + priority scores
    ├── predictions/           # Tomorrow's predictions CSV
    └── models/                # Saved LightGBM/XGBoost models
```

## Setup

### Prerequisites

- Python 3.10+
- 4GB+ RAM recommended (dataset is ~104MB, 298K records)

### Installation

```bash
cd THEME_1_New

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

### First Launch

1. Click **"Run Full Pipeline"** in the sidebar
2. Wait ~40 seconds for the full pipeline to complete:
   - Data loading & cleaning (298K → 115K approved records)
   - HDBSCAN spatial clustering (~1,000+ hotspots)
   - Traffic impact analysis (Greenshields model)
   - Priority scoring + Parking Risk Index
   - ML model training (LightGBM + XGBoost with Optuna)
   - Advanced analytics (XAI, propagation, recommendations, warnings)
   - Visualization generation (4 HTML maps + charts)
3. Explore the 5 tabs

After the first run, subsequent page loads are **instant** from disk cache.

## Dashboard Tabs

### 1. Overview & Maps
- Executive KPIs: total hotspots, critical count, avg speed drop, vehicle-hours lost
- **Heatmaps tab**: Interactive Folium maps (historical, impact, temporal)
- **Tomorrow's Forecast tab**: Predicted violations with confidence intervals
- **Dispatch Report tab**: Text-based dispatch recommendations

### 2. What-If Simulator
- Slide to remove illegally parked vehicles (0-200)
- Select weather: Clear, Light Rain, Heavy Rain, Fog
- Select day type: Weekday, Saturday, Sunday, Festival Day, Public Holiday
- Focus on a specific hotspot or run city-wide
- See before/after comparison: speed, delay, queue length, violations

### 3. Congestion Propagation
- Select incident origin hotspot
- Set forecast horizon (15-120 minutes)
- Set start hour
- Visualize cascade timeline and map
- See which hotspots get affected and when

### 4. Insights (XAI + PRI)
- **Explainable AI tab**: Select any hotspot → see root cause pie chart with percentage breakdown
- **Parking Risk Index tab**: Ranked list of hotspots by proprietary PRI score, distribution histogram

### 5. Actions (Warnings + Dispatch)
- **Early Warning tab**: 15/30/60 minute micro-forecasts with threat levels (HIGH/MEDIUM/LOW)
- **Action Recommendations tab**: Decision cards — deploy officers, dispatch timing, expected delay reduction

## Dataset

Source: Bengaluru police illegal parking violations (anonymized)
- **Records**: 298,450 raw → 115,400 approved
- **Time range**: November 2023 – March 2024
- **Columns**: 24 (geospatial, temporal, vehicle type, violation type, enforcement status)
- **Coverage**: Bengaluru, Karnataka, India (Madiwala, Bellandur, HSR Layout, Electronic City, etc.)

## Data Pipeline

```
Raw CSV (298K rows)
  → Clean (approved only, valid geo, India bbox): 115K rows
  → Preprocess (+15 features: temporal, severity, road, junction, event score)
  → HDBSCAN Clustering (~1,000+ spatial hotspots)
  → Traffic Impact (Greenshields speed model, hourly breakdown)
  → Priority Scoring (0-100 composite) + Parking Risk Index
  → LightGBM/XGBoost Ensemble (tomorrow's forecast + confidence intervals)
  → Advanced Analytics (XAI, propagation, recommendations, warnings)
  → Dashboard visualizations
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | pandas, numpy |
| Clustering | HDBSCAN (haversine metric) |
| ML Models | LightGBM, XGBoost, Optuna (hyperparameter tuning) |
| Traffic Model | Greenshields speed-density + HCM capacity |
| Dashboard | Streamlit |
| Maps | Folium (CartoDB tiles) |
| Charts | Plotly, matplotlib, seaborn |
| Caching | pickle (disk cache for instant restarts) |

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FREE_FLOW_SPEED_KMH` | 40.0 | Free-flow speed for Greenshields model |
| `JAM_DENSITY_VPKM` | 200.0 | Jam density (vehicles/km) |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 20 | Minimum cluster size for HDBSCAN |
| `PRIORITY_WEIGHTS` | freq=0.25, impact=0.35, urgency=0.20, criticality=0.20 | Priority score weights |
| `PRI_WEIGHTS` | illegal=0.4, density=0.3, road=0.2, event=0.1 | Parking Risk Index weights |
| `PROPAGATION_RADIUS_KM` | 2.0 | Max distance for congestion propagation |
| `CHRONIC_THRESHOLD_PCT` | 0.30 | % days a hotspot must appear to be "chronic" |

## Output Files

After running the pipeline:

| File | Description |
|------|-------------|
| `output/heatmaps/hotspot_heatmap.html` | Violation density map |
| `output/heatmaps/impact_map.html` | Traffic impact severity map |
| `output/heatmaps/predictions_heatmap.html` | Tomorrow's predicted hotspots |
| `output/heatmaps/temporal_heatmap.png` | Day-of-week × hour heatmap |
| `output/reports/dispatch_YYYY-MM-DD.txt` | Human-readable dispatch report |
| `output/reports/priority_scores_YYYY-MM-DD.csv` | Priority scores for all hotspots |
| `output/reports/priority_chart.png` | Priority score bar charts |
| `output/predictions/predictions_YYYY-MM-DD.csv` | ML predictions with confidence |
| `output/models/lgb_model.pkl` | Trained LightGBM model |
| `output/models/xgb_model.pkl` | Trained XGBoost model |
| `output/pipeline_cache.pkl` | Cached pipeline results (instant reload) |

## Build by Trixie