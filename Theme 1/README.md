# AI-Driven Illegal Parking Intelligence System

**Bengaluru Traffic Department — Flipkart Grid Hackathon (Theme 1)**

An AI system that answers two critical questions for traffic enforcement:
1. **Where & When** are the worst illegal parking hotspots happening?
2. **How badly** is each hotspot choking the city's traffic?

---

## Quick Start

### 1. Install Dependencies

```bash
cd "Theme 1"
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python main.py
```

This will:
- Load and clean 115K+ approved violations from the 300K dataset
- Cluster hotspots using HDBSCAN spatial-temporal analysis
- Quantify traffic impact using Greenshields + HCM physics model
- Score priority for enforcement dispatch
- Train a LightGBM predictive model
- Generate heatmaps, charts, and dispatch reports

**Runtime:** ~66 seconds on CPU

### 3. View Dashboard

```bash
cd "Theme 1"
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501** with 5 tabs:
- **Overview** — Summary metrics, violation distribution
- **Hotspot Map** — Interactive Folium heatmap
- **Priority Rankings** — Sortable table with score breakdown
- **Impact Analysis** — Speed reduction and vehicle-hours lost
- **Dispatch Report** — Daily enforcement recommendations

### 4. CLI Dispatch Recommendations

```bash
python cli.py --top 10
```

Outputs the top 10 enforcement priorities with location, dispatch time, speed impact, and severity.

---

## Project Structure

```
Theme 1/
├── main.py                  # Full pipeline orchestrator
├── cli.py                   # CLI for daily dispatch recommendations
├── requirements.txt         # Python dependencies
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── src/
│   ├── config.py            # Constants, HCM configs, scoring weights
│   ├── utils.py             # Haversine, address parsing, helpers
│   ├── data_loader.py       # Load & clean 300K CSV
│   ├── preprocessing.py     # Temporal features, severity, road extraction
│   ├── hotspot_clustering.py # HDBSCAN spatial-temporal clustering
│   ├── traffic_impact.py    # Greenshields + HCM speed estimation
│   ├── priority_scorer.py   # Composite priority score (0-100)
│   ├── predictive_model.py  # LightGBM hotspot recurrence forecasting
│   └── visualization.py     # Heatmaps, charts, Folium maps
├── output/
│   ├── heatmaps/            # Interactive HTML heatmaps
│   ├── reports/             # Dispatch reports, priority CSVs
│   └── predictions/         # Model predictions
└── data/                    # (symlink to dataset)
```

---

## Architecture

### Step 1: Hotspot Clustering (The "Where" & "When")
- **Algorithm:** HDBSCAN (haversine metric, min_cluster_size=20)
- **Output:** 1,086 hotspot clusters from 115K violations
- **Features:** Peak hours, peak day, dominant violation type, chronic flag

### Step 2: Traffic Impact Quantification (The "How Bad")
- **Model:** Greenshields linear speed-density relationship
- **Formula:** `v = v_free × (1 - k / k_jam)`
- **Parameters:** Free-flow speed = 40 km/h, jam density = 150 veh/km/lane
- **Severity weighting:** DOUBLE PARKING = 2.0x, MAIN ROAD = 1.5x, etc.
- **Ripple effect:** 30% decay factor for adjacent junctions in same jurisdiction

### Step 3: Priority Scoring
```
Priority = 0.25×Frequency + 0.35×Impact + 0.20×Urgency + 0.20×Criticality
```
- **Frequency:** How often the hotspot recurs
- **Impact:** Speed drop percentage (Greenshields output)
- **Urgency:** Whether current time falls in peak window
- **Criticality:** Road type (Ring Road > Main Road > Cross Road)

### Step 4: Predictive Model
- **Algorithm:** LightGBM binary classifier
- **Features:** Rolling 3/7/14-day violation counts, severity, days since last
- **Performance:** AUC 0.638 (5-fold CV)
- **Output:** Probability of each hotspot being active tomorrow

---

## Key Results

| Metric | Value |
|--------|-------|
| Raw dataset | 298,450 violations |
| Approved violations | 115,400 |
| Hotspot clusters | 1,086 |
| Top hotspot | New Horizon College Road (1,682 violations) |
| Most impactful | BTP051 - Safina Plaza Junction (-90% speed) |
| Model AUC | 0.638 |
| Pipeline runtime | 66 seconds |

---

## Output Examples

### Dispatch Report
```
#1  PRIORITY SCORE: 73.1/100
    Location: BTP040 - Elite Junction
    Action:  Send enforcement team to BTP040 - Elite Junction at 3:00
    Reason:  Illegal parking there is currently dropping average traffic
             speed by 90%. NO PARKING violations
    Impact:  Severity=CRITICAL
    Violations: 763 total, 66 active days
```

### Generated Files
- `output/heatmaps/hotspot_heatmap.html` — Interactive violation heatmap
- `output/heatmaps/impact_map.html` — Traffic impact visualization
- `output/heatmaps/temporal_heatmap.html` — Day × Hour violation matrix
- `output/reports/dispatch_YYYY-MM-DD.txt` — Daily dispatch report
- `output/reports/priority_scores_YYYY-MM-DD.csv` — All hotspot scores

---

## Dataset

**File:** `jan to may police violation_anonymized791b166.csv`

| Column | Description |
|--------|-------------|
| `latitude`, `longitude` | GPS coordinates |
| `created_datetime` | Violation timestamp |
| `closed_datetime` | Resolution timestamp |
| `violation_type` | WRONG PARKING, NO PARKING, DOUBLE PARKING, etc. |
| `vehicle_type` | CAR, SCOOTER, MOTOR CYCLE, AUTO, etc. |
| `police_station` | Jurisdiction |
| `junction_name` | Intersection identifier |
| `location` | Full address string |

---

## Tech Stack

- **Clustering:** HDBSCAN
- **ML:** LightGBM, Scikit-learn
- **Traffic Model:** Greenshields + Highway Capacity Manual (HCM)
- **Visualization:** Folium, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **Data:** Pandas, NumPy
