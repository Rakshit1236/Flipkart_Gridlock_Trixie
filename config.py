import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

RAW_CSV = os.path.join(BASE_DIR, "jan to may police violation_anonymized791b166.csv")
if not os.path.exists(RAW_CSV):
    RAW_CSV = os.path.join(DATA_DIR, "jan to may police violation_anonymized791b166.csv")

for d in [HEATMAP_DIR, REPORT_DIR, PRED_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Traffic Model ---
FREE_FLOW_SPEED_KMH = 40.0
JAM_DENSITY_VPKM = 200.0
HCM_LANE_CAPACITY_VPH = {1: 1800, 2: 3600, 3: 5400, 4: 7200}
LANE_ESTIMATE = {
    "Ring Road": 4, "Main Road": 3, "Cross Road": 2,
    "Underpass": 2, "Other": 2,
}

# --- Severity ---
SEVERITY_WEIGHTS = {
    "PARKING IN A MAIN ROAD": 2.0,
    "WRONG PARKING": 1.5,
    "DOUBLE PARKING": 1.8,
    "NO PARKING": 1.2,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 1.4,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 1.6,
    "DEFECTIVE NUMBER PLATE": 0.8,
}

# --- Priority Scoring ---
PRIORITY_WEIGHTS = {
    "frequency": 0.25,
    "impact": 0.35,
    "urgency": 0.20,
    "criticality": 0.20,
}

# --- Parking Risk Index ---
PRI_WEIGHTS = {
    "illegal_parking": 0.4,
    "density": 0.3,
    "road_importance": 0.2,
    "event_score": 0.1,
}

# --- Clustering ---
HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES = 10
CHRONIC_THRESHOLD_PCT = 0.30

# --- Time ---
RUSH_HOURS = [(7, 10), (17, 21)]
IMPACT_DECAY_FACTOR = 0.3
PROPAGATION_RADIUS_KM = 2.0
