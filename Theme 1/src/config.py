import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_CSV = os.path.join(BASE_DIR, "jan to may police violation_anonymized791b166.csv")

HEATMAP_DIR = os.path.join(OUTPUT_DIR, "heatmaps")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

FREE_FLOW_SPEED_KMH = 40.0
JAM_DENSITY_VPKM = 150.0

HCM_LANE_CAPACITY_VPH = {1: 1800, 2: 3600, 3: 5400, 4: 7200}

LANE_ESTIMATE = {
    "Main Road": 3,
    "Cross Road": 2,
    "Underpass": 2,
    "Ring Road": 4,
    "Other": 2,
}

SEVERITY_WEIGHTS = {
    "DOUBLE PARKING": 2.0,
    "PARKING IN A MAIN ROAD": 1.5,
    "PARKING NEAR ROAD CROSSING": 1.5,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 1.8,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 1.3,
    "WRONG PARKING": 1.0,
    "NO PARKING": 0.8,
}

PRIORITY_WEIGHTS = {
    "frequency": 0.25,
    "impact": 0.35,
    "urgency": 0.20,
    "criticality": 0.20,
}

HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES = 10

RUSH_HOURS = [(7, 10), (17, 21)]

IMPACT_DECAY_FACTOR = 0.3
CHRONIC_THRESHOLD_PCT = 0.30

os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
