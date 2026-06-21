import math
import re
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEVERITY_WEIGHTS


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def classify_road_type(text):
    if not isinstance(text, str):
        return "Other"
    t = text.lower()
    if "ring road" in t or "ring rd" in t:
        return "Ring Road"
    if "main road" in t or "main rd" in t:
        return "Main Road"
    if "underpass" in t:
        return "Underpass"
    if "cross" in t:
        return "Cross Road"
    return "Other"


def extract_road_name(text):
    if not isinstance(text, str):
        return "Unknown"
    parts = [p.strip() for p in text.split(",")]
    return parts[0] if parts else "Unknown"


def extract_area(text):
    if not isinstance(text, str):
        return "Unknown"
    parts = [p.strip() for p in text.split(",")]
    return parts[1] if len(parts) > 1 else (parts[0] if parts else "Unknown")


def is_rush_hour(hour, rush_hours):
    return any(start <= hour < end for start, end in rush_hours)


def severity_from_violation_types(violation_str):
    if not isinstance(violation_str, str):
        return 0.8, "NO PARKING"
    best_w, best_t = 0.8, "NO PARKING"
    for vtype, weight in SEVERITY_WEIGHTS.items():
        if vtype in violation_str.upper():
            if weight > best_w:
                best_w, best_t = weight, vtype
    return best_w, best_t


def normalize_to_0_100(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50, index=series.index)
    return ((series - mn) / (mx - mn) * 100).clip(0, 100)


import pandas as pd
