import math
import re


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def classify_road_type(location_text):
    if not location_text or not isinstance(location_text, str):
        return "Other"
    lower = location_text.lower()
    if "ring road" in lower or "outer ring" in lower:
        return "Ring Road"
    if "main road" in lower or "main street" in lower:
        return "Main Road"
    if "underpass" in lower:
        return "Underpass"
    if "cross road" in lower or "cross" in lower:
        return "Cross Road"
    return "Other"


def extract_road_name(location_text):
    if not location_text or not isinstance(location_text, str):
        return "Unknown"
    parts = location_text.split(",")
    first_part = parts[0].strip()
    road_match = re.match(
        r"^([\d\w\s]+(?:Road|Street|Lane|Avenue|Cross|Path|Ring|Underpass|Junction))",
        first_part,
        re.IGNORECASE,
    )
    if road_match:
        return road_match.group(1).strip()
    return first_part


def extract_area(location_text):
    if not location_text or not isinstance(location_text, str):
        return "Unknown"
    parts = [p.strip() for p in location_text.split(",")]
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


def is_rush_hour(hour, rush_hours=None):
    if rush_hours is None:
        rush_hours = [(7, 10), (17, 21)]
    for start, end in rush_hours:
        if start <= hour < end:
            return True
    return False


def severity_from_violation_types(violation_str):
    from src.config import SEVERITY_WEIGHTS
    if not violation_str or not isinstance(violation_str, str):
        return 0.8, "NO PARKING"
    max_weight = 0.0
    dominant = "NO PARKING"
    for vtype, weight in SEVERITY_WEIGHTS.items():
        if vtype in violation_str.upper():
            if weight > max_weight:
                max_weight = weight
                dominant = vtype
    if max_weight == 0.0:
        return 0.8, "NO PARKING"
    return max_weight, dominant
