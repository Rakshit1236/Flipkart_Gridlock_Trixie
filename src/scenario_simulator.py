import pandas as pd
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FREE_FLOW_SPEED_KMH, JAM_DENSITY_VPKM, RUSH_HOURS, PRI_WEIGHTS
from src.utils import haversine_km, severity_from_violation_types


# Scenario presets
WEATHER_MULTIPLIERS = {
    "Clear": 1.0,
    "Light Rain": 1.25,
    "Heavy Rain": 1.6,
    "Fog": 1.35,
}

DAY_TYPE_MULTIPLIERS = {
    "Weekday": 1.0,
    "Saturday": 1.15,
    "Sunday": 0.85,
    "Festival Day": 1.5,
    "Public Holiday": 0.7,
}


def greenshields_speed_local(blocked_fraction, v_free=None, k_jam=None):
    v = v_free or FREE_FLOW_SPEED_KMH
    k = k_jam or JAM_DENSITY_VPKM
    speed = max(0, v * (1 - blocked_fraction * k / k))
    speed = max(0, v * (1 - blocked_fraction))
    drop = max(0, (1 - speed / v) * 100) if v > 0 else 0
    return speed, drop


def run_scenario(profiles, impact_df, df,
                 vehicle_reduction=0, weather="Clear", day_type="Weekday",
                 focus_cluster_id=None):
    t0 = time.time()
    weather_mult = WEATHER_MULTIPLIERS.get(weather, 1.0)
    day_mult = DAY_TYPE_MULTIPLIERS.get(day_type, 1.0)
    combined_mult = weather_mult * day_mult

    target_ids = [focus_cluster_id] if focus_cluster_id else profiles["cluster_id"].tolist()

    results = []
    for cid in target_ids:
        prof = profiles[profiles["cluster_id"] == cid]
        if len(prof) == 0:
            continue
        prof = prof.iloc[0]
        imp = impact_df[impact_df["cluster_id"] == cid]
        if len(imp) == 0:
            continue
        imp = imp.iloc[0]

        base_drop = imp.get("worst_speed_drop_pct", 0)
        base_speed = imp.get("worst_speed_kmh", FREE_FLOW_SPEED_KMH)
        base_vhl = imp.get("total_vehicle_hours_lost", 0)
        base_violations = imp.get("total_violations", 0)

        # Vehicle reduction effect
        reduction_factor = max(0, 1 - vehicle_reduction / max(base_violations, 1))
        adjusted_drop = base_drop * reduction_factor * combined_mult
        adjusted_speed = FREE_FLOW_SPEED_KMH * (1 - adjusted_drop / 100)
        adjusted_vhl = base_vhl * reduction_factor * combined_mult
        adjusted_violations = int(base_violations * reduction_factor * combined_mult)

        # Queue length estimate (simplified)
        base_queue = int(base_violations * 0.3)
        adjusted_queue = int(adjusted_violations * 0.3)

        # Delay per vehicle (minutes)
        base_delay = max(0, (1 - base_speed / FREE_FLOW_SPEED_KMH) * 15)
        adjusted_delay = max(0, (1 - adjusted_speed / FREE_FLOW_SPEED_KMH) * 15)

        results.append({
            "cluster_id": cid,
            "junction_name": prof.get("junction_name", "Unknown"),
            "area": prof.get("area", ""),
            "centroid_lat": prof["centroid_lat"],
            "centroid_lon": prof["centroid_lon"],
            # Baseline
            "baseline_speed_kmh": round(base_speed, 1),
            "baseline_speed_drop_pct": round(base_drop, 1),
            "baseline_violations": int(base_violations),
            "baseline_queue_length": base_queue,
            "baseline_delay_min": round(base_delay, 1),
            "baseline_vhl": round(base_vhl, 1),
            # Scenario
            "scenario_speed_kmh": round(adjusted_speed, 1),
            "scenario_speed_drop_pct": round(adjusted_drop, 1),
            "scenario_violations": adjusted_violations,
            "scenario_queue_length": adjusted_queue,
            "scenario_delay_min": round(adjusted_delay, 1),
            "scenario_vhl": round(adjusted_vhl, 1),
            # Deltas
            "speed_change": round(adjusted_speed - base_speed, 1),
            "delay_change_min": round(base_delay - adjusted_delay, 1),
            "vhl_change": round(base_vhl - adjusted_vhl, 1),
            "violations_change": int(base_violations - adjusted_violations),
        })

    result_df = pd.DataFrame(results)
    print(f"  Scenario simulated in {time.time()-t0:.1f}s ({len(result_df)} hotspots)")
    return result_df
