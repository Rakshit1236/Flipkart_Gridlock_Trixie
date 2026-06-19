import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    FREE_FLOW_SPEED_KMH, JAM_DENSITY_VPKM, HCM_LANE_CAPACITY_VPH,
    IMPACT_DECAY_FACTOR, LANE_ESTIMATE,
)


def compute_blocked_lanes(violations_in_window):
    return violations_in_window["severity_weight"].sum()


def greenshields_speed(blocked_lanes, total_lanes, v_free=None, k_jam=None):
    if v_free is None:
        v_free = FREE_FLOW_SPEED_KMH
    if k_jam is None:
        k_jam = JAM_DENSITY_VPKM
    effective_lanes = max(total_lanes - blocked_lanes, 0.5)
    density = (blocked_lanes * 100) / max(total_lanes, 1)
    speed = v_free * max(1 - density / k_jam, 0.1)
    speed_drop_pct = (1 - speed / v_free) * 100 if v_free > 0 else 0
    return speed, speed_drop_pct


def capacity_remaining_pct(blocked_lanes, total_lanes):
    effective = max(total_lanes - blocked_lanes, 0)
    return (effective / max(total_lanes, 1)) * 100


def vehicle_hours_lost(violations_in_window):
    total_minutes = (violations_in_window["severity_weight"] * violations_in_window["violation_duration_minutes"]).sum()
    return total_minutes / 60.0


def estimate_hourly_impact(cluster_df, total_lanes):
    results = []
    for hour in range(24):
        hour_df = cluster_df[cluster_df["hour"] == hour]
        if len(hour_df) == 0:
            results.append({
                "hour": hour,
                "violations": 0,
                "blocked_lanes": 0,
                "estimated_speed_kmh": FREE_FLOW_SPEED_KMH,
                "speed_drop_pct": 0,
                "capacity_remaining_pct": 100.0,
                "vehicle_hours_lost": 0,
                "impact_severity": "NONE",
            })
            continue

        blocked = compute_blocked_lanes(hour_df)
        speed, speed_drop = greenshields_speed(blocked, total_lanes)
        cap_remain = capacity_remaining_pct(blocked, total_lanes)
        vhl = vehicle_hours_lost(hour_df)

        if speed_drop >= 40:
            severity = "CRITICAL"
        elif speed_drop >= 25:
            severity = "HIGH"
        elif speed_drop >= 10:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        results.append({
            "hour": hour,
            "violations": len(hour_df),
            "blocked_lanes": round(blocked, 2),
            "estimated_speed_kmh": round(speed, 1),
            "speed_drop_pct": round(speed_drop, 1),
            "capacity_remaining_pct": round(cap_remain, 1),
            "vehicle_hours_lost": round(vhl, 2),
            "impact_severity": severity,
        })
    return pd.DataFrame(results)


def compute_ripple_effect(cluster_profiles, df):
    print("[traffic_impact] Computing ripple effects...")
    if "police_station" in df.columns:
        jurisdiction_groups = df.groupby("police_station")
    else:
        jurisdiction_groups = df.groupby("area")

    ripple_scores = {}
    for jname, jdf in jurisdiction_groups:
        clusters_in_jurisdiction = jdf["cluster_id"].unique()
        clusters_in_jurisdiction = [c for c in clusters_in_jurisdiction if c != -1]
        if len(clusters_in_jurisdiction) <= 1:
            continue

        cluster_profiles_j = cluster_profiles[cluster_profiles["cluster_id"].isin(clusters_in_jurisdiction)]
        total_impact = cluster_profiles_j["avg_severity"].mean() * cluster_profiles_j["total_violations"].mean()

        for cid in clusters_in_jurisdiction:
            ripple_scores[cid] = IMPACT_DECAY_FACTOR * total_impact / len(clusters_in_jurisdiction)

    return ripple_scores


def run_impact_analysis(df, cluster_profiles):
    t0 = time.time()
    print("[traffic_impact] Running impact analysis for each hotspot...")

    impact_results = []
    hourly_details = {}

    for _, profile in cluster_profiles.iterrows():
        cid = profile["cluster_id"]
        cluster_df = df[df["cluster_id"] == cid].copy()

        if "num_lanes" in profile.index:
            total_lanes = profile["num_lanes"]
        else:
            from src.utils import classify_road_type
            road_type = cluster_df["road_type"].mode()
            road_type = road_type.iloc[0] if len(road_type) > 0 else "Other"
            total_lanes = LANE_ESTIMATE.get(road_type, 2)

        hourly = estimate_hourly_impact(cluster_df, total_lanes)
        hourly_details[cid] = hourly

        peak_hour_data = hourly[hourly["violations"] > 0].sort_values("speed_drop_pct", ascending=False)
        if len(peak_hour_data) == 0:
            worst_hour = 0
            worst_speed_drop = 0
            worst_speed = FREE_FLOW_SPEED_KMH
            total_vhl = 0
            max_severity = "NONE"
        else:
            worst = peak_hour_data.iloc[0]
            worst_hour = int(worst["hour"])
            worst_speed_drop = worst["speed_drop_pct"]
            worst_speed = worst["estimated_speed_kmh"]
            total_vhl = hourly["vehicle_hours_lost"].sum()
            max_severity = worst["impact_severity"]

        impact_results.append({
            "cluster_id": cid,
            "label": profile["label"],
            "centroid_lat": profile["centroid_lat"],
            "centroid_lon": profile["centroid_lon"],
            "total_lanes": total_lanes,
            "worst_hour": worst_hour,
            "worst_speed_drop_pct": worst_speed_drop,
            "worst_estimated_speed_kmh": worst_speed,
            "total_vehicle_hours_lost": round(total_vhl, 2),
            "impact_severity": max_severity,
            "total_violations": profile["total_violations"],
        })

    impact_df = pd.DataFrame(impact_results)

    ripple_scores = compute_ripple_effect(cluster_profiles, df)
    impact_df["ripple_score"] = impact_df["cluster_id"].map(ripple_scores).fillna(0)

    impact_df = impact_df.sort_values("worst_speed_drop_pct", ascending=False).reset_index(drop=True)

    t1 = time.time()
    print("  Impact analysis complete in {:.1f}s".format(t1 - t0))
    print("  Top 5 most impactful hotspots:")
    for _, row in impact_df.head(5).iterrows():
        print("    {}: -{:.0f}% speed at hour {}, severity={}".format(
            row["label"], row["worst_speed_drop_pct"], row["worst_hour"], row["impact_severity"]))

    return impact_df, hourly_details
