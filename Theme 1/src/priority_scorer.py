import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import PRIORITY_WEIGHTS, RUSH_HOURS


def normalize_score(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(50.0, index=series.index)
    return ((series - min_val) / (max_val - min_val)) * 100


def compute_frequency_score(cluster_profiles, total_days):
    scores = (cluster_profiles["unique_days"] / max(total_days, 1)) * 100
    return scores.clip(0, 100)


def compute_impact_score(impact_df):
    return normalize_score(impact_df["worst_speed_drop_pct"])


def compute_urgency_score(cluster_profiles, current_hour=None):
    if current_hour is None:
        current_hour = pd.Timestamp.now().hour

    scores = []
    for _, row in cluster_profiles.iterrows():
        peak_hours = row["peak_hours"]
        if isinstance(peak_hours, str):
            import ast
            peak_hours = ast.literal_eval(peak_hours)

        min_dist = min(abs(current_hour - ph) for ph in peak_hours) if peak_hours else 24
        if min_dist == 0:
            score = 100
        elif min_dist <= 2:
            score = 70
        elif min_dist <= 4:
            score = 40
        else:
            score = 10
        scores.append(score)
    return pd.Series(scores, index=cluster_profiles.index)


def compute_criticality_score(cluster_profiles):
    road_scores = {
        "Ring Road": 100,
        "Main Road": 80,
        "Underpass": 70,
        "Cross Road": 50,
        "Other": 30,
    }
    scores = cluster_profiles.apply(
        lambda row: road_scores.get(row.get("road_type", "Other"), 30)
        + (15 if row.get("has_junction", 0) == 1 else 0),
        axis=1,
    )
    return scores.clip(0, 100)


def compute_priority_scores(cluster_profiles, impact_df, total_days, current_hour=None):
    t0 = time.time()
    print("[priority] Computing priority scores...")

    merged = cluster_profiles.merge(
        impact_df[["cluster_id", "worst_speed_drop_pct", "impact_severity", "total_vehicle_hours_lost"]],
        on="cluster_id", how="left"
    )

    freq = compute_frequency_score(merged, total_days)
    impact = compute_impact_score(impact_df)
    urgency = compute_urgency_score(merged, current_hour)
    criticality = compute_criticality_score(merged)

    w = PRIORITY_WEIGHTS
    priority = (
        w["frequency"] * freq.values
        + w["impact"] * impact.values
        + w["urgency"] * urgency.values
        + w["criticality"] * criticality.values
    )

    merged["frequency_score"] = freq.values
    merged["impact_score"] = impact.values
    merged["urgency_score"] = urgency.values
    merged["criticality_score"] = criticality.values
    merged["priority_score"] = np.round(priority, 1)
    merged = merged.sort_values("priority_score", ascending=False).reset_index(drop=True)

    t1 = time.time()
    print("  Priority scoring complete in {:.1f}s".format(t1 - t0))
    print("  Top 10 priority hotspots:")
    for idx, (_, row) in enumerate(merged.head(10).iterrows()):
        spd = row.get("worst_speed_drop_pct", 0)
        sev = row.get("impact_severity", "N/A")
        print("    #{} [{}/100] {} - Speed drop: {:.0f}%, Severity: {}".format(
            idx + 1, row["priority_score"], row["label"], spd, sev))

    return merged
