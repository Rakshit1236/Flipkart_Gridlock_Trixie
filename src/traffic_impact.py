import pandas as pd
import numpy as np
import os, sys, time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FREE_FLOW_SPEED_KMH, JAM_DENSITY_VPKM, RUSH_HOURS, IMPACT_DECAY_FACTOR, PROPAGATION_RADIUS_KM


def run_impact(df, profiles):
    print("[Impact] Computing traffic impact (vectorized)...")
    t0 = time.time()

    # Pre-group once instead of filtering 1086 times
    df_valid = df[df["cluster_id"] != -1].copy()

    # Vectorized hourly aggregation for ALL clusters at once
    hourly_agg = df_valid.groupby(["cluster_id", "hour"]).agg(
        blocked_severity=("severity_weight", "sum"),
        vhl=("severity_weight", lambda x: (x * df_valid.loc[x.index, "violation_duration_minutes"] / 60).sum()),
        violations=("id", "count"),
    ).reset_index()

    # Merge lanes from profiles
    lane_map = profiles.set_index("cluster_id")["num_lanes"].to_dict() if "num_lanes" in profiles.columns else {}
    hourly_agg["total_lanes"] = hourly_agg["cluster_id"].map(lane_map).fillna(2)
    hourly_agg["blocked_fraction"] = (hourly_agg["blocked_severity"] / hourly_agg["total_lanes"]).clip(0, 0.95)

    # Vectorized Greenshields
    hourly_agg["speed_kmh"] = (FREE_FLOW_SPEED_KMH * (1 - hourly_agg["blocked_fraction"])).clip(0)
    hourly_agg["speed_drop_pct"] = ((1 - hourly_agg["speed_kmh"] / FREE_FLOW_SPEED_KMH) * 100).clip(0)
    hourly_agg["capacity_remaining"] = ((1 - hourly_agg["blocked_fraction"]) * 100).clip(0)

    def _sev(d):
        return "CRITICAL" if d >= 40 else ("HIGH" if d >= 25 else ("MEDIUM" if d >= 10 else "LOW"))
    hourly_agg["severity"] = hourly_agg["speed_drop_pct"].apply(_sev)

    # Find worst hour per cluster
    idx = hourly_agg.groupby("cluster_id")["speed_drop_pct"].idxmax()
    worst = hourly_agg.loc[idx].copy()

    # Total VHL per cluster
    vhl_sum = hourly_agg.groupby("cluster_id")["vhl"].sum().reset_index()
    vhl_sum.columns = ["cluster_id", "total_vehicle_hours_lost"]

    avg_speed = hourly_agg.groupby("cluster_id")["speed_kmh"].mean().reset_index()
    avg_speed.columns = ["cluster_id", "avg_speed_kmh"]

    total_viol = hourly_agg.groupby("cluster_id")["violations"].sum().reset_index()
    total_viol.columns = ["cluster_id", "total_violations"]

    prof_cols = ["cluster_id", "junction_name", "centroid_lat", "centroid_lon"]
    impact_df = profiles[prof_cols].copy()
    impact_df = impact_df.merge(worst[["cluster_id", "hour", "speed_kmh", "speed_drop_pct", "severity"]],
                                on="cluster_id", how="left")
    impact_df = impact_df.merge(vhl_sum, on="cluster_id", how="left")
    impact_df = impact_df.merge(avg_speed, on="cluster_id", how="left")
    impact_df = impact_df.merge(total_viol, on="cluster_id", how="left")

    impact_df.rename(columns={"hour": "worst_hour", "speed_kmh": "worst_speed_kmh",
                              "speed_drop_pct": "worst_speed_drop_pct", "severity": "worst_severity"}, inplace=True)
    impact_df = impact_df.fillna(0).sort_values("worst_speed_drop_pct", ascending=False).reset_index(drop=True)

    # Ripple - only top 100 hotspots by violations (skip O(n^2) for all)
    top_cids = profiles.nlargest(100, "total_violations")["cluster_id"].tolist()
    prof_top = profiles[profiles["cluster_id"].isin(top_cids)]
    latlons = prof_top[["centroid_lat", "centroid_lon"]].values
    ripples = []
    for i, cid in enumerate(top_cids):
        neighbors = []
        for j, cid2 in enumerate(top_cids):
            if i >= j:
                continue
            dlat = (latlons[i][0] - latlons[j][0]) * 111
            dlon = (latlons[i][1] - latlons[j][1]) * 111 * np.cos(np.radians(latlons[i][0]))
            d = np.sqrt(dlat**2 + dlon**2)
            if d <= PROPAGATION_RADIUS_KM:
                neighbors.append({"neighbor_id": cid2, "distance_km": round(d, 2),
                                  "decay_factor": round(IMPACT_DECAY_FACTOR ** d, 3)})
        if neighbors:
            ripples.append({"cluster_id": cid, "neighbors": neighbors})
    ripples_df = pd.DataFrame(ripples) if ripples else pd.DataFrame()

    print(f"  Impact computed in {time.time()-t0:.1f}s")
    return impact_df, {}, ripples_df
