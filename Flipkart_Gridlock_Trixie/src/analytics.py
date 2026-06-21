import pandas as pd
import numpy as np
import os, sys, time
from datetime import timedelta, datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROPAGATION_RADIUS_KM, IMPACT_DECAY_FACTOR, FREE_FLOW_SPEED_KMH, RUSH_HOURS
from src.utils import haversine_km, is_rush_hour


# ═══════════════════════════════════════════════════════
#  1. EXPLAINABLE AI — Root Cause Attribution
# ═══════════════════════════════════════════════════════

def compute_xai_breakdown(df, impact_df, profiles):
    t0 = time.time()
    contrib_weights = {
        "illegal_parking": 0.40,
        "road_width": 0.21,
        "density": 0.20,
        "time_of_day": 0.12,
        "junction_proximity": 0.07,
    }
    results = []
    for _, prof in profiles.iterrows():
        cid = prof["cluster_id"]
        cdf = df[df["cluster_id"] == cid]
        n = len(cdf)
        if n == 0:
            continue

        illegal_pct = min(100, (n / max(len(df), 1)) * 100 * 5)
        avg_lanes = prof.get("num_lanes", 2) if "num_lanes" in prof.index else 2
        if "num_lanes" not in profiles.columns:
            avg_lanes = 2
        road_width_pct = max(5, 100 - avg_lanes * 25)

        density_pct = min(100, prof.get("avg_daily_rate", 1) * 3)

        rush_pct = cdf["is_rush_hour"].mean() * 100 if "is_rush_hour" in cdf.columns else 30
        time_of_day_pct = rush_pct

        junction_pct = prof.get("has_junction", 0) * 80 + 10

        raw = {
            "illegal_parking": illegal_pct,
            "road_width": road_width_pct,
            "density": density_pct,
            "time_of_day": time_of_day_pct,
            "junction_proximity": junction_pct,
        }
        total_raw = sum(raw.values()) or 1
        normalized = {k: round(v / total_raw * 100, 1) for k, v in raw.items()}

        results.append({
            "cluster_id": cid,
            "junction_name": prof["junction_name"],
            "centroid_lat": prof["centroid_lat"],
            "centroid_lon": prof["centroid_lon"],
            "breakdown": normalized,
            "dominant_factor": max(normalized, key=normalized.get),
        })
    print(f"  XAI computed in {time.time()-t0:.1f}s")
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════
#  2. CONGESTION PROPAGATION — Timeline Forecast
# ═══════════════════════════════════════════════════════

def build_adjacency(profiles, radius_km=None):
    r = radius_km or PROPAGATION_RADIUS_KM
    adj = defaultdict(list)
    prof_list = profiles.to_dict("records")
    for i, p in enumerate(prof_list):
        for j, q in enumerate(prof_list):
            if i >= j:
                continue
            d = haversine_km(p["centroid_lat"], p["centroid_lon"], q["centroid_lat"], q["centroid_lon"])
            if d <= r:
                adj[p["cluster_id"]].append({"id": q["cluster_id"], "dist": round(d, 2)})
                adj[q["cluster_id"]].append({"id": p["cluster_id"], "dist": round(d, 2)})
    return dict(adj)


def simulate_propagation(source_id, adj, profiles, start_hour=8, minutes_ahead=60):
    timeline = []
    visited = {source_id: 0}
    queue = [(source_id, 0, start_hour)]
    prof_dict = profiles.set_index("cluster_id").to_dict("index")

    while queue:
        cid, minute, hour = queue.pop(0)
        chain_entry = {
            "cluster_id": cid,
            "minute_offset": minute,
            "timestamp": f"{hour + minute // 60}:{minute % 60:02d}",
            "delay_minutes": minute,
        }
        p = prof_dict.get(cid, {})
        chain_entry["junction_name"] = p.get("junction_name", "Unknown")
        chain_entry["speed_drop_pct"] = round(
            min(80, 15 + minute * 0.8 + np.random.uniform(0, 5)), 1
        )
        timeline.append(chain_entry)

        for neighbor in adj.get(cid, []):
            nid = neighbor["id"]
            if nid in visited:
                continue
            travel_min = int(neighbor["dist"] / FREE_FLOW_SPEED_KMH * 60 * 3) + 8
            new_min = minute + travel_min
            if new_min <= minutes_ahead:
                visited[nid] = new_min
                new_hour = start_hour + (start_hour * 60 + new_min) // 60
                queue.append((nid, new_min, new_hour))

    return sorted(timeline, key=lambda x: x["minute_offset"])


def get_propagation_chains(profiles, adj, top_n=5):
    t0 = time.time()
    chains = []
    top_profiles = profiles.nlargest(top_n, "total_violations")
    for _, prof in top_profiles.iterrows():
        chain = simulate_propagation(prof["cluster_id"], adj, profiles)
        if chain:
            chains.append({
                "source_id": prof["cluster_id"],
                "source_name": prof["junction_name"],
                "chain": chain,
                "affected_count": len(chain),
            })
    print(f"  Propagation chains computed in {time.time()-t0:.1f}s")
    return chains


# ═══════════════════════════════════════════════════════
#  3. ACTION RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════

def generate_recommendations(priority_df, impact_df, risk_df):
    t0 = time.time()
    m = priority_df.merge(impact_df[["cluster_id", "worst_speed_drop_pct", "worst_hour", "total_vehicle_hours_lost"]],
                          on="cluster_id", how="left")
    if "parking_risk_index" in risk_df.columns:
        m = m.merge(risk_df[["cluster_id", "parking_risk_index"]], on="cluster_id", how="left")

    recs = []
    for _, row in m.head(20).iterrows():
        severity = row.get("worst_severity", "LOW")
        drop = row.get("worst_speed_drop_pct", 0)
        pri = row.get("priority_score", 0)
        risk = row.get("parking_risk_index", 0)
        vhl = row.get("total_vehicle_hours_lost", 0)
        peak = row.get("peak_hours", [])

        if severity == "CRITICAL":
            officers = 4
            expected_reduction = min(50, drop * 0.6)
            action = "Deploy rapid response team + barricades"
        elif severity == "HIGH":
            officers = 3
            expected_reduction = min(40, drop * 0.5)
            action = "Deploy enforcement officers + towing alert"
        elif severity == "MEDIUM":
            officers = 2
            expected_reduction = min(30, drop * 0.4)
            action = "Deploy patrol officers + warning signs"
        else:
            officers = 1
            expected_reduction = min(20, drop * 0.3)
            action = "Monitor + periodic patrol"

        peak_str = ", ".join([f"{h}:00" for h in peak[:3]]) if peak else "N/A"

        recs.append({
            "cluster_id": row["cluster_id"],
            "junction_name": row["junction_name"],
            "area": row.get("area", ""),
            "priority_score": pri,
            "parking_risk_index": risk,
            "recommended_action": action,
            "deploy_officers": officers,
            "dispatch_time": peak_str,
            "expected_delay_reduction_pct": round(expected_reduction, 1),
            "current_speed_drop_pct": drop,
            "vehicle_hours_lost": vhl,
            "urgency": severity,
        })
    recs_df = pd.DataFrame(recs).sort_values("priority_score", ascending=False).reset_index(drop=True)
    print(f"  {len(recs_df)} recommendations in {time.time()-t0:.1f}s")
    return recs_df


# ═══════════════════════════════════════════════════════
#  4. SUB-HOUR EARLY WARNING SYSTEM (Live, time-relative)
# ═══════════════════════════════════════════════════════

def compute_early_warning(df, profiles, base_time=None, prediction_horizons=None):
    t0 = time.time()
    horizons = prediction_horizons or [15, 30, 60]

    if base_time is None:
        base_time = datetime.now()
    base_hour = base_hour = base_time.hour
    base_minute = base_time.minute

    recent = df[df["hour"].isin(range(max(0, base_hour - 3), base_hour + 1))]
    warnings = []
    for _, prof in profiles.iterrows():
        cid = prof["cluster_id"]
        cdf = recent[recent["cluster_id"] == cid]
        base_rate = len(cdf) / max(3, 1)
        is_rush = int(is_rush_hour(base_hour, RUSH_HOURS))
        rush_mult = 1.5 if is_rush else 1.0
        chronic_mult = 1.3 if prof.get("is_chronic", False) else 1.0

        for h in horizons:
            future_minutes = base_minute + h
            future_hour = base_hour + future_minutes // 60
            future_minute = future_minutes % 60
            actual_future_hour = future_hour % 24

            future_rush = int(is_rush_hour(actual_future_hour, RUSH_HOURS))
            rush_factor = 1.5 if future_rush else 1.0

            time_of_day_factor = 1.0
            if 7 <= actual_future_hour <= 10:
                time_of_day_factor = 1.4
            elif 17 <= actual_future_hour <= 21:
                time_of_day_factor = 1.5
            elif 22 <= actual_future_hour or actual_future_hour <= 5:
                time_of_day_factor = 0.3

            predicted_rate = base_rate * rush_mult * chronic_mult * rush_factor * time_of_day_factor * (1 + h * 0.003)
            prob = min(99, max(1, round(predicted_rate * 12, 1)))

            if prob >= 70:
                level = "HIGH"
            elif prob >= 40:
                level = "MEDIUM"
            else:
                level = "LOW"

            warnings.append({
                "cluster_id": cid,
                "junction_name": prof["junction_name"],
                "centroid_lat": prof["centroid_lat"],
                "centroid_lon": prof["centroid_lon"],
                "horizon_minutes": h,
                "congestion_probability": prob,
                "threat_level": level,
                "base_time": base_time.strftime("%H:%M"),
                "predicts_for": f"{actual_future_hour:02d}:{future_minute:02d}",
                "rush_hour_at_forecast": bool(future_rush),
            })

    warnings_df = pd.DataFrame(warnings)
    print(f"  Early warnings computed in {time.time()-t0:.1f}s")
    return warnings_df


# ═══════════════════════════════════════════════════════
#  RUN ALL ANALYTICS
# ═══════════════════════════════════════════════════════

def run_analytics(df, profiles, impact_df, priority_df, risk_df):
    print("[Analytics] Running advanced analytics...")
    xai = compute_xai_breakdown(df, impact_df, profiles)
    adj = build_adjacency(profiles)
    chains = get_propagation_chains(profiles, adj)
    recs = generate_recommendations(priority_df, impact_df, risk_df)
    warnings = compute_early_warning(df, profiles)
    return {
        "xai": xai,
        "adjacency": adj,
        "propagation_chains": chains,
        "recommendations": recs,
        "early_warnings": warnings,
    }
