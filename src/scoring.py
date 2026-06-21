import pandas as pd
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRIORITY_WEIGHTS, PRI_WEIGHTS, RUSH_HOURS


def _norm(s):
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)


# ── Priority Scorer ──

def compute_priority(profiles, impact_df, total_days):
    t0 = time.time()
    prof_cols = [c for c in profiles.columns if c not in impact_df.columns or c == "cluster_id"]
    m = profiles[prof_cols].merge(impact_df, on="cluster_id", how="left").fillna(0)

    freq = (m["unique_days"] / max(total_days, 1)) * 100
    impact = _norm(m["worst_speed_drop_pct"]) * 100
    cur_hour = pd.Timestamp.now().hour
    def urg(h):
        peaks = h if isinstance(h, list) else []
        for p in peaks:
            if p == cur_hour:
                return 100
            if abs(p - cur_hour) <= 2:
                return 70
            if abs(p - cur_hour) <= 4:
                return 40
        return 10
    urg_score = m["peak_hours"].apply(urg)

    road_rank = {"Ring Road": 100, "Main Road": 80, "Underpass": 70, "Cross Road": 50, "Other": 30}
    crit = m["road_type"].map(road_rank).fillna(30) + m["has_junction"] * 15
    crit = _norm(crit) * 100

    w = PRIORITY_WEIGHTS
    m["priority_score"] = (
        w["frequency"] * freq + w["impact"] * impact +
        w["urgency"] * urg_score + w["criticality"] * crit
    ).round(1)

    m["freq_score"] = freq.round(1)
    m["impact_score"] = impact.round(1)
    m["urgency_score"] = urg_score.round(1)
    m["criticality_score"] = crit.round(1)

    m = m.sort_values("priority_score", ascending=False).reset_index(drop=True)
    print(f"  Priority scored in {time.time()-t0:.1f}s")
    return m


# ── Parking Risk Index ──

def compute_parking_risk_index(profiles, impact_df):
    t0 = time.time()
    prof_cols = [c for c in profiles.columns if c not in impact_df.columns or c == "cluster_id"]
    m = profiles[prof_cols].merge(impact_df, on="cluster_id", how="left").fillna(0)

    # Illegal parking proxy: total violations normalized by max
    m["illegal_parking_norm"] = _norm(m["total_violations"])

    # Density: avg daily rate normalized
    m["density_norm"] = _norm(m["avg_daily_rate"])

    # Road importance: already computed in preprocessing, fallback if missing
    if "road_importance" in m.columns:
        m["road_importance_norm"] = _norm(m["road_importance"])
    else:
        road_imp = {"Ring Road": 1.0, "Main Road": 0.8, "Underpass": 0.7, "Cross Road": 0.5, "Other": 0.3}
        m["road_importance_norm"] = m["road_type"].map(road_imp).fillna(0.3)

    # Event score: chronic + peak hour overlap
    m["event_score_norm"] = _norm(
        m["is_chronic"].astype(float) * 0.6 + m["avg_severity"].fillna(0) * 0.4
    )

    w = PRI_WEIGHTS
    m["parking_risk_index"] = (
        w["illegal_parking"] * m["illegal_parking_norm"] * 100 +
        w["density"] * m["density_norm"] * 100 +
        w["road_importance"] * m["road_importance_norm"] * 100 +
        w["event_score"] * m["event_score_norm"] * 100
    ).clip(0, 100).round(1)

    m = m.sort_values("parking_risk_index", ascending=False).reset_index(drop=True)
    print(f"  Parking Risk Index computed in {time.time()-t0:.1f}s")
    return m


def run_scoring(profiles, impact_df, total_days):
    print("[Scoring] Computing priority + risk index...")
    priority_df = compute_priority(profiles, impact_df, total_days)
    risk_df = compute_parking_risk_index(profiles, impact_df)
    return priority_df, risk_df
