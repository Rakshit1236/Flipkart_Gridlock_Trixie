import pandas as pd
import numpy as np
import hdbscan
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, CHRONIC_THRESHOLD_PCT


def cluster_hotspots(df, min_cluster_size=None, min_samples=None):
    t0 = time.time()
    mcs = min_cluster_size or HDBSCAN_MIN_CLUSTER_SIZE
    ms = min_samples or HDBSCAN_MIN_SAMPLES

    coords = np.radians(df[["latitude", "longitude"]].values).astype(np.float32)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs, min_samples=ms,
        metric="haversine", cluster_selection_method="eom",
        core_dist_n_jobs=-1, algorithm="best",
        cluster_selection_epsilon=0.0,
    )
    df = df.copy()
    df["cluster_id"] = clusterer.fit_predict(coords)
    n_clusters = df["cluster_id"].nunique() - (1 if -1 in df["cluster_id"].values else 0)
    print(f"  {n_clusters} clusters in {time.time()-t0:.1f}s")
    return df, clusterer


def build_profiles(df):
    t0 = time.time()
    total_days = df["date"].nunique()
    df_valid = df[df["cluster_id"] != -1]

    profiles = []
    for cid, grp in df_valid.groupby("cluster_id"):
        lat, lon = grp["latitude"].median(), grp["longitude"].median()
        junc = grp["junction_name"].fillna("No Junction")
        junc_name = junc[junc != "No Junction"].mode()
        junc_name = junc_name.iloc[0] if len(junc_name) > 0 else grp["road_name"].mode().iloc[0] if len(grp["road_name"].mode()) > 0 else "Unknown"
        unique_days = grp["date"].nunique()
        peak_hours = grp.groupby("hour").size().nlargest(3).index.tolist()
        peak_day = grp.groupby("day_of_week").size().idxmax()
        dom_violation = grp["dominant_violation"].mode().iloc[0] if len(grp["dominant_violation"].mode()) > 0 else "Unknown"
        is_chronic = (unique_days / total_days) >= CHRONIC_THRESHOLD_PCT

        profiles.append({
            "cluster_id": cid, "centroid_lat": lat, "centroid_lon": lon,
            "junction_name": junc_name, "area": grp["area"].mode().iloc[0] if len(grp["area"].mode()) > 0 else "Unknown",
            "road_type": grp["road_type"].mode().iloc[0] if len(grp["road_type"].mode()) > 0 else "Other",
            "has_junction": int(grp["has_junction"].mean() > 0.5),
            "num_lanes": grp["num_lanes"].mode().iloc[0] if "num_lanes" in grp.columns else 2,
            "total_violations": len(grp), "unique_days": unique_days,
            "peak_hours": peak_hours, "peak_day": int(peak_day),
            "dominant_violation": dom_violation, "is_chronic": is_chronic,
            "avg_duration": grp["violation_duration_minutes"].mean(),
            "avg_severity": grp["severity_weight"].mean(),
            "avg_daily_rate": unique_days / total_days * len(grp) if total_days > 0 else 0,
        })

    profiles_df = pd.DataFrame(profiles)
    print(f"  {len(profiles_df)} profiles in {time.time()-t0:.1f}s")
    return profiles_df


def get_timeseries(df):
    ts = df[df["cluster_id"] != -1].groupby(["cluster_id", "hour"]).size().unstack(fill_value=0)
    return ts


def run_clustering(df):
    print("[Clustering] Running HDBSCAN...")
    df, model = cluster_hotspots(df)
    profiles = build_profiles(df)
    ts = get_timeseries(df)
    return df, profiles, ts, model
