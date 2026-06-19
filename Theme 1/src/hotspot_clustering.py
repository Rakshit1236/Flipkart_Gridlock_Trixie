import pandas as pd
import numpy as np
import hdbscan
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, CHRONIC_THRESHOLD_PCT


def cluster_hotspots(df, min_cluster_size=None, min_samples=None):
    if min_cluster_size is None:
        min_cluster_size = HDBSCAN_MIN_CLUSTER_SIZE
    if min_samples is None:
        min_samples = HDBSCAN_MIN_SAMPLES

    print("[clustering] Running HDBSCAN on {:,} points...".format(len(df)))
    print("  Parameters: min_cluster_size={}, min_samples={}".format(min_cluster_size, min_samples))
    t0 = time.time()

    coords = df[["latitude", "longitude"]].values
    coords_rad = np.radians(coords)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="haversine",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    df = df.copy()
    df["cluster_id"] = clusterer.fit_predict(coords_rad)

    n_clusters = len(set(df["cluster_id"])) - (1 if -1 in df["cluster_id"].values else 0)
    n_noise = (df["cluster_id"] == -1).sum()
    t1 = time.time()
    print("  Found {} clusters, {:,} noise points in {:.1f}s".format(n_clusters, n_noise, t1 - t0))
    return df, clusterer


def build_cluster_profiles(df):
    print("[clustering] Building cluster profiles...")
    total_days = df["date"].nunique()
    profiles = []

    day_names = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                 4: "Friday", 5: "Saturday", 6: "Sunday"}

    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            continue
        cluster_df = df[df["cluster_id"] == cid]

        centroid_lat = cluster_df["latitude"].median()
        centroid_lon = cluster_df["longitude"].median()

        junction_votes = cluster_df["junction_name"].dropna()
        junction_votes = junction_votes[junction_votes != "No Junction"]
        if len(junction_votes) > 0:
            label = junction_votes.value_counts().index[0]
        else:
            road_votes = cluster_df["road_name"].value_counts()
            label = road_votes.index[0] if len(road_votes) > 0 else "Cluster {}".format(cid)

        hour_dist = cluster_df["hour"].value_counts().sort_index()
        peak_hours = hour_dist.nlargest(3).index.tolist()

        day_dist = cluster_df["day_of_week"].value_counts().sort_index()
        peak_day = day_dist.idxmax() if len(day_dist) > 0 else 0
        peak_day_name = day_names.get(peak_day, "Unknown")

        violation_type_dist = cluster_df["dominant_violation"].value_counts()
        dominant_vtype = violation_type_dist.index[0] if len(violation_type_dist) > 0 else "UNKNOWN"

        unique_days = cluster_df["date"].nunique()
        chronic = unique_days / max(total_days, 1) >= CHRONIC_THRESHOLD_PCT

        profiles.append({
            "cluster_id": cid,
            "label": label,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "total_violations": len(cluster_df),
            "unique_days": unique_days,
            "chronic": chronic,
            "peak_hours": peak_hours,
            "peak_day": peak_day_name,
            "dominant_violation": dominant_vtype,
            "avg_duration_minutes": cluster_df["violation_duration_minutes"].mean(),
            "avg_severity": cluster_df["severity_weight"].mean(),
            "hour_distribution": hour_dist.to_dict(),
            "day_distribution": day_dist.to_dict(),
        })

    profiles_df = pd.DataFrame(profiles)
    profiles_df = profiles_df.sort_values("total_violations", ascending=False).reset_index(drop=True)
    print("  Built {} cluster profiles".format(len(profiles_df)))
    print("  Top 5 hotspots by violation count:")
    for _, row in profiles_df.head(5).iterrows():
        print("    {}: {} violations, peak={}, chronic={}".format(
            row["label"], row["total_violations"], row["peak_hours"], row["chronic"]))
    return profiles_df


def get_cluster_timeseries(df):
    print("[clustering] Building cluster-time matrix...")
    cluster_hours = df.groupby(["cluster_id", "hour"]).size().reset_index(name="count")
    pivot = cluster_hours.pivot_table(index="cluster_id", columns="hour", values="count", fill_value=0)
    print("  Time-series matrix shape: {}".format(pivot.shape))
    return pivot


def run_clustering(df, min_cluster_size=None, min_samples=None):
    t0 = time.time()
    df_clustered, model = cluster_hotspots(df, min_cluster_size, min_samples)
    profiles = build_cluster_profiles(df_clustered)
    ts_matrix = get_cluster_timeseries(df_clustered)
    t1 = time.time()
    print("[clustering] Total pipeline: {:.1f}s".format(t1 - t0))
    return df_clustered, profiles, ts_matrix, model
