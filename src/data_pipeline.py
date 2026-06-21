import pandas as pd
import numpy as np
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_CSV, RUSH_HOURS, LANE_ESTIMATE
from src.utils import classify_road_type, extract_road_name, extract_area, is_rush_hour, severity_from_violation_types


def load_raw(csv_path=None):
    path = csv_path or RAW_CSV
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def clean(df):
    t0 = time.time()
    df = df[df["validation_status"] == "approved"].copy()
    df = df.dropna(subset=["latitude", "longitude"])
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_datetime"])

    def parse_duration(r):
        if pd.notna(r.get("closed_datetime")):
            try:
                cl = pd.to_datetime(r["closed_datetime"], utc=True, errors="coerce")
                if pd.notna(cl):
                    mins = (cl - r["created_datetime"]).total_seconds() / 60
                    return max(1, min(1440, mins))
            except:
                pass
        return 30.0

    df["violation_duration_minutes"] = df.apply(parse_duration, axis=1)
    df = df[(df["latitude"] >= 6) & (df["latitude"] <= 38) &
            (df["longitude"] >= 68) & (df["longitude"] <= 98)]
    df = df.reset_index(drop=True)
    print(f"  Cleaned to {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def preprocess(df):
    t0 = time.time()
    dt = df["created_datetime"]
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].apply(lambda h: int(is_rush_hour(h, RUSH_HOURS)))
    df["time_bin"] = (df["hour"] // 2) * 2
    df["date"] = dt.dt.date

    sev = df["violation_type"].apply(severity_from_violation_types)
    df["severity_weight"] = sev.apply(lambda x: x[0])
    df["dominant_violation"] = sev.apply(lambda x: x[1])

    df["road_type"] = df["location"].apply(classify_road_type)
    df["road_name"] = df["location"].apply(extract_road_name)
    df["area"] = df["location"].apply(extract_area)
    df["num_lanes"] = df["road_type"].map(LANE_ESTIMATE).fillna(2).astype(int)
    df["has_junction"] = (df["junction_name"].fillna("No Junction") != "No Junction").astype(int)

    # Event score: proxy from violation density spikes and weekend/festival patterns
    daily = df.groupby("date").size()
    median_daily = daily.median()
    df["event_score"] = df["date"].map(lambda d: min(1.0, daily.get(d, 0) / max(median_daily, 1)))
    if df["is_weekend"].any():
        df.loc[df["is_weekend"] == 1, "event_score"] = df.loc[df["is_weekend"] == 1, "event_score"].clip(lower=0.3)

    # Road importance
    road_imp = {"Ring Road": 1.0, "Main Road": 0.8, "Underpass": 0.7, "Cross Road": 0.5, "Other": 0.3}
    df["road_importance"] = df["road_type"].map(road_imp).fillna(0.3)

    print(f"  Preprocessed in {time.time()-t0:.1f}s — {len(df.columns)} columns")
    return df


def run_pipeline(csv_path=None):
    print("[1/3] Loading raw data...")
    df = load_raw(csv_path)
    print("[2/3] Cleaning...")
    df = clean(df)
    print("[3/3] Preprocessing...")
    df = preprocess(df)
    return df
