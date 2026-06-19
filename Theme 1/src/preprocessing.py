import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RUSH_HOURS, LANE_ESTIMATE
from src.utils import classify_road_type, extract_road_name, extract_area, is_rush_hour, severity_from_violation_types


def add_temporal_features(df):
    print("[preprocessing] Adding temporal features...")
    dt = df["created_datetime"]
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].apply(lambda h: is_rush_hour(h, RUSH_HOURS)).astype(int)
    df["time_bin"] = (df["hour"] // 2) * 2
    df["date"] = dt.dt.date
    df["year_month"] = dt.dt.to_period("M").astype(str)
    print("  Added: hour, day_of_week, month, is_weekend, is_rush_hour, time_bin, date")
    return df


def add_severity_features(df):
    print("[preprocessing] Adding severity features...")
    results = df["violation_type"].apply(severity_from_violation_types)
    df["severity_weight"] = results.apply(lambda x: x[0])
    df["dominant_violation"] = results.apply(lambda x: x[1])
    print("  Severity distribution:")
    for vtype, count in df["dominant_violation"].value_counts().head(10).items():
        print("    {}: {:,}".format(vtype, count))
    return df


def add_road_features(df):
    print("[preprocessing] Adding road features...")
    df["road_type"] = df["location"].apply(classify_road_type)
    df["road_name"] = df["location"].apply(extract_road_name)
    df["area"] = df["location"].apply(extract_area)
    df["num_lanes"] = df["road_type"].map(LANE_ESTIMATE).fillna(2).astype(int)
    print("  Road type distribution:")
    for rtype, count in df["road_type"].value_counts().items():
        print("    {}: {:,}".format(rtype, count))
    return df


def add_junction_features(df):
    print("[preprocessing] Adding junction features...")
    if "junction_name" in df.columns:
        df["has_junction"] = df["junction_name"].notna() & (df["junction_name"] != "No Junction") & (df["junction_name"] != "")
        df["has_junction"] = df["has_junction"].astype(int)
    else:
        df["has_junction"] = 0
    junction_count = df["has_junction"].sum()
    print("  Records with junction data: {:,} ({:.1f}%)".format(junction_count, 100 * junction_count / len(df)))
    return df


def preprocess(df):
    t0 = time.time()
    df = add_temporal_features(df)
    df = add_severity_features(df)
    df = add_road_features(df)
    df = add_junction_features(df)
    t1 = time.time()
    print("[preprocessing] Complete in {:.1f}s. Shape: {}".format(t1 - t0, df.shape))
    return df
