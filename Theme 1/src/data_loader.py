import pandas as pd
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RAW_CSV


def load_raw_data(csv_path=None):
    if csv_path is None:
        csv_path = RAW_CSV
    print("[data_loader] Loading CSV from:", csv_path)
    t0 = time.time()
    df = pd.read_csv(csv_path, low_memory=False)
    t1 = time.time()
    print("[data_loader] Loaded {:,} rows in {:.1f}s".format(len(df), t1 - t0))
    return df


def clean_data(df):
    print("[data_loader] Cleaning data...")
    initial = len(df)

    if "validation_status" in df.columns:
        df = df[df["validation_status"].astype(str).str.lower() == "approved"].copy()
        print("  Filtered to approved only: {:,} rows (dropped {:,})".format(len(df), initial - len(df)))

    df = df.dropna(subset=["latitude", "longitude"], how="any")
    print("  After dropping missing lat/lon: {:,} rows".format(len(df)))

    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["created_datetime"])
    print("  After dropping missing created_datetime: {:,} rows".format(len(df)))

    if "closed_datetime" in df.columns:
        df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], errors="coerce", utc=True)
        df["violation_duration_minutes"] = (
            df["closed_datetime"] - df["created_datetime"]
        ).dt.total_seconds() / 60.0
        df["violation_duration_minutes"] = df["violation_duration_minutes"].clip(lower=1, upper=1440)
    else:
        df["violation_duration_minutes"] = 30.0

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    india_mask = (df["latitude"] > 6) & (df["latitude"] < 38) & (df["longitude"] > 68) & (df["longitude"] < 98)
    df = df[india_mask].copy()
    print("  After India bounding box filter: {:,} rows".format(len(df)))

    print("[data_loader] Cleaning complete: {:,} rows retained from {:,}".format(len(df), initial))
    return df.reset_index(drop=True)


def load_and_clean(csv_path=None):
    df = load_raw_data(csv_path)
    return clean_data(df)
