import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_prediction_features(df, cluster_profiles):
    print("[predictive] Building prediction features...")
    daily = df.groupby(["cluster_id", "date"]).agg(
        violation_count=("id", "count"),
        avg_severity=("severity_weight", "mean"),
        avg_duration=("violation_duration_minutes", "mean"),
    ).reset_index()

    daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["month"] = pd.to_datetime(daily["date"]).dt.month

    daily = daily.sort_values(["cluster_id", "date"])

    for window in [3, 7, 14]:
        daily["violations_last_{}d".format(window)] = (
            daily.groupby("cluster_id")["violation_count"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        daily["severity_last_{}d".format(window)] = (
            daily.groupby("cluster_id")["avg_severity"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily["days_since_last"] = (
        daily.groupby("cluster_id")["date_dt"]
        .transform(lambda x: x.diff().dt.days.fillna(999))
    )
    daily = daily.drop(columns=["date_dt"])

    daily["violations_next_day"] = (
        daily.groupby("cluster_id")["violation_count"].shift(-1)
    )
    daily["target"] = (daily["violations_next_day"] > daily["violation_count"].quantile(0.5)).astype(int)

    daily = daily.dropna(subset=["violations_next_day"])
    print("  Feature matrix shape: {}".format(daily.shape))
    return daily


def train_predictor(feature_df):
    print("[predictive] Training LightGBM model...")
    t0 = time.time()

    feature_cols = [
        "day_of_week", "is_weekend", "month", "violation_count",
        "avg_severity", "avg_duration",
        "violations_last_3d", "violations_last_7d", "violations_last_14d",
        "severity_last_3d", "severity_last_7d", "severity_last_14d",
        "days_since_last",
    ]

    available_cols = [c for c in feature_cols if c in feature_df.columns]
    X = feature_df[available_cols].fillna(0)
    y = feature_df["target"].fillna(0)

    from sklearn.model_selection import cross_val_score
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
    print("  Cross-validation AUC: {:.3f} (+/- {:.3f})".format(scores.mean(), scores.std()))

    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=available_cols).sort_values(ascending=False)
    print("  Top features:")
    for feat, imp in importances.head(5).items():
        print("    {}: {}".format(feat, imp))

    t1 = time.time()
    print("  Training complete in {:.1f}s".format(t1 - t0))
    return model, available_cols, scores.mean()


def predict_tomorrow(feature_df, model, feature_cols, cluster_profiles):
    print("[predictive] Predicting tomorrow's hotspots...")

    latest = feature_df.sort_values("date").groupby("cluster_id").last().reset_index()
    if len(latest) == 0:
        return pd.DataFrame()

    X_pred = latest[feature_cols].fillna(0)
    latest["activation_probability"] = model.predict_proba(X_pred)[:, 1]

    profile_cols = ["cluster_id", "label", "centroid_lat", "centroid_lon",
                    "total_violations", "unique_days", "chronic",
                    "peak_hours", "peak_day", "dominant_violation",
                    "avg_duration_minutes", "avg_severity"]
    available_profile = [c for c in profile_cols if c in cluster_profiles.columns]
    latest = latest.merge(cluster_profiles[available_profile], on="cluster_id", how="left")

    latest["estimated_violations_tomorrow"] = (
        latest["activation_probability"] * latest["violation_count"]
    ).round(0).astype(int)

    latest = latest.sort_values("activation_probability", ascending=False)

    print("  Top 10 predicted active hotspots tomorrow:")
    for _, row in latest.head(10).iterrows():
        label = row.get("label", "Cluster " + str(row["cluster_id"]))
        prob = row["activation_probability"]
        est = row["estimated_violations_tomorrow"]
        print("    {} - {:.0%} probability, ~{} violations expected".format(label, prob, est))

    return latest


def run_prediction(df, cluster_profiles):
    t0 = time.time()
    feature_df = build_prediction_features(df, cluster_profiles)
    model, feature_cols, auc = train_predictor(feature_df)
    predictions = predict_tomorrow(feature_df, model, feature_cols, cluster_profiles)
    t1 = time.time()
    print("[predictive] Total pipeline: {:.1f}s".format(t1 - t0))
    return predictions, model, feature_cols, auc
