import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_prediction_features(df, cluster_profiles):
    print("[predictive] Building prediction features...")
    daily = df.groupby(["cluster_id", "date"]).agg(
        violation_count=("id", "count"),
        avg_severity=("severity_weight", "mean"),
        avg_duration=("violation_duration_minutes", "mean"),
        unique_vehicles=("vehicle_number", "nunique"),
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

    daily["violations_lag_1d"] = daily.groupby("cluster_id")["violation_count"].shift(1)
    daily["violations_lag_2d"] = daily.groupby("cluster_id")["violation_count"].shift(2)
    daily["violations_lag_7d"] = daily.groupby("cluster_id")["violation_count"].shift(7)

    daily["trend_3d"] = daily["violations_lag_1d"] - daily["violations_last_3d"]
    daily["trend_7d"] = daily["violations_lag_1d"] - daily["violations_last_7d"]
    daily["trend_direction"] = np.sign(daily["trend_3d"].fillna(0))

    daily["violations_std_7d"] = (
        daily.groupby("cluster_id")["violation_count"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std())
    )
    daily["violations_std_14d"] = (
        daily.groupby("cluster_id")["violation_count"]
        .transform(lambda x: x.shift(1).rolling(14, min_periods=1).std())
    )

    daily["weekend_x_count"] = daily["is_weekend"] * daily["violation_count"]
    daily["severity_x_duration"] = daily["avg_severity"] * daily["avg_duration"]

    profile_merge = cluster_profiles[["cluster_id", "chronic", "total_violations", "unique_days"]].copy()
    profile_merge["avg_daily_rate"] = profile_merge["total_violations"] / profile_merge["unique_days"].clip(lower=1)
    profile_merge["is_chronic"] = profile_merge["chronic"].astype(int)
    daily = daily.merge(profile_merge, on="cluster_id", how="left", suffixes=("", "_profile"))

    daily["violations_next_day"] = (
        daily.groupby("cluster_id")["violation_count"].shift(-1)
    )
    daily = daily.dropna(subset=["violations_next_day"])

    numeric_cols = daily.select_dtypes(include=[np.number]).columns
    daily[numeric_cols] = daily[numeric_cols].fillna(0)

    print("  Feature matrix shape: {}".format(daily.shape))
    return daily


def get_feature_cols():
    return [
        "day_of_week", "is_weekend", "month", "violation_count",
        "avg_severity", "avg_duration", "unique_vehicles",
        "violations_last_3d", "violations_last_7d", "violations_last_14d",
        "severity_last_3d", "severity_last_7d", "severity_last_14d",
        "days_since_last",
        "violations_lag_1d", "violations_lag_2d", "violations_lag_7d",
        "trend_3d", "trend_7d", "trend_direction",
        "violations_std_7d", "violations_std_14d",
        "weekend_x_count", "severity_x_duration",
        "is_chronic", "total_violations", "unique_days", "avg_daily_rate",
    ]


def optimize_lightgbm(X, y, tscv, n_trials=30):
    print("[predictive] Running Optuna search for LightGBM ({} trials)...".format(n_trials))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        scores = cross_val_score(model, X, y, cv=tscv, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print("  Best LightGBM R2: {:.3f}".format(study.best_value))
    return study.best_params


def optimize_xgboost(X, y, tscv, n_trials=30):
    print("[predictive] Running Optuna search for XGBoost ({} trials)...".format(n_trials))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5.0),
        }
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=tscv, scoring="r2", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print("  Best XGBoost R2: {:.3f}".format(study.best_value))
    return study.best_params


def train_ensemble(feature_df, feature_cols, n_trials=30):
    print("[predictive] Training ensemble model...")
    t0 = time.time()

    X = feature_df[feature_cols].values
    y = feature_df["violations_next_day"].values

    tscv = TimeSeriesSplit(n_splits=5)

    lgb_params = optimize_lightgbm(X, y, tscv, n_trials)
    xgb_params = optimize_xgboost(X, y, tscv, n_trials)

    lgb_model = lgb.LGBMRegressor(**lgb_params, random_state=42, verbose=-1)
    xgb_model = xgb.XGBRegressor(**xgb_params, random_state=42, verbosity=0, n_jobs=-1)

    lgb_scores = cross_val_score(lgb_model, X, y, cv=tscv, scoring="r2", n_jobs=-1)
    xgb_scores = cross_val_score(xgb_model, X, y, cv=tscv, scoring="r2", n_jobs=-1)

    lgb_r2 = max(lgb_scores.mean(), 0.01)
    xgb_r2 = max(xgb_scores.mean(), 0.01)

    w_lgb = lgb_r2 / (lgb_r2 + xgb_r2)
    w_xgb = xgb_r2 / (lgb_r2 + xgb_r2)

    print("  LightGBM R2: {:.3f}, XGBoost R2: {:.3f}".format(lgb_r2, xgb_r2))
    print("  Ensemble weights: LightGBM={:.2f}, XGBoost={:.2f}".format(w_lgb, w_xgb))

    lgb_model.fit(X, y)
    xgb_model.fit(X, y)

    importances = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("  Top 10 features (LightGBM):")
    for feat, imp in importances.head(10).items():
        print("    {}: {}".format(feat, imp))

    t1 = time.time()
    print("  Training complete in {:.1f}s".format(t1 - t0))

    return lgb_model, xgb_model, w_lgb, w_xgb, feature_cols, lgb_r2


def predict_tomorrow(feature_df, lgb_model, xgb_model, w_lgb, w_xgb, feature_cols, cluster_profiles):
    print("[predictive] Predicting tomorrow's hotspots...")

    latest = feature_df.sort_values("date").groupby("cluster_id").last().reset_index()
    if len(latest) == 0:
        return pd.DataFrame()

    X_pred = latest[feature_cols].values
    lgb_pred = lgb_model.predict(X_pred)
    xgb_pred = xgb_model.predict(X_pred)
    blended_pred = w_lgb * lgb_pred + w_xgb * xgb_pred
    blended_pred = np.maximum(blended_pred, 0)

    latest["predicted_violation_count"] = blended_pred.round(0).astype(int)

    profile_cols = ["cluster_id", "label", "centroid_lat", "centroid_lon",
                    "total_violations", "unique_days", "chronic",
                    "peak_hours", "peak_day", "dominant_violation",
                    "avg_duration_minutes", "avg_severity"]
    available_profile = [c for c in profile_cols if c in cluster_profiles.columns]
    latest = latest.merge(cluster_profiles[available_profile], on="cluster_id", how="left")

    if "total_violations_profile" in latest.columns:
        latest["total_violations"] = latest["total_violations_profile"]
        latest = latest.drop(columns=["total_violations_profile"], errors="ignore")
    if "unique_days_profile" in latest.columns:
        latest["unique_days"] = latest["unique_days_profile"]
        latest = latest.drop(columns=["unique_days_profile"], errors="ignore")

    cluster_median = feature_df.groupby("cluster_id")["violation_count"].median()
    latest["cluster_median"] = latest["cluster_id"].map(cluster_median).fillna(1)
    ratio = latest["predicted_violation_count"] / latest["cluster_median"].clip(lower=1)
    latest["activation_probability"] = np.where(
        ratio > 1,
        np.minimum(0.5 + 0.5 * np.log1p(ratio - 1) / 3, 0.99),
        np.maximum(0.1 * ratio, 0.05)
    )
    latest["activation_probability"] = latest["activation_probability"].clip(0, 1)

    latest["estimated_violations_tomorrow"] = latest["predicted_violation_count"]
    latest = latest.sort_values("predicted_violation_count", ascending=False)

    print("  Top 10 predicted active hotspots tomorrow:")
    for _, row in latest.head(10).iterrows():
        label = row.get("label", "Cluster " + str(row["cluster_id"]))
        pred = row["predicted_violation_count"]
        prob = row["activation_probability"]
        print("    {} - ~{} violations (activation: {:.0%})".format(label, pred, prob))

    return latest


def run_prediction(df, cluster_profiles, n_trials=30):
    t0 = time.time()
    feature_df = build_prediction_features(df, cluster_profiles)
    feature_cols = get_feature_cols()
    available_cols = [c for c in feature_cols if c in feature_df.columns]
    lgb_model, xgb_model, w_lgb, w_xgb, final_cols, r2 = train_ensemble(
        feature_df, available_cols, n_trials
    )
    predictions = predict_tomorrow(feature_df, lgb_model, xgb_model, w_lgb, w_xgb, final_cols, cluster_profiles)
    t1 = time.time()
    print("[predictive] Total pipeline: {:.1f}s".format(t1 - t0))
    return predictions, lgb_model, final_cols, r2
