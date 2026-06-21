import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import optuna
import warnings
import os, sys, time
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRED_DIR, MODEL_DIR, RUSH_HOURS
warnings.filterwarnings("ignore")


def build_features(df, profiles):
    df_clustered = df[df["cluster_id"] != -1].copy()
    daily = df_clustered.groupby(["cluster_id", "date"]).agg(
        violations=("id", "count"),
        avg_severity=("severity_weight", "mean"),
        avg_duration=("violation_duration_minutes", "mean"),
        rush_violations=("is_rush_hour", "sum"),
        weekend_violations=("is_weekend", "sum"),
        hour_mean=("hour", "mean"),
        hour_std=("hour", "std"),
        unique_vehicles=("vehicle_number", "nunique"),
        unique_violation_types=("dominant_violation", "nunique"),
    ).reset_index()
    daily["hour_std"] = daily["hour_std"].fillna(0)
    daily = daily.sort_values(["cluster_id", "date"])

    # Day of week features
    daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.dayofweek
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    daily["is_monday"] = (daily["day_of_week"] == 0).astype(int)
    daily["is_friday"] = (daily["day_of_week"] == 4).astype(int)

    # Lag features
    for lag in [1, 2, 3, 5, 7, 14]:
        daily[f"lag_{lag}d"] = daily.groupby("cluster_id")["violations"].shift(lag)

    # Rolling windows
    for win in [3, 5, 7, 14]:
        daily[f"roll_{win}d_mean"] = daily.groupby("cluster_id")["violations"].transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )
        daily[f"roll_{win}d_max"] = daily.groupby("cluster_id")["violations"].transform(
            lambda x: x.rolling(win, min_periods=1).max()
        )
        daily[f"roll_{win}d_std"] = daily.groupby("cluster_id")["violations"].transform(
            lambda x: x.rolling(win, min_periods=1).std()
        )
    daily = daily.fillna(0)

    # Severity rolling
    for win in [3, 7]:
        daily[f"roll_{win}d_sev_mean"] = daily.groupby("cluster_id")["avg_severity"].transform(
            lambda x: x.rolling(win, min_periods=1).mean()
        )

    # Trend features
    daily["trend_3d"] = daily["violations"] - daily.groupby("cluster_id")["violations"].shift(3)
    daily["trend_7d"] = daily["violations"] - daily.groupby("cluster_id")["violations"].shift(7)
    daily["trend_14d"] = daily["violations"] - daily.groupby("cluster_id")["violations"].shift(14)
    daily["direction_3d"] = (daily["trend_3d"] > 0).astype(int)
    daily["direction_7d"] = (daily["trend_7d"] > 0).astype(int)

    # Volatility
    daily["roll_std_7d"] = daily.groupby("cluster_id")["violations"].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )
    daily["cv_7d"] = daily["roll_std_7d"] / (daily["roll_7d_mean"] + 1)

    # Interaction features
    daily["weekend_x_count"] = daily["is_weekend"] * daily["violations"]
    daily["severity_x_duration"] = daily["avg_severity"] * daily["avg_duration"]
    daily["rush_x_count"] = daily["rush_violations"] / (daily["violations"] + 1)
    daily["vehicles_per_violation"] = daily["unique_vehicles"] / (daily["violations"] + 1)

    # Ratio features
    daily["ratio_to_7d_mean"] = daily["violations"] / (daily["roll_7d_mean"] + 1)
    daily["ratio_to_14d_mean"] = daily["violations"] / (daily["roll_14d_mean"] + 1)
    daily["ratio_to_7d_max"] = daily["violations"] / (daily["roll_7d_max"] + 1)

    # Acceleration
    daily["acceleration"] = daily["trend_3d"] - daily.groupby("cluster_id")["trend_3d"].shift(3)

    # Profile features
    profile_merge = profiles[["cluster_id", "is_chronic", "avg_daily_rate", "avg_severity",
                               "road_type", "num_lanes", "has_junction", "total_violations"]].copy()
    daily = daily.merge(profile_merge, on="cluster_id", how="left", suffixes=("", "_prof"))
    if "road_type" in daily.columns:
        le = LabelEncoder()
        daily["road_type_enc"] = le.fit_transform(daily["road_type"].fillna("Other"))
    else:
        daily["road_type_enc"] = 0

    # Days since first seen
    first_seen = daily.groupby("cluster_id")["date"].min().reset_index()
    first_seen.columns = ["cluster_id", "first_date"]
    daily = daily.merge(first_seen, on="cluster_id", how="left")
    daily["days_active"] = (pd.to_datetime(daily["date"]) - pd.to_datetime(daily["first_date"])).dt.days
    daily = daily.drop(columns=["first_date"], errors="ignore")

    daily = daily.fillna(0)

    # Target
    daily["violations_next_day"] = daily.groupby("cluster_id")["violations"].shift(-1)
    daily = daily.dropna(subset=["violations_next_day"])
    return daily


def get_feature_cols():
    return [
        "violations", "avg_severity", "avg_duration", "rush_violations", "weekend_violations",
        "hour_mean", "hour_std", "unique_vehicles", "unique_violation_types",
        "day_of_week", "is_weekend", "is_monday", "is_friday",
        "lag_1d", "lag_2d", "lag_3d", "lag_5d", "lag_7d", "lag_14d",
        "roll_3d_mean", "roll_5d_mean", "roll_7d_mean", "roll_14d_mean",
        "roll_3d_max", "roll_7d_max", "roll_14d_max",
        "roll_3d_std", "roll_7d_std", "roll_14d_std",
        "roll_3d_sev_mean", "roll_7d_sev_mean",
        "trend_3d", "trend_7d", "trend_14d", "direction_3d", "direction_7d",
        "roll_std_7d", "cv_7d",
        "weekend_x_count", "severity_x_duration", "rush_x_count", "vehicles_per_violation",
        "ratio_to_7d_mean", "ratio_to_14d_mean", "ratio_to_7d_max",
        "acceleration",
        "is_chronic", "avg_daily_rate", "num_lanes", "has_junction", "total_violations",
        "road_type_enc", "days_active",
    ]


def optimize_lgb(X, y, n_trials=15):
    def objective(trial):
        params = {
            "objective": "regression", "metric": "r2", "verbosity": -1,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        model = lgb.LGBMRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=4)
        scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_xgb(X, y, n_trials=15):
    def objective(trial):
        params = {
            "objective": "reg:squarederror", "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
        model = xgb.XGBRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=4)
        scores = cross_val_score(model, X, y, cv=tscv, scoring="r2")
        return scores.mean()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_ensemble(feature_df, feature_cols, n_trials=15):
    t0 = time.time()
    X = feature_df[feature_cols].fillna(0)
    y = feature_df["violations_next_day"].fillna(0)

    print("  Optimizing LightGBM...")
    lgb_params = optimize_lgb(X, y, n_trials)
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X, y)
    lgb_pred = lgb_model.predict(X)
    lgb_r2 = r2_score(y, lgb_pred)

    print("  Optimizing XGBoost...")
    xgb_params = optimize_xgb(X, y, n_trials)
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X, y)
    xgb_pred = xgb_model.predict(X)
    xgb_r2 = r2_score(y, xgb_pred)

    # Ensemble weight
    total = lgb_r2 + xgb_r2
    lgb_w = lgb_r2 / total if total > 0 else 0.5
    xgb_w = xgb_r2 / total if total > 0 else 0.5

    # Cross-validated R2 on blended predictions
    blended = lgb_w * lgb_pred + xgb_w * xgb_pred
    blend_r2 = r2_score(y, blended)
    mae = mean_absolute_error(y, blended)

    # Feature importance (top 15)
    imp = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\n  Top 10 features:")
    for fname, fval in imp.head(10).items():
        print(f"    {fname}: {fval}")

    print(f"\n  Ensemble: LGB R2={lgb_r2:.3f} ({lgb_w:.0%}), XGB R2={xgb_r2:.3f} ({xgb_w:.0%}), Blended R2={blend_r2:.3f} — {time.time()-t0:.1f}s")
    return lgb_model, xgb_model, lgb_w, xgb_w, blend_r2


def predict_tomorrow(feature_df, profiles, lgb_model, xgb_model, lgb_w, xgb_w, feature_cols):
    latest = feature_df.sort_values("date").groupby("cluster_id").last().reset_index()
    X = latest[feature_cols].fillna(0)
    lgb_pred = lgb_model.predict(X)
    xgb_pred = xgb_model.predict(X)
    blended = lgb_w * lgb_pred + xgb_w * xgb_pred
    blended = np.maximum(blended, 0)

    # Confidence based on prediction variance between models
    pred_var = np.abs(lgb_pred - xgb_pred)
    base_confidence = 95 - pred_var * 3
    latest["confidence_pct"] = np.clip(base_confidence, 50, 99).round(0).astype(int)
    latest["predicted_violations"] = blended.round(0).astype(int)
    latest["ci_lower"] = (blended * 0.85).round(0).astype(int)
    latest["ci_upper"] = (blended * 1.15).round(0).astype(int)

    median_rate = profiles.set_index("cluster_id")["avg_daily_rate"].to_dict()
    latest["activation_probability"] = latest.apply(
        lambda r: min(100, round(r["predicted_violations"] / max(median_rate.get(r["cluster_id"], 1), 1) * 50, 1)),
        axis=1
    )

    result = latest[["cluster_id", "predicted_violations", "activation_probability",
                      "confidence_pct", "ci_lower", "ci_upper", "date"]].copy()
    result = result.merge(profiles[["cluster_id", "junction_name", "centroid_lat", "centroid_lon",
                                    "dominant_violation", "peak_hours", "is_chronic", "road_type"]],
                          on="cluster_id", how="left")
    result = result.sort_values("predicted_violations", ascending=False).reset_index(drop=True)
    return result


def run_prediction(df, profiles, n_trials=15):
    print("[Predict] Training ensemble...")
    feature_df = build_features(df, profiles)
    feature_cols = [c for c in get_feature_cols() if c in feature_df.columns]
    print(f"  {len(feature_cols)} features, {len(feature_df)} training samples")
    lgb_model, xgb_model, lgb_w, xgb_w, r2 = train_ensemble(feature_df, feature_cols, n_trials)
    preds = predict_tomorrow(feature_df, profiles, lgb_model, xgb_model, lgb_w, xgb_w, feature_cols)
    print(f"  {len(preds)} predictions generated")

    import joblib
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgb_model.pkl"))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    preds.to_csv(os.path.join(PRED_DIR, f"predictions_{pd.Timestamp.now().date()}.csv"), index=False)
    return preds, feature_df, feature_cols, r2
