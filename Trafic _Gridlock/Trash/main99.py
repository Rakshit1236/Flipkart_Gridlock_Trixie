import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import category_encoders as ce
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Load Data
# ==========================================
print("Loading datasets...")
train = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')

all_data_temp = pd.concat([train.drop(columns=['demand']), test], axis=0)

# ==========================================
# 2. Grandmaster Feature Engineering
# ==========================================
def feature_engineering(df):
    data = df.copy()

    # A. Temporal Features (Fixed format)
    data['hour'] = data['timestamp'].astype(str).str.split(':').str[0].astype(int)
    data['minute'] = data['timestamp'].astype(str).str.split(':').str[1].astype(int)

    # NEW: Cyclical Time Encoding (Tells the model that 23:00 is next to 00:00)
    data['sin_hour'] = np.sin(2 * np.pi * data['hour'] / 24.0)
    data['cos_hour'] = np.cos(2 * np.pi * data['hour'] / 24.0)

    # B. Spatial Features (Going deeper to level 5)
    # Handle both 'geohash' and 'geohash6' column names
    geohash_col = 'geohash' if 'geohash' in data.columns else 'geohash6'
    data['geohash_3'] = data[geohash_col].astype(str).str[:3]
    data['geohash_4'] = data[geohash_col].astype(str).str[:4]
    data['geohash_5'] = data[geohash_col].astype(str).str[:5] # NEW

    # C. Complex Spatio-Temporal Interactions
    data['geo4_hour'] = data['geohash_4'] + '_' + data['hour'].astype(str)
    
    if 'RoadType' in data.columns:
        data['road_hour'] = data['RoadType'].astype(str) + '_' + data['hour'].astype(str)
    
    if 'Weather' in data.columns and 'RoadType' in data.columns:
        data['weather_road'] = data['Weather'].astype(str) + '_' + data['RoadType'].astype(str) # NEW

    # D. Frequency/Density Encoding
    data['geohash_density'] = data[geohash_col].map(all_data_temp[geohash_col].value_counts())

    # E. Clean Missing Values
    if 'Temperature' in data.columns:
        data['Temperature'] = data['Temperature'].fillna(data['Temperature'].median())
    if 'NumberofLanes' in data.columns:
        data['NumberofLanes'] = data['NumberofLanes'].fillna(data['NumberofLanes'].mode()[0])

    data = data.drop(columns=['timestamp'])

    cat_cols = data.select_dtypes(include=['object']).columns
    for col in cat_cols:
        data[col] = data[col].astype(str).fillna('Unknown')

    return data

print("Engineering micro-features...")
train_df = feature_engineering(train)
test_df = feature_engineering(test)

# Drop only columns that exist
train_cols_to_drop = [col for col in ['Index', 'demand'] if col in train_df.columns]
test_cols_to_drop = [col for col in ['Index'] if col in test_df.columns]

X = train_df.drop(columns=train_cols_to_drop)
y = np.log1p(train_df['demand'])
X_test = test_df.drop(columns=test_cols_to_drop)

# Encoding grouping - only include columns that exist
base_target_encode_cols = ['geohash_5', 'geo4_hour']
optional_cols = ['road_hour', 'weather_road', 'Weather']
target_encode_cols = base_target_encode_cols + [col for col in optional_cols if col in X.columns]

cat_features = [col for col in X.columns if X[col].dtype == 'object' and col not in target_encode_cols]
all_cat_cols = target_encode_cols + cat_features

# Align X and X_test to have same columns (drop extra columns from test, add missing with NaN to train)
common_cols = set(X.columns) & set(X_test.columns)
X = X[list(common_cols)].copy()
X_test = X_test[list(common_cols)].copy()

# Re-calculate cat_cols after alignment - all categorical columns should be encoded
all_cat_cols = [col for col in X.columns if X[col].dtype == 'object']
# For target encoder with CB, use a subset
target_encode_cols = [col for col in all_cat_cols if col in ['geo4_hour']]

# ==========================================
# 3. Model Training & Out-Of-Fold Collection
# ==========================================
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Matrices to store Out-Of-Fold (OOF) predictions for the Optimizer
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cb = np.zeros(len(train_df))

cb_test_preds = np.zeros(len(X_test))
lgb_test_preds = np.zeros(len(X_test))
xgb_test_preds = np.zeros(len(X_test))

print(f"\nStarting {N_SPLITS}-Fold Training for Meta-Optimization...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test_fold = X_test.copy()

    # Target Encoding (Higher smoothing to prevent overfitting on new features)
    lgb_xgb_encoder = ce.TargetEncoder(cols=all_cat_cols, smoothing=30)
    cb_encoder = ce.TargetEncoder(cols=target_encode_cols, smoothing=30)

    X_train_num = lgb_xgb_encoder.fit_transform(X_train, y_train)
    X_val_num = lgb_xgb_encoder.transform(X_val)
    X_test_num = lgb_xgb_encoder.transform(X_test_fold)

    X_train_cb = cb_encoder.fit_transform(X_train.copy(), y_train)
    X_val_cb = cb_encoder.transform(X_val.copy())
    X_test_cb = cb_encoder.transform(X_test_fold.copy())

    # -- LightGBM --
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.02, num_leaves=63, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train_num, y_train, eval_set=[(X_val_num, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[val_idx] = lgb_model.predict(X_val_num)
    lgb_test_preds += lgb_model.predict(X_test_num) / N_SPLITS

    # -- XGBoost --
    xgb_model = xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.02, max_depth=7,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        tree_method='hist', early_stopping_rounds=50
    )
    xgb_model.fit(X_train_num, y_train, eval_set=[(X_val_num, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val_num)
    xgb_test_preds += xgb_model.predict(X_test_num) / N_SPLITS

    # -- CatBoost --
    cb_model = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=7,
        random_seed=42, od_type='Iter', od_wait=50, verbose=0
    )
    cb_model.fit(X_train_cb, y_train, cat_features=cat_features, eval_set=(X_val_cb, y_val), use_best_model=True)
    oof_cb[val_idx] = cb_model.predict(X_val_cb)
    cb_test_preds += cb_model.predict(X_test_cb) / N_SPLITS

    print(f"Fold {fold+1} Models Trained.")

# ==========================================
# 4. SciPy Weight Optimization (The Magic)
# ==========================================
print("\nMathematically Optimizing Ensemble Weights...")

# Convert predictions back to original scale for accurate R2 scoring
true_target = np.expm1(y)
pred_lgb = np.expm1(oof_lgb)
pred_xgb = np.expm1(oof_xgb)
pred_cb = np.expm1(oof_cb)

# We want to MINIMIZE the negative R2 score
def objective_func(weights):
    blended_pred = (weights[0] * pred_lgb) + (weights[1] * pred_xgb) + (weights[2] * pred_cb)
    # Penalize negative predictions
    blended_pred = np.clip(blended_pred, 0, None)
    score = r2_score(true_target, blended_pred)
    return -score

# Initial guess (even split)
init_weights = [0.33, 0.33, 0.34]

# Constraints: Weights must sum to 1. Bounds: Weights must be between 0 and 1.
cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
bounds = [(0, 1), (0, 1), (0, 1)]

# Run the optimizer
opt_res = minimize(objective_func, init_weights, method='SLSQP', bounds=bounds, constraints=cons)
best_w = opt_res.x

print(f"Optimal Weights Found:")
print(f"  LightGBM: {best_w[0]:.4f}")
print(f"  XGBoost:  {best_w[1]:.4f}")
print(f"  CatBoost: {best_w[2]:.4f}")

# Calculate final OOF Score using perfect weights
final_oof_pred = (best_w[0] * pred_lgb) + (best_w[1] * pred_xgb) + (best_w[2] * pred_cb)
final_r2 = r2_score(true_target, np.clip(final_oof_pred, 0, None))
final_metric = max(0, 100 * final_r2)

print(f"\n---> FINAL VALIDATION METRIC SCORE: {final_metric:.4f} <---")

# ==========================================
# 5. Final Submission Output
# ==========================================
# Apply the exact mathematically optimal weights to the test predictions
final_test_preds = np.expm1(
    (best_w[0] * lgb_test_preds) +
    (best_w[1] * xgb_test_preds) +
    (best_w[2] * cb_test_preds)
)

final_test_preds = np.clip(final_test_preds, 0, None)

submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': final_test_preds
})

submission.to_csv('submission_beyond_90.csv', index=False)
print("Finished! Saved optimized predictions to 'submission_beyond_90.csv'.")