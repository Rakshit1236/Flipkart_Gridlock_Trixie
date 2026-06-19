import pandas as pd
import numpy as np
import lightgbm as lgb
import pygeohash as pgh
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

# Combine for consistent feature engineering
train['is_train'] = 1
test['is_train'] = 0
test['demand'] = -1
df = pd.concat([train, test], axis=0).reset_index(drop=True)

# 2. Advanced Feature Engineering
def feature_engineering(data):
    # A. Spatial Features: Decode Geohash
    # Geohashes represent physical bounding boxes. Extracting Lat/Lon is critical.
    data['latitude'] = data['geohash'].apply(lambda x: pgh.decode(x)[0])
    data['longitude'] = data['geohash'].apply(lambda x: pgh.decode(x)[1])
    
    # B. Temporal Features: Extracting Time Components
    # Assuming timestamp is in 'HH:MM' format based on standard gridlock data
    # If it's a full datetime object, use pd.to_datetime first.
    try:
        data['hour'] = data['timestamp'].str.split(':').str[0].astype(int)
        data['minute'] = data['timestamp'].str.split(':').str[1].astype(int)
    except:
        # Fallback if timestamp is a continuous integer or full datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['minute'] = data['timestamp'].dt.minute

    # Cyclical Time Encoding (Crucial for time-based neural nets or trees)
    # This tells the model that 23:00 and 01:00 are close to each other
    data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24.0)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24.0)
    
    # C. Categorical Encoding
    cat_cols = ['geohash', 'RoadType', 'Weather', 'Landmarks', 'LargeVehicles']
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))
        
    return data, cat_cols

df, cat_features = feature_engineering(df)

# Split back to train and test
train_df = df[df['is_train'] == 1].drop(['is_train', 'Index', 'timestamp'], axis=1)
test_df = df[df['is_train'] == 0].drop(['is_train', 'Index', 'timestamp', 'demand'], axis=1)

X = train_df.drop('demand', axis=1)
y = train_df['demand']

# 3. Hyperparameter Tuning with Optuna
def objective(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'random_state': 42,
        'verbose': -1
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
        val_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_features)
        
        # Early stopping prevents overfitting
        model = lgb.train(
            params, 
            train_data, 
            valid_sets=[val_data], 
            num_boost_round=1000, 
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds = model.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        cv_scores.append(rmse)
        
    return np.mean(cv_scores)

print("Starting Optuna optimization...")
study = optuna.create_study(direction='minimize')
# NOTE: Set n_trials to 50 or 100 for the actual hackathon run
study.optimize(objective, n_trials=15) 

print("Best parameters found: ", study.best_params)

# 4. Train Final Model with Cross Validation (K-Fold OOF Predictions)
best_params = study.best_params
best_params['objective'] = 'regression'
best_params['metric'] = 'rmse'
best_params['verbose'] = -1

kf = KFold(n_splits=5, shuffle=True, random_state=42)
test_predictions = np.zeros(len(test_df))
oof_rmse = []

print("\nTraining final models on 5 Folds...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_features)
    val_data = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_features)
    
    model = lgb.train(
        best_params, 
        train_data, 
        valid_sets=[val_data], 
        num_boost_round=2000, 
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    val_preds = model.predict(X_va)
    score = np.sqrt(mean_squared_error(y_va, val_preds))
    oof_rmse.append(score)
    print(f"Fold {fold+1} RMSE: {score:.4f}")
    
    # Accumulate test predictions
    test_predictions += model.predict(test_df) / kf.n_splits

print(f"Mean OOF RMSE: {np.mean(oof_rmse):.4f}")

# 5. Prepare Submission
submission = pd.DataFrame({
    'Index': test['Index'],
    'demand': test_predictions
})

submission.to_csv('optimized_submission.csv', index=False)
print("Submission saved to optimized_submission.csv")