import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import category_encoders as ce
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
def p(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

p("=" * 60)
p("MAIN893 - Combined Spatial+Temporal+Adversarial+GroupKFold")
p("=" * 60)

train_main = pd.read_csv('train.csv')
train_extra = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
p(f"Loaded: train={train_main.shape}, extra={train_extra.shape}, test={test.shape}")

# ==========================================
# STEP 1: Geohash decode (spatial)
# ==========================================
p("Decoding geohash coordinates...")
import pygeohash as pgh
gmap = {}
for gh in set(list(train_main['geohash'].unique()) + list(test['geohash'].unique())):
    try:
        lat, lon = pgh.decode(gh)
        gmap[gh] = (lat, lon)
    except: pass
p(f"Decoded {len(gmap)} geohashes")

# Spatial stats from training.csv
te_df = train_extra.copy()
te_df['hour'] = te_df['timestamp'].astype(str).str.split(':').str[0].astype(int)
te_df['minute'] = te_df['timestamp'].astype(str).str.split(':').str[1].astype(int)
te_df['slot'] = te_df['hour'] * 4 + te_df['minute'] // 15
te_df['day_of_week'] = te_df['day'] % 7

# ==========================================
# STEP 2: Spatial features
# ==========================================
p("Computing spatial features...")
test_geos = test[['geohash','RoadType','Weather']].drop_duplicates('geohash')
tw = te_df.merge(test_geos, left_on='geohash6', right_on='geohash', how='inner')

# Geo coordinate stats
geo_demand = te_df.groupby('geohash6')['demand'].agg(['mean','std','min','max','count','median']).reset_index()
geo_demand.columns = ['geohash','geo_demand_mean','geo_demand_std','geo_demand_min','geo_demand_max','geo_demand_count','geo_demand_median']

# Lat/lon for all geohashes
lat_map = {gh: coords[0] for gh, coords in gmap.items()}
lon_map = {gh: coords[1] for gh, coords in gmap.items()}

# K-means clusters
from sklearn.cluster import KMeans
coords = np.array([[lat_map.get(gh, 0), lon_map.get(gh, 0)] for gh in geo_demand['geohash']])
km10 = KMeans(n_clusters=10, random_state=42, n_init=10).fit(coords)
km15 = KMeans(n_clusters=15, random_state=42, n_init=10).fit(coords)
geo_demand['cluster_10'] = km10.labels_
geo_demand['cluster_15'] = km15.labels_
geo_demand['lat'] = [lat_map.get(gh, 0) for gh in geo_demand['geohash']]
geo_demand['lon'] = [lon_map.get(gh, 0) for gh in geo_demand['geohash']]

center_lat, center_lon = geo_demand['lat'].mean(), geo_demand['lon'].mean()
geo_demand['dist_to_center'] = np.sqrt((geo_demand['lat'] - center_lat)**2 + (geo_demand['lon'] - center_lon)**2)
geo_demand['dist_to_cluster'] = [np.sqrt((geo_demand['lat'].iloc[i] - km10.cluster_centers_[km10.labels_[i]][0])**2 +
    (geo_demand['lon'].iloc[i] - km10.cluster_centers_[km10.labels_[i]][1])**2) for i in range(len(geo_demand))]

# Neighbor features
from scipy.spatial import cKDTree
tree = cKDTree(coords)
k = 8
dists, idxs = tree.query(coords, k=k+1)
geo_demand['neighbor_demand_mean'] = [geo_demand['geo_demand_mean'].iloc[idxs[i][1:]].mean() for i in range(len(geo_demand))]
geo_demand['neighbor_demand_std'] = [geo_demand['geo_demand_std'].iloc[idxs[i][1:]].mean() for i in range(len(geo_demand))]

# Prefix stats
geo_demand['prefix_3'] = geo_demand['geohash'].astype(str).str[:3]
prefix_demand = geo_demand.groupby('prefix_3')['geo_demand_mean'].agg(['mean','std']).reset_index()
prefix_demand.columns = ['prefix_3','prefix_demand_mean','prefix_demand_std']
geo_demand = geo_demand.merge(prefix_demand, on='prefix_3', how='left')
geo_demand = geo_demand.drop(columns=['prefix_3'])

p(f"Spatial features: {geo_demand.shape}")

# ==========================================
# STEP 3: Temporal lag features
# ==========================================
p("Building temporal lag features...")
te_df = te_df.sort_values(['geohash6', 'day', 'slot']).reset_index(drop=True)

# Lag features per geohash
for lag in [1, 4, 96]:
    te_df[f'demand_lag_{lag}'] = te_df.groupby('geohash6')['demand'].shift(lag)

# Rolling windows per geohash
for win in [4, 12, 96]:
    te_df[f'demand_roll_{win}_mean'] = te_df.groupby('geohash6')['demand'].transform(
        lambda x: x.shift(1).rolling(win, min_periods=1).mean())
    te_df[f'demand_roll_{win}_std'] = te_df.groupby('geohash6')['demand'].transform(
        lambda x: x.shift(1).rolling(win, min_periods=1).std())

te_df['demand_exp_mean'] = te_df.groupby('geohash6')['demand'].transform(
    lambda x: x.shift(1).expanding(min_periods=1).mean())
te_df['demand_diff_1day'] = te_df.groupby('geohash6')['demand'].transform(lambda x: x - x.shift(96))

lag_cols = [c for c in te_df.columns if 'demand_lag' in c or 'demand_roll' in c or 'demand_exp' in c or 'demand_diff' in c]
gh_lag_stats = te_df.groupby(['geohash6', 'hour'])[lag_cols].agg(['mean', 'std']).reset_index()
gh_lag_stats.columns = ['geohash', 'hour'] + [f'{c}_{agg}' for c, agg in gh_lag_stats.columns[2:]]

hourly_stats = te_df.groupby(['geohash6', 'hour'])['demand'].agg(['median', 'min', 'max', 'count']).reset_index()
hourly_stats.columns = ['geohash', 'hour', 'gh_median2', 'gh_min2', 'gh_max2', 'gh_count2']

p(f"Temporal lag stats: {gh_lag_stats.shape}")

# ==========================================
# STEP 4: Standard stats from training.csv
# ==========================================
g1 = te_df.groupby('geohash6')['demand'].agg(['mean','std','min','max','count','median']).reset_index()
g1.columns = ['geohash','geo_mean','geo_std','geo_min','geo_max','geo_count','geo_median']
g2 = te_df.groupby(['geohash6','hour'])['demand'].agg(['mean','std','median']).reset_index()
g2.columns = ['geohash','hour','gh_mean','gh_std','gh_median']
g3 = te_df.groupby(['geohash6','hour'])['demand'].count().reset_index()
g3.columns = ['geohash','hour','gh_count']
rth = tw.groupby(['RoadType','hour'])['demand'].agg(['mean','std']).reset_index()
rth.columns = ['RoadType','hour','rth_mean','rth_std']
rts = tw.groupby('RoadType')['demand'].agg(['mean','std']).reset_index()
rts.columns = ['RoadType','rt_mean','rt_std']
ws = tw.groupby('Weather')['demand'].agg(['mean','std']).reset_index()
ws.columns = ['Weather','w_mean','w_std']

# ==========================================
# STEP 5: Feature Engineering (all combined)
# ==========================================
def fe(df):
    d = df.copy()
    d['hour'] = d['timestamp'].astype(str).str.split(':').str[0].astype(int)
    d['minute'] = d['timestamp'].astype(str).str.split(':').str[1].astype(int)
    d['sin_hour'] = np.sin(2*np.pi*d['hour']/24)
    d['cos_hour'] = np.cos(2*np.pi*d['hour']/24)
    d['sin_minute'] = np.sin(2*np.pi*d['minute']/60)
    d['cos_minute'] = np.cos(2*np.pi*d['minute']/60)
    d['is_rush_hour'] = d['hour'].isin([8,9,17,18,19]).astype(int)
    d['is_morning'] = d['hour'].between(6,12).astype(int)
    d['is_evening'] = d['hour'].between(17,21).astype(int)
    gc = 'geohash6' if 'geohash6' in d.columns else 'geohash'
    d['geohash_3'] = d[gc].astype(str).str[:3]
    d['geohash_4'] = d[gc].astype(str).str[:4]
    d['geohash_5'] = d[gc].astype(str).str[:5]
    d['geo4_hour'] = d['geohash_4'] + '_' + d['hour'].astype(int).astype(str)
    d['day_of_week'] = d['day'] % 7
    d['sin_dow'] = np.sin(2*np.pi*d['day_of_week']/7)
    d['cos_dow'] = np.cos(2*np.pi*d['day_of_week']/7)
    if 'RoadType' in d.columns:
        d['RoadType_enc'] = d['RoadType'].map({'Street':0,'Residential':1,'Highway':2}).fillna(-1).astype(int)
    if 'LargeVehicles' in d.columns:
        d['LargeVehicles_enc'] = d['LargeVehicles'].map({'Not Allowed':0,'Allowed':1}).fillna(-1).astype(int)
    if 'Landmarks' in d.columns:
        d['Landmarks_enc'] = d['Landmarks'].map({'No':0,'Yes':1}).fillna(-1).astype(int)
    if 'Weather' in d.columns:
        d['Weather_enc'] = d['Weather'].map({'Snowy':0,'Rainy':1,'Foggy':2,'Sunny':3}).fillna(-1).astype(int)
    if 'Temperature' in d.columns:
        d['Temperature'] = d['Temperature'].fillna(d['Temperature'].median())
        d['temp_missing'] = (d['Temperature'].isna()).astype(int)
    if 'NumberofLanes' in d.columns:
        d['NumberofLanes'] = d['NumberofLanes'].fillna(d['NumberofLanes'].mode()[0])
    d = d.drop(columns=['timestamp'])

    # Spatial features
    d = d.merge(geo_demand, on='geohash', how='left')
    d['lat_x_hour'] = d['lat'] * d['hour']
    d['lon_x_hour'] = d['lon'] * d['hour']
    d['dist_x_rush'] = d['dist_to_center'] * d['is_rush_hour']

    # Standard stats
    d = d.merge(g1, on='geohash', how='left')
    d = d.merge(g2, on=['geohash','hour'], how='left')
    d = d.merge(g3, on=['geohash','hour'], how='left')
    d = d.merge(gh_lag_stats, on=['geohash','hour'], how='left')
    d = d.merge(hourly_stats, on=['geohash','hour'], how='left')
    if 'RoadType' in d.columns:
        d = d.merge(rth, on=['RoadType','hour'], how='left')
        d = d.merge(rts, on='RoadType', how='left')
    if 'Weather' in d.columns:
        d = d.merge(ws, on='Weather', how='left')

    # Interactions
    if 'RoadType_enc' in d.columns:
        d['rt_x_hour'] = d['RoadType_enc'] * d['hour']
        d['rt_x_rush'] = d['RoadType_enc'] * d['is_rush_hour']
        d['rt_x_lat'] = d['RoadType_enc'] * d.get('lat', 0)
    if 'NumberofLanes' in d.columns:
        d['lanes_x_rt'] = d['NumberofLanes'] * d.get('RoadType_enc', 0)

    return d

p("Building all features...")
train_fe = fe(train_main)
test_fe = fe(test)

# Adversarial validation feature
feat_cols_ad = [c for c in train_fe.columns if c not in ['Index','demand'] and train_fe[c].dtype.kind in ('i','f','b')]
X_adv = pd.concat([train_fe[feat_cols_ad].assign(is_test=0), test_fe[feat_cols_ad].assign(is_test=1)]).fillna(0)
if X_adv.shape[0] > 5000:
    from sklearn.utils import shuffle
    X_adv = shuffle(X_adv, random_state=42).iloc[:5000]
adv_model = lgb.LGBMClassifier(n_estimators=50, max_depth=6, verbose=-1, random_state=42)
adv_model.fit(X_adv.drop('is_test', axis=1), X_adv['is_test'])
train_fe['adversarial'] = adv_model.predict_proba(train_fe[feat_cols_ad].fillna(0))[:, 1]
test_fe['adversarial'] = adv_model.predict_proba(test_fe[feat_cols_ad].fillna(0))[:, 1]
adv_score = np.mean(train_fe['adversarial'])
p(f"Adversarial score: train={adv_score:.4f}")

# Fill NaN
for c in train_fe.columns:
    if c == 'demand': continue
    if train_fe[c].dtype.kind in ('f','i'):
        med = train_fe[c].median()
        train_fe[c] = train_fe[c].fillna(med)
        if c in test_fe.columns: test_fe[c] = test_fe[c].fillna(med)

y = np.log1p(train_main['demand'].values)
obj_cols = [c for c in train_fe.columns if train_fe[c].dtype.kind not in ('i','f','b')]
te_cols = ['geohash_5', 'geo4_hour']
drop = ['Index','demand'] + obj_cols
nc = [c for c in train_fe.columns if c not in drop]
p(f"Features: {len(nc)}")

# ==========================================
# Params (hardcoded sweet spot)
# ==========================================
best_lgb = {'lr':0.025,'nl':90,'md':9,'ra':0.8,'rl':1.5,'mcs':20}
best_xgb = {'lr':0.025,'md':8,'ra':1.0,'rl':3.0,'mcw':5}
best_cb = {'lr':0.04,'d':8,'l2':2.0}

N_FOLDS = 5
N_SEEDS = 1
p(f"\nTraining: {N_FOLDS}f x {N_SEEDS}s")

# Use GroupKFold on geohash prefix
groups = train_fe['geohash'].astype(str).str[:3].values

oof_lgb = np.zeros(len(train_main))
oof_xgb = np.zeros(len(train_main))
oof_cb = np.zeros(len(train_main))
t_lgb = np.zeros(len(test))
t_xgb = np.zeros(len(test))
t_cb = np.zeros(len(test))

for seed in range(N_SEEDS):
    p(f"  Seed {seed+1}/{N_SEEDS}...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed*42+7)
    for ti, vi in kf.split(train_main, y):
        dtr = train_fe.iloc[ti].copy()
        dvl = train_fe.iloc[vi].copy()
        dte = test_fe.copy()
        yt, yv = y[ti], y[vi]
        for col in te_cols:
            if col in dtr.columns:
                enc = ce.TargetEncoder(cols=[col], smoothing=30)
                dtr[col] = enc.fit_transform(dtr[col], yt)
                dvl[col] = enc.transform(dvl[col])
                dte[col] = enc.transform(dte[col])
        X_tr = dtr[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        X_vl = dvl[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        av = [c for c in nc if c in dte.columns]
        X_te = dte[av].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

        m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=best_lgb['lr'], num_leaves=best_lgb['nl'],
            max_depth=best_lgb['md'], subsample=0.8, colsample_bytree=0.8,
            reg_alpha=best_lgb['ra'], reg_lambda=best_lgb['rl'], min_child_samples=best_lgb['mcs'],
            verbose=-1, n_jobs=-1, random_state=seed)
        m.fit(X_tr, yt, eval_set=[(X_vl,yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[vi] += m.predict(X_vl) / N_SEEDS
        t_lgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

        m = xgb.XGBRegressor(n_estimators=1000, learning_rate=best_xgb['lr'], max_depth=best_xgb['md'],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=best_xgb['ra'], reg_lambda=best_xgb['rl'],
            min_child_weight=best_xgb['mcw'], tree_method='hist', random_state=seed, n_jobs=-1,
            early_stopping_rounds=50)
        m.fit(X_tr, yt, eval_set=[(X_vl,yv)], verbose=False)
        oof_xgb[vi] += m.predict(X_vl) / N_SEEDS
        t_xgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

        m = CatBoostRegressor(iterations=1000, learning_rate=best_cb['lr'], depth=best_cb['d'],
            l2_leaf_reg=best_cb['l2'], random_seed=seed, verbose=0, od_type='Iter', od_wait=50)
        m.fit(X_tr, yt, eval_set=(X_vl,yv), use_best_model=True)
        oof_cb[vi] += m.predict(X_vl) / N_SEEDS
        t_cb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

true = np.expm1(y)
s_lgb = r2_score(true, np.expm1(np.clip(oof_lgb,0,None)))
s_xgb = r2_score(true, np.expm1(np.clip(oof_xgb,0,None)))
s_cb = r2_score(true, np.expm1(np.clip(oof_cb,0,None)))
p(f"\nOOF: LGB={s_lgb*100:.4f}% XGB={s_xgb*100:.4f}% CB={s_cb*100:.4f}%")

all_oof = np.column_stack([np.expm1(np.clip(oof_lgb,0,None)), np.expm1(np.clip(oof_xgb,0,None)), np.expm1(np.clip(oof_cb,0,None))])
all_test = np.column_stack([np.expm1(np.clip(t_lgb,0,None)), np.expm1(np.clip(t_xgb,0,None)), np.expm1(np.clip(t_cb,0,None))])
def obj_fn(w): return -r2_score(true, np.clip(all_oof @ w, 0, None))
res = minimize(obj_fn, [1/3,1/3,1/3], method='SLSQP', bounds=[(0,1)]*3,
               constraints={'type':'eq','fun':lambda w: 1-sum(w)})
bw = res.x
ws = r2_score(true, np.clip(all_oof @ bw, 0, None))
p(f"Weights: LGB={bw[0]:.4f} XGB={bw[1]:.4f} CB={bw[2]:.4f} => {ws*100:.4f}%")

fp = np.clip(all_test @ bw, 0, None)
sub = pd.DataFrame({'Index':test['Index'], 'demand':fp})
sub.to_csv('submission_893.csv', index=False)
p(f"\nSaved submission_893.csv")

p(f"\nTop 20 features (LGB):")
imp = sorted(zip(nc, m.feature_importances_), key=lambda x: -x[1])[:20]
for name, val in imp:
    p(f"  {name}: {val}")
p(f"Total: {time.time()-t0:.1f}s")
