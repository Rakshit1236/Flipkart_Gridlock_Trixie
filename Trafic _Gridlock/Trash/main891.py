import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import category_encoders as ce
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import pygeohash as pgh
import time
import warnings
warnings.filterwarnings('ignore')

t0 = time.time()
def p(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

p("=" * 60)
p("MAIN891 - Spatial Features from Geohash")
p("=" * 60)

train_main = pd.read_csv('train.csv')
train_extra = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
p(f"Loaded: train={train_main.shape}, extra={train_extra.shape}, test={test.shape}")

# ==========================================
# Decode geohash → lat/lon
# ==========================================
p("Decoding geohash coordinates...")
all_geos = sorted(set(train_main['geohash'].unique()) | set(test['geohash'].unique()) | set(train_extra['geohash6'].unique()))
geo_coords = {}
for gh in all_geos:
    try:
        lat, lon = pgh.decode(gh)
        geo_coords[gh] = (lat, lon)
    except:
        geo_coords[gh] = (np.nan, np.nan)

geo_df = pd.DataFrame([{'geohash': gh, 'lat': c[0], 'lon': c[1]} for gh, c in geo_coords.items()])
p(f"Decoded {len(geo_df)} geohashes")
p(f"  Lat: [{geo_df['lat'].min():.6f}, {geo_df['lat'].max():.6f}]")
p(f"  Lon: [{geo_df['lon'].min():.6f}, {geo_df['lon'].max():.6f}]")

# ==========================================
# Spatial Clustering
# ==========================================
coords = geo_df[['lat', 'lon']].values
for k in [10, 15]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    geo_df[f'cluster_{k}'] = km.fit_predict(coords)

center_lat, center_lon = geo_df['lat'].mean(), geo_df['lon'].mean()
geo_df['dist_to_center'] = np.sqrt((geo_df['lat'] - center_lat)**2 + (geo_df['lon'] - center_lon)**2)
geo_df['dist_to_cluster'] = np.sqrt(np.min(KMeans(n_clusters=15, random_state=42, n_init=10).fit(coords).transform(coords), axis=1))
p("Spatial clusters computed")

# ==========================================
# Neighbor demand aggregation
# ==========================================
p("Computing neighbor features...")
tree = cKDTree(coords)
geo_list = list(geo_df['geohash'].values)

K_N = 5
neighbor_map = {}
for i, gh in enumerate(geo_list):
    dists, idxs = tree.query(coords[i], k=K_N + 1)
    neighbor_map[gh] = [geo_list[j] for j in idxs[1:K_N + 1]]

te_df = train_extra.copy()
te_df['hour'] = te_df['timestamp'].astype(str).str.split(':').str[0].astype(int)

geo_demand = te_df.groupby('geohash6')['demand'].agg(['mean','std']).reset_index()
geo_demand.columns = ['geohash','geo_demand_mean','geo_demand_std']

neighbor_stats = []
for gh in geo_list:
    nghs = neighbor_map.get(gh, [])
    ngh_means = []
    ngh_stds = []
    for n in nghs:
        row = geo_demand.loc[geo_demand['geohash'] == n, 'geo_demand_mean']
        if len(row) > 0:
            ngh_means.append(row.values[0])
        row2 = geo_demand.loc[geo_demand['geohash'] == n, 'geo_demand_std']
        if len(row2) > 0:
            ngh_stds.append(row2.values[0])
    neighbor_stats.append({
        'geohash': gh,
        'neighbor_demand_mean': np.mean(ngh_means) if ngh_means else np.nan,
        'neighbor_demand_std': np.mean(ngh_stds) if ngh_stds else np.nan,
    })
neighbor_df = pd.DataFrame(neighbor_stats)

# Prefix-based demand
geo_df['prefix_3'] = geo_df['geohash'].str[:3]
prefix_demand = te_df.groupby(te_df['geohash6'].str[:3])['demand'].agg(['mean','std']).reset_index()
prefix_demand.columns = ['prefix_3','prefix_demand_mean','prefix_demand_std']

p("Neighbor features computed")

# ==========================================
# Stats from training.csv
# ==========================================
g1 = te_df.groupby('geohash6')['demand'].agg(['mean','std','min','max','count','median']).reset_index()
g1.columns = ['geohash','geo_mean','geo_std','geo_min','geo_max','geo_count','geo_median']
g2 = te_df.groupby(['geohash6','hour'])['demand'].agg(['mean','std','median']).reset_index()
g2.columns = ['geohash','hour','gh_mean','gh_std','gh_median']
g3 = te_df.groupby(['geohash6','hour'])['demand'].count().reset_index()
g3.columns = ['geohash','hour','gh_count']
test_geos = test[['geohash','RoadType','Weather']].drop_duplicates('geohash')
tw = te_df.merge(test_geos, left_on='geohash6', right_on='geohash', how='inner')
rth = tw.groupby(['RoadType','hour'])['demand'].agg(['mean','std']).reset_index()
rth.columns = ['RoadType','hour','rth_mean','rth_std']
rts = tw.groupby('RoadType')['demand'].agg(['mean','std']).reset_index()
rts.columns = ['RoadType','rt_mean','rt_std']
ws = tw.groupby('Weather')['demand'].agg(['mean','std']).reset_index()
ws.columns = ['Weather','w_mean','w_std']

# ==========================================
# Feature Engineering
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
    if 'NumberofLanes' in d.columns:
        d['NumberofLanes'] = d['NumberofLanes'].fillna(d['NumberofLanes'].mode()[0])
    d = d.drop(columns=['timestamp'])
    d = d.merge(g1, on='geohash', how='left')
    d = d.merge(g2, on=['geohash','hour'], how='left')
    d = d.merge(g3, on=['geohash','hour'], how='left')
    if 'RoadType' in d.columns:
        d = d.merge(rth, on=['RoadType','hour'], how='left')
        d = d.merge(rts, on='RoadType', how='left')
    if 'Weather' in d.columns:
        d = d.merge(ws, on='Weather', how='left')

    # SPATIAL FEATURES
    d = d.merge(geo_df[['geohash','lat','lon','cluster_10','cluster_15','dist_to_center','dist_to_cluster']], on='geohash', how='left')
    d = d.merge(geo_demand, on='geohash', how='left')
    d = d.merge(neighbor_df, on='geohash', how='left')
    d = d.merge(prefix_demand, left_on='geohash_3', right_on='prefix_3', how='left')
    d = d.drop(columns=['prefix_3'], errors='ignore')

    # Spatial x time interactions
    d['lat_x_hour'] = d['lat'] * d['sin_hour']
    d['lon_x_hour'] = d['lon'] * d['cos_hour']
    d['dist_x_rush'] = d['dist_to_center'] * d['is_rush_hour']

    return d

p("Building features...")
train_fe = fe(train_main)
test_fe = fe(test)
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
# Params (main888 sweet spot)
# ==========================================
best_lgb = {'lr':0.025,'nl':90,'md':9,'ra':0.8,'rl':1.5,'mcs':20}
best_xgb = {'lr':0.025,'md':8,'ra':1.0,'rl':3.0,'mcw':5}
best_cb = {'lr':0.04,'d':8,'l2':2.0}

N_FOLDS = 5
N_SEEDS = 3
p(f"\nTraining: {N_FOLDS}f x {N_SEEDS}s")

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

        m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=best_lgb['lr'], num_leaves=best_lgb['nl'],
            max_depth=best_lgb['md'], subsample=0.8, colsample_bytree=0.8,
            reg_alpha=best_lgb['ra'], reg_lambda=best_lgb['rl'], min_child_samples=best_lgb['mcs'],
            verbose=-1, n_jobs=-1, random_state=seed)
        m.fit(X_tr, yt, eval_set=[(X_vl,yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[vi] += m.predict(X_vl) / N_SEEDS
        t_lgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

        m = xgb.XGBRegressor(n_estimators=1500, learning_rate=best_xgb['lr'], max_depth=best_xgb['md'],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=best_xgb['ra'], reg_lambda=best_xgb['rl'],
            min_child_weight=best_xgb['mcw'], tree_method='hist', random_state=seed, n_jobs=-1,
            early_stopping_rounds=50)
        m.fit(X_tr, yt, eval_set=[(X_vl,yv)], verbose=False)
        oof_xgb[vi] += m.predict(X_vl) / N_SEEDS
        t_xgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

        m = CatBoostRegressor(iterations=1500, learning_rate=best_cb['lr'], depth=best_cb['d'],
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
sub.to_csv('submission_891.csv', index=False)
p(f"\nSaved submission_891.csv")
p(f"Range: [{fp.min():.6f}, {fp.max():.6f}], Mean: {fp.mean():.6f}")

# Feature importance
p(f"\nTop 20 features (LGB):")
imp = sorted(zip(nc, m.feature_importances_), key=lambda x: -x[1])[:20]
for name, val in imp:
    p(f"  {name}: {val}")
p(f"Total: {time.time()-t0:.1f}s")
