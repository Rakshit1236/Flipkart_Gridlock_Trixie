import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import category_encoders as ce
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("MAIN88 Final")
print("=" * 60)

train_main = pd.read_csv('train.csv')
train_extra = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
print(f"train: {train_main.shape}, extra: {train_extra.shape}, test: {test.shape}")

# ==========================================
# Stats from training.csv
# ==========================================
geo_s = train_extra.groupby('geohash6')['demand'].agg(['mean','std','min','max','count']).reset_index()
geo_s.columns = ['geohash','gmean','gstd','gmin','gmax','gcnt']

train_extra['hour'] = train_extra['timestamp'].astype(str).str.split(':').str[0].astype(int)
train_extra['minute'] = train_extra['timestamp'].astype(str).str.split(':').str[1].astype(int)

gh = train_extra.groupby(['geohash6','hour'])['demand'].agg(['mean','std']).reset_index()
gh.columns = ['geohash','hour','ghm','ghs']

# RoadType stats
test_geos = test[['geohash','RoadType','Weather']].drop_duplicates('geohash')
tw = train_extra.merge(test_geos, left_on='geohash6', right_on='geohash', how='inner')
rth = tw.groupby(['RoadType','hour'])['demand'].agg(['mean','std']).reset_index()
rth.columns = ['RoadType','hour','rthm','rths']
rt_s = tw.groupby('RoadType')['demand'].agg(['mean','std']).reset_index()
rt_s.columns = ['RoadType','rtm','rts']
w_s = tw.groupby('Weather')['demand'].agg(['mean','std']).reset_index()
w_s.columns = ['Weather','wm','ws']

# ==========================================
# Feature Engineering
# ==========================================
def feature_engineering(df):
    data = df.copy()
    data['hour'] = data['timestamp'].astype(str).str.split(':').str[0].astype(int)
    data['minute'] = data['timestamp'].astype(str).str.split(':').str[1].astype(int)
    data['sin_h'] = np.sin(2*np.pi*data['hour']/24)
    data['cos_h'] = np.cos(2*np.pi*data['hour']/24)
    data['is_rush'] = data['hour'].isin([8,9,17,18,19]).astype(int)

    gc = 'geohash6' if 'geohash6' in data.columns else 'geohash'
    data['g3'] = data[gc].astype(str).str[:3]
    data['g4'] = data[gc].astype(str).str[:4]
    data['g5'] = data[gc].astype(str).str[:5]
    data['g4h'] = data['g4'] + '_' + data['hour'].astype(str)

    if 'RoadType' in data.columns:
        data['rte'] = data['RoadType'].map({'Street':0,'Residential':1,'Highway':2}).fillna(-1).astype(int)
    if 'LargeVehicles' in data.columns:
        data['lve'] = data['LargeVehicles'].map({'Not Allowed':0,'Allowed':1}).fillna(-1).astype(int)
    if 'Landmarks' in data.columns:
        data['lme'] = data['Landmarks'].map({'No':0,'Yes':1}).fillna(-1).astype(int)
    if 'Weather' in data.columns:
        data['we'] = data['Weather'].map({'Snowy':0,'Rainy':1,'Foggy':2,'Sunny':3}).fillna(-1).astype(int)
    if 'Temperature' in data.columns:
        data['Temperature'] = data['Temperature'].fillna(data['Temperature'].median())
    if 'NumberofLanes' in data.columns:
        data['NumberofLanes'] = data['NumberofLanes'].fillna(data['NumberofLanes'].mode()[0])

    data = data.drop(columns=['timestamp'])

    data = data.merge(geo_s, on='geohash', how='left')
    data = data.merge(gh, on=['geohash','hour'], how='left')
    if 'RoadType' in data.columns:
        data = data.merge(rth, on=['RoadType','hour'], how='left')
        data = data.merge(rt_s, on='RoadType', how='left')
    if 'Weather' in data.columns:
        data = data.merge(w_s, on='Weather', how='left')

    return data

print("Building features...")
train_fe = feature_engineering(train_main)
test_fe = feature_engineering(test)

for c in train_fe.columns:
    if c == 'demand': continue
    if train_fe[c].dtype.kind in ('f','i'):
        med = train_fe[c].median()
        train_fe[c] = train_fe[c].fillna(med)
        if c in test_fe.columns:
            test_fe[c] = test_fe[c].fillna(med)

print(f"Features: {train_fe.shape[1]}")

y = np.log1p(train_main['demand'].values)
obj_cols = [c for c in train_fe.columns if train_fe[c].dtype.kind not in ('i','f','b')]
te_cols = ['g5', 'g4h']
print(f"Object cols: {obj_cols}")

# ==========================================
# Training: 5-fold x 3-seeds
# ==========================================
N_FOLDS = 5
N_SEEDS = 3
print(f"\nTraining: {N_FOLDS} folds x {N_SEEDS} seeds")

oof_lgb = np.zeros(len(train_main))
oof_xgb = np.zeros(len(train_main))
oof_cb = np.zeros(len(train_main))
test_lgb = np.zeros(len(test))
test_xgb = np.zeros(len(test))
test_cb = np.zeros(len(test))

for seed in range(N_SEEDS):
    print(f"  Seed {seed+1}/{N_SEEDS}...")
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

        drop = ['Index','demand'] + obj_cols
        X_tr = dtr.drop(columns=[c for c in drop if c in dtr.columns])
        X_vl = dvl.drop(columns=[c for c in drop if c in dvl.columns])
        X_te = dte.drop(columns=[c for c in obj_cols + ['Index'] if c in dte.columns])

        nc = X_tr.columns.tolist()
        X_tr = X_tr[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        X_vl = X_vl[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        av = [c for c in nc if c in X_te.columns]
        X_te = X_te[av].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

        # LGB
        m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=63, max_depth=8,
            subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=-1, random_state=seed)
        m.fit(X_tr, yt, eval_set=[(X_vl, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[vi] += m.predict(X_vl) / N_SEEDS
        test_lgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

        # XGB
        m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, tree_method='hist', random_state=seed,
            n_jobs=-1, early_stopping_rounds=50)
        m.fit(X_tr, yt, eval_set=[(X_vl, yv)], verbose=False)
        oof_xgb[vi] += m.predict(X_vl) / N_SEEDS
        test_xgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

        # CB
        m = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=7,
            random_seed=seed, verbose=0, od_type='Iter', od_wait=50)
        m.fit(X_tr, yt, eval_set=(X_vl, yv), use_best_model=True)
        oof_cb[vi] += m.predict(X_vl) / N_SEEDS
        test_cb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

true = np.expm1(y)
s_lgb = r2_score(true, np.expm1(np.clip(oof_lgb, 0, None)))
s_xgb = r2_score(true, np.expm1(np.clip(oof_xgb, 0, None)))
s_cb = r2_score(true, np.expm1(np.clip(oof_cb, 0, None)))
print(f"\nOOF: LGB={s_lgb*100:.4f}%  XGB={s_xgb*100:.4f}%  CB={s_cb*100:.4f}%")

# ==========================================
# Ensemble
# ==========================================
def obj_fn(w):
    b = w[0]*np.expm1(oof_lgb)+w[1]*np.expm1(oof_xgb)+w[2]*np.expm1(oof_cb)
    return -r2_score(true, np.clip(b, 0, None))

res = minimize(obj_fn, [1/3,1/3,1/3], method='SLSQP',
               bounds=[(0,1)]*3, constraints={'type':'eq','fun':lambda w:1-sum(w)})
bw = res.x
print(f"Weights: LGB={bw[0]:.4f} XGB={bw[1]:.4f} CB={bw[2]:.4f}")

weighted = np.clip(bw[0]*np.expm1(oof_lgb)+bw[1]*np.expm1(oof_xgb)+bw[2]*np.expm1(oof_cb), 0, None)
ws = r2_score(true, weighted)
print(f"Weighted: {ws*100:.4f}%")

stk = np.column_stack([oof_lgb, oof_xgb, oof_cb])
stk_t = np.column_stack([test_lgb, test_xgb, test_cb])
moof = np.zeros(len(train_main)); mtest = np.zeros(len(test))
for ti, vi in KFold(5, shuffle=True, random_state=42).split(stk, y):
    m = Ridge(alpha=1.0); m.fit(stk[ti], y[ti])
    moof[vi] = m.predict(stk[vi]); mtest += m.predict(stk_t) / 5
ss = r2_score(true, np.expm1(moof))
print(f"Stacking: {ss*100:.4f}%")

bl = np.clip(0.5*np.expm1(moof)+0.5*weighted, 0, None)
bs = r2_score(true, bl)
print(f"Blend: {bs*100:.4f}%")

best = max(ws, ss, bs)
if bs >= ws and bs >= ss:
    final = np.clip(0.5*np.expm1(mtest)+0.5*(bw[0]*np.expm1(test_lgb)+bw[1]*np.expm1(test_xgb)+bw[2]*np.expm1(test_cb)), 0, None)
    print(f"\n>>> Blend ({bs*100:.4f}%)")
elif ss > ws:
    final = np.clip(np.expm1(mtest), 0, None)
    print(f"\n>>> Stacking ({ss*100:.4f}%)")
else:
    final = np.clip(bw[0]*np.expm1(test_lgb)+bw[1]*np.expm1(test_xgb)+bw[2]*np.expm1(test_cb), 0, None)
    print(f"\n>>> Weighted ({ws*100:.4f}%)")

print(f">>> FINAL: {best*100:.4f}%")
sub = pd.DataFrame({'Index': test['Index'], 'demand': final})
sub.to_csv('submission_main88.csv', index=False)
print(f"Saved: submission_main88.csv")
