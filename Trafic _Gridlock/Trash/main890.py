import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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
p("MAIN890 - main888 Params x 3 Variations + 5 Seeds")
p("=" * 60)

train_main = pd.read_csv('train.csv')
train_extra = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')
p(f"Loaded: train={train_main.shape}, extra={train_extra.shape}, test={test.shape}")

te_df = train_extra.copy()
te_df['hour'] = te_df['timestamp'].astype(str).str.split(':').str[0].astype(int)
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

# 3 variations around main888's sweet spot
param_sets = [
    {'name': 'v1_exact', 'lgb': {'lr':0.025,'nl':90,'md':9,'ra':0.8,'rl':1.5,'mcs':20}, 'xgb': {'lr':0.025,'md':8,'ra':1.0,'rl':3.0,'mcw':5}, 'cb': {'lr':0.04,'d':8,'l2':2.0}},
    {'name': 'v2_more_reg', 'lgb': {'lr':0.02,'nl':80,'md':8,'ra':1.2,'rl':2.5,'mcs':25}, 'xgb': {'lr':0.02,'md':7,'ra':1.5,'rl':4.0,'mcw':7}, 'cb': {'lr':0.035,'d':7,'l2':3.5}},
]

N_FOLDS = 5
N_SEEDS = 3
all_preds = {}
all_oofs = {}
true = np.expm1(y)

for ps in param_sets:
    p(f"\n--- {ps['name']} ---")
    blg, bxb, bcb = ps['lgb'], ps['xgb'], ps['cb']

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

            m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=blg['lr'], num_leaves=blg['nl'],
                max_depth=blg['md'], subsample=0.8, colsample_bytree=0.8,
                reg_alpha=blg['ra'], reg_lambda=blg['rl'], min_child_samples=blg['mcs'],
                verbose=-1, n_jobs=-1, random_state=seed)
            m.fit(X_tr, yt, eval_set=[(X_vl,yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof_lgb[vi] += m.predict(X_vl) / N_SEEDS
            t_lgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

            m = xgb.XGBRegressor(n_estimators=1500, learning_rate=bxb['lr'], max_depth=bxb['md'],
                subsample=0.8, colsample_bytree=0.8, reg_alpha=bxb['ra'], reg_lambda=bxb['rl'],
                min_child_weight=bxb['mcw'], tree_method='hist', random_state=seed, n_jobs=-1,
                early_stopping_rounds=50)
            m.fit(X_tr, yt, eval_set=[(X_vl,yv)], verbose=False)
            oof_xgb[vi] += m.predict(X_vl) / N_SEEDS
            t_xgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

            m = CatBoostRegressor(iterations=1500, learning_rate=bcb['lr'], depth=bcb['d'],
                l2_leaf_reg=bcb['l2'], random_seed=seed, verbose=0, od_type='Iter', od_wait=50)
            m.fit(X_tr, yt, eval_set=(X_vl,yv), use_best_model=True)
            oof_cb[vi] += m.predict(X_vl) / N_SEEDS
            t_cb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

    s_lgb = r2_score(true, np.expm1(np.clip(oof_lgb,0,None)))
    s_xgb = r2_score(true, np.expm1(np.clip(oof_xgb,0,None)))
    s_cb = r2_score(true, np.expm1(np.clip(oof_cb,0,None)))
    p(f"  OOF: LGB={s_lgb*100:.4f}% XGB={s_xgb*100:.4f}% CB={s_cb*100:.4f}%")

    all_oof_np = np.column_stack([np.expm1(np.clip(oof_lgb,0,None)), np.expm1(np.clip(oof_xgb,0,None)), np.expm1(np.clip(oof_cb,0,None))])
    all_test_np = np.column_stack([np.expm1(np.clip(t_lgb,0,None)), np.expm1(np.clip(t_xgb,0,None)), np.expm1(np.clip(t_cb,0,None))])
    def obj_fn(w): return -r2_score(true, np.clip(all_oof_np @ w, 0, None))
    res = minimize(obj_fn, [1/3,1/3,1/3], method='SLSQP', bounds=[(0,1)]*3,
                   constraints={'type':'eq','fun':lambda w: 1-sum(w)})
    bw = res.x
    ws = r2_score(true, np.clip(all_oof_np @ bw, 0, None))
    p(f"  Weights: LGB={bw[0]:.4f} XGB={bw[1]:.4f} CB={bw[2]:.4f} => {ws*100:.4f}%")

    all_preds[ps['name']] = np.clip(all_test_np @ bw, 0, None)
    all_oofs[ps['name']] = all_oof_np
    p(f"  Weighted: {ws*100:.4f}%")

# Save individual best
p(f"\n{'='*60}")

# Avg all 2
avg3 = np.mean([v for v in all_preds.values()], axis=0)
sub = pd.DataFrame({'Index':test['Index'], 'demand':avg3})
sub.to_csv('submission_890.csv', index=False)
p(f"\nSaved submission_890.csv (avg of 3 param sets)")

# Avg v1+v2 only (skip v3 which is less regularized)
avg12 = (all_preds['v1_exact'] + all_preds['v2_more_reg']) / 2
sub2 = pd.DataFrame({'Index':test['Index'], 'demand':avg12})
sub2.to_csv('submission_890_v12.csv', index=False)
p(f"Saved submission_890_v12.csv (avg of v1+v2)")

# v1 only (exact main888 params with 5 seeds)
sub3 = pd.DataFrame({'Index':test['Index'], 'demand':all_preds['v1_exact']})
sub3.to_csv('submission_890_v1.csv', index=False)
p(f"Saved submission_890_v1.csv (v1 exact)")

# v2 only
sub4 = pd.DataFrame({'Index':test['Index'], 'demand':all_preds['v2_more_reg']})
sub4.to_csv('submission_890_v2.csv', index=False)
p(f"Saved submission_890_v2.csv (v2 more reg)")

p(f"\nTotal: {time.time()-t0:.1f}s")
