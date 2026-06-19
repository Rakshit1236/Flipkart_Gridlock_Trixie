import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import category_encoders as ce
from scipy.optimize import minimize
import optuna, time
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

t0 = time.time()
def p(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

p("=" * 60)
p("MAIN885 - Pseudo-Labeling + Optuna + Enhanced")
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
p("Stats computed")

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

# Optuna
p("\nOptuna...")
kf_q = KFold(n_splits=2, shuffle=True, random_state=42)
qidx = list(kf_q.split(train_main, y))
fd = []
for ti, vi in qidx:
    dtr, dvl = train_fe.iloc[ti].copy(), train_fe.iloc[vi].copy()
    for col in te_cols:
        enc = ce.TargetEncoder(cols=[col], smoothing=30)
        dtr[col] = enc.fit_transform(dtr[col], y[ti])
        dvl[col] = enc.transform(dvl[col])
    Xt = dtr[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    Xv = dvl[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    fd.append((Xt, Xv, y[ti], y[vi]))

def tune_lgb(trial):
    p_ = {'n_estimators':500,'learning_rate':trial.suggest_float('lr',0.01,0.05),
          'num_leaves':trial.suggest_int('nl',31,155),'max_depth':trial.suggest_int('md',5,12),
          'subsample':0.8,'colsample_bytree':0.8,
          'reg_alpha':trial.suggest_float('ra',1e-3,10,log=True),
          'reg_lambda':trial.suggest_float('rl',1e-3,10,log=True),
          'min_child_samples':trial.suggest_int('mcs',5,50),
          'verbose':-1,'n_jobs':-1,'random_state':42}
    s = [r2_score(yv, lgb.LGBMRegressor(**p_).fit(Xt,yt,eval_set=[(Xv,yv)],
         callbacks=[lgb.early_stopping(15, verbose=False)]).predict(Xv)) for Xt,Xv,yt,yv in fd]
    return np.mean(s)

def tune_xgb(trial):
    p_ = {'n_estimators':500,'learning_rate':trial.suggest_float('lr',0.01,0.05),
          'max_depth':trial.suggest_int('md',5,12),'subsample':0.8,'colsample_bytree':0.8,
          'reg_alpha':trial.suggest_float('ra',1e-3,10,log=True),
          'reg_lambda':trial.suggest_float('rl',1e-3,10,log=True),
          'min_child_weight':trial.suggest_int('mcw',1,20),
          'tree_method':'hist','random_state':42,'n_jobs':-1,'early_stopping_rounds':15}
    s = [r2_score(yv, xgb.XGBRegressor(**p_).fit(Xt,yt,eval_set=[(Xv,yv)],
         verbose=False).predict(Xv)) for Xt,Xv,yt,yv in fd]
    return np.mean(s)

st = optuna.create_study(direction='maximize')
st.optimize(tune_lgb, n_trials=8, show_progress_bar=False)
best_lgb = st.best_params
p(f"  LGB: {st.best_value:.6f}")
st = optuna.create_study(direction='maximize')
st.optimize(tune_xgb, n_trials=8, show_progress_bar=False)
best_xgb = st.best_params
p(f"  XGB: {st.best_value:.6f}")

# Helper
def run_fold(dtr, dvl, dte, yt, yv, seed, mtype, sample_w=None):
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
    if mtype == 'lgb':
        m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=best_lgb['lr'], num_leaves=best_lgb['nl'],
            max_depth=best_lgb['md'], subsample=0.8, colsample_bytree=0.8,
            reg_alpha=best_lgb['ra'], reg_lambda=best_lgb['rl'], min_child_samples=best_lgb['mcs'],
            verbose=-1, n_jobs=-1, random_state=seed)
        m.fit(X_tr, yt, sample_weight=sample_w, eval_set=[(X_vl,yv)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m = xgb.XGBRegressor(n_estimators=1500, learning_rate=best_xgb['lr'], max_depth=best_xgb['md'],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=best_xgb['ra'], reg_lambda=best_xgb['rl'],
            min_child_weight=best_xgb['mcw'], tree_method='hist', random_state=seed, n_jobs=-1,
            early_stopping_rounds=50)
        m.fit(X_tr, yt, sample_weight=sample_w, eval_set=[(X_vl,yv)], verbose=False)
    return m.predict(X_vl), m.predict(X_te)

# ==========================================
# Round 1: 5-fold x 3-seed (fast with 2 models)
# ==========================================
N_FOLDS = 5
N_SEEDS = 3
p(f"\nRound 1: {N_FOLDS}f x {N_SEEDS}s")

oof_lgb = np.zeros(len(train_main))
oof_xgb = np.zeros(len(train_main))
t_lgb = np.zeros(len(test))
t_xgb = np.zeros(len(test))

for seed in range(N_SEEDS):
    p(f"  Seed {seed+1}/{N_SEEDS}...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed*42+7)
    for ti, vi in kf.split(train_main, y):
        oof_lgb[vi], pl = run_fold(train_fe.iloc[ti].copy(), train_fe.iloc[vi].copy(), test_fe.copy(), y[ti], y[vi], seed, 'lgb')
        oof_xgb[vi], px = run_fold(train_fe.iloc[ti].copy(), train_fe.iloc[vi].copy(), test_fe.copy(), y[ti], y[vi], seed, 'xgb')
        t_lgb += pl / (N_FOLDS * N_SEEDS)
        t_xgb += px / (N_FOLDS * N_SEEDS)

true = np.expm1(y)
s_lgb = r2_score(true, np.expm1(np.clip(oof_lgb,0,None)))
s_xgb = r2_score(true, np.expm1(np.clip(oof_xgb,0,None)))
p(f"\nRound 1: LGB={s_lgb*100:.4f}% XGB={s_xgb*100:.4f}%")

# ==========================================
# Pseudo-Labeling (top 20% most confident)
# ==========================================
p("\nPseudo-Labeling...")
test_mean = (np.expm1(t_lgb) + np.expm1(t_xgb)) / 2
test_std = np.abs(np.expm1(t_lgb) - np.expm1(t_xgb))
thresh = np.percentile(test_std, 20)
pseudo_idx = np.where(test_std < thresh)[0]
p(f"  Confident: {len(pseudo_idx)}/{len(test)} ({100*len(pseudo_idx)/len(test):.1f}%)")

p_y = test_mean[pseudo_idx]
p_w = np.clip(1.0 - test_std[pseudo_idx] / (test_std.max() + 1e-8), 0.1, 1.0) * 0.3

p_train = pd.concat([train_fe, test_fe.iloc[pseudo_idx].reset_index(drop=True)], ignore_index=True)
p_y_full = np.concatenate([y, p_y])
p_w_full = np.concatenate([np.ones(len(y)), p_w])
p(f"  Augmented: {len(p_train)} rows")

# Round 2: 5-fold x 3-seed with pseudo-labels
p(f"\nRound 2: {N_FOLDS}f x {N_SEEDS}s (pseudo)")

oof2_lgb = np.zeros(len(train_main))
oof2_xgb = np.zeros(len(train_main))
t2_lgb = np.zeros(len(test))
t2_xgb = np.zeros(len(test))

for seed in range(N_SEEDS):
    p(f"  Seed {seed+1}/{N_SEEDS}...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed*42+7)
    for ti, vi in kf.split(p_train, p_y_full):
        dtr = p_train.iloc[ti].copy()
        dvl = p_train.iloc[vi].copy()
        dte = test_fe.copy()
        yt, yv = p_y_full[ti], p_y_full[vi]
        w_tr = p_w_full[ti]

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
        m.fit(X_tr, yt, sample_weight=w_tr, eval_set=[(X_vl,yv)],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        vl_pred_lgb = m.predict(X_vl)
        real_mask_vl = vi < len(train_main)
        if real_mask_vl.any():
            real_vi = vi[real_mask_vl]
            oof2_lgb[real_vi] += vl_pred_lgb[real_mask_vl] / N_SEEDS
        t2_lgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

        m = xgb.XGBRegressor(n_estimators=1500, learning_rate=best_xgb['lr'], max_depth=best_xgb['md'],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=best_xgb['ra'], reg_lambda=best_xgb['rl'],
            min_child_weight=best_xgb['mcw'], tree_method='hist', random_state=seed, n_jobs=-1,
            early_stopping_rounds=50)
        m.fit(X_tr, yt, sample_weight=w_tr, eval_set=[(X_vl,yv)], verbose=False)
        vl_pred_xgb = m.predict(X_vl)
        if real_mask_vl.any():
            oof2_xgb[real_vi] += vl_pred_xgb[real_mask_vl] / N_SEEDS
        t2_xgb += m.predict(X_te) / (N_FOLDS * N_SEEDS)

s2_lgb = r2_score(true, np.expm1(np.clip(oof2_lgb,0,None)))
s2_xgb = r2_score(true, np.expm1(np.clip(oof2_xgb,0,None)))
p(f"\nRound 2: LGB={s2_lgb*100:.4f}% XGB={s2_xgb*100:.4f}%")

# ==========================================
# Combine: 4 models x weight optimization
# ==========================================
all_oof = np.column_stack([
    np.expm1(np.clip(oof_lgb,0,None)),
    np.expm1(np.clip(oof_xgb,0,None)),
    np.expm1(np.clip(oof2_lgb,0,None)),
    np.expm1(np.clip(oof2_xgb,0,None))
])
all_test = np.column_stack([
    np.expm1(np.clip(t_lgb,0,None)),
    np.expm1(np.clip(t_xgb,0,None)),
    np.expm1(np.clip(t2_lgb,0,None)),
    np.expm1(np.clip(t2_xgb,0,None))
])

def obj_fn(w): return -r2_score(true, np.clip(all_oof @ w, 0, None))
res = minimize(obj_fn, np.ones(4)/4, method='SLSQP', bounds=[(0,1)]*4,
               constraints={'type':'eq','fun':lambda w: 1-sum(w)})
bw = res.x
ws = r2_score(true, np.clip(all_oof @ bw, 0, None))
p(f"\n4-model weights: R1_LGB={bw[0]:.3f} R1_XGB={bw[1]:.3f} R2_LGB={bw[2]:.3f} R2_XGB={bw[3]:.3f}")
p(f"Weighted OOF: {ws*100:.4f}%")

# Ridge stacking
meta_oof = np.zeros(len(train_main))
meta_test = np.zeros(len(test))
for ti,vi in KFold(5, shuffle=True, random_state=42).split(all_oof, y):
    m = Ridge(alpha=1.0); m.fit(all_oof[ti], y[ti])
    meta_oof[vi] = m.predict(all_oof[vi])
    meta_test += m.predict(all_test) / 5
ss = r2_score(true, np.expm1(meta_oof))
p(f"Stacking OOF: {ss*100:.4f}%")

# Blend
blend = np.clip(0.5*np.expm1(meta_oof) + 0.5*np.clip(all_oof @ bw, 0, None), 0, None)
bs = r2_score(true, blend)
p(f"Blend OOF: {bs*100:.4f}%")

# Final
scores = {'Weighted4':ws, 'Stacking':ss, 'Blend':bs}
best_name = max(scores, key=scores.get)
best_score = scores[best_name]
p(f"\nAll: {', '.join(f'{k}={v*100:.4f}%' for k,v in scores.items())}")
p(f">>> Best: {best_name} ({best_score*100:.4f}%)")

if best_name == 'Blend':
    fp = np.clip(0.5*meta_test + 0.5*(all_test @ bw), 0, None)
elif best_name == 'Stacking':
    fp = np.clip(meta_test, 0, None)
else:
    fp = np.clip(all_test @ bw, 0, None)

sub = pd.DataFrame({'Index':test['Index'], 'demand':fp})
sub.to_csv('submission_885.csv', index=False)
p(f"\nSaved submission_885.csv")
p(f"Range: [{fp.min():.6f}, {fp.max():.6f}], Mean: {fp.mean():.6f}")
p(f"Total: {time.time()-t0:.1f}s")
