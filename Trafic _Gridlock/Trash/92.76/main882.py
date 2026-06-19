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
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("MAIN882v2 - Back to 92.82% features + Optuna"); import sys; sys.stdout.flush()

train_main = pd.read_csv('train.csv')
train_extra = pd.read_csv('training.csv')
test = pd.read_csv('test.csv')

# EXACT same features as the 92.82% v7
te = train_extra.copy()
te['hour'] = te['timestamp'].astype(str).str.split(':').str[0].astype(int)
g1 = te.groupby('geohash6')['demand'].agg(['mean','std','min','max','count']).reset_index()
g1.columns = ['geohash','geo_mean','geo_std','geo_min','geo_max','geo_count']
g2 = te.groupby(['geohash6','hour'])['demand'].agg(['mean','std']).reset_index()
g2.columns = ['geohash','hour','gh_mean','gh_std']
test_geos = test[['geohash','RoadType','Weather']].drop_duplicates('geohash')
tw = te.merge(test_geos, left_on='geohash6', right_on='geohash', how='inner')
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
    d['geo4_hour'] = d['geohash_4'] + '_' + d['hour'].astype(str)
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
    if 'RoadType' in d.columns:
        d = d.merge(rth, on=['RoadType','hour'], how='left')
        d = d.merge(rts, on='RoadType', how='left')
    if 'Weather' in d.columns:
        d = d.merge(ws, on='Weather', how='left')
    return d

g1 = te.groupby('geohash6')['demand'].agg(['mean','std','min','max','count']).reset_index()
g1.columns = ['geohash','geo_mean','geo_std','geo_min','geo_max','geo_count']

print("Building features..."); sys.stdout.flush()
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
print(f"Features: {train_fe.shape[1]}, Obj: {obj_cols}"); sys.stdout.flush()

# ==========================================
# Optuna: pre-compute fold data for speed
# ==========================================
print("\nPre-computing folds..."); sys.stdout.flush()
kf_q = KFold(n_splits=3, shuffle=True, random_state=42)
qidx = list(kf_q.split(train_main, y))
fold_data = []
for ti, vi in qidx:
    dtr = train_fe.iloc[ti].copy(); dvl = train_fe.iloc[vi].copy()
    for col in te_cols:
        if col in dtr.columns:
            enc = ce.TargetEncoder(cols=[col], smoothing=30)
            dtr[col] = enc.fit_transform(dtr[col], y[ti])
            dvl[col] = enc.transform(dvl[col])
    drop = ['Index','demand'] + obj_cols
    nc = [c for c in dtr.columns if c not in drop]
    Xt = dtr[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    Xv = dvl[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    fold_data.append((Xt, Xv, y[ti], y[vi]))
print(f"Pre-computed {len(fold_data)} folds"); sys.stdout.flush()

def tune_lgb(trial):
    p = {'n_estimators':800,'learning_rate':trial.suggest_float('lr',0.01,0.05),
         'num_leaves':trial.suggest_int('nl',31,155),'max_depth':trial.suggest_int('md',5,12),
         'subsample':0.8,'colsample_bytree':0.8,
         'reg_alpha':trial.suggest_float('ra',1e-3,10,log=True),
         'reg_lambda':trial.suggest_float('rl',1e-3,10,log=True),
         'min_child_samples':trial.suggest_int('mcs',5,50),
         'verbose':-1,'n_jobs':-1,'random_state':42}
    s=[]
    for Xt,Xv,yt,yv in fold_data:
        m=lgb.LGBMRegressor(**p); m.fit(Xt,yt,eval_set=[(Xv,yv)],callbacks=[lgb.early_stopping(20,verbose=False)])
        s.append(r2_score(yv,m.predict(Xv)))
    return np.mean(s)

def tune_xgb(trial):
    p = {'n_estimators':800,'learning_rate':trial.suggest_float('lr',0.01,0.05),
         'max_depth':trial.suggest_int('md',5,12),'subsample':0.8,'colsample_bytree':0.8,
         'reg_alpha':trial.suggest_float('ra',1e-3,10,log=True),
         'reg_lambda':trial.suggest_float('rl',1e-3,10,log=True),
         'min_child_weight':trial.suggest_int('mcw',1,20),
         'tree_method':'hist','random_state':42,'n_jobs':-1,'early_stopping_rounds':20}
    s=[]
    for Xt,Xv,yt,yv in fold_data:
        m=xgb.XGBRegressor(**p); m.fit(Xt,yt,eval_set=[(Xv,yv)],verbose=False)
        s.append(r2_score(yv,m.predict(Xv)))
    return np.mean(s)

def tune_cb(trial):
    p = {'iterations':800,'learning_rate':trial.suggest_float('lr',0.02,0.08),
         'depth':trial.suggest_int('d',5,10),'l2_leaf_reg':trial.suggest_float('l2',1e-3,10,log=True),
         'random_seed':42,'verbose':0,'od_type':'Iter','od_wait':20}
    s=[]
    for Xt,Xv,yt,yv in fold_data:
        m=CatBoostRegressor(**p); m.fit(Xt,yt,eval_set=(Xv,yv),use_best_model=True)
        s.append(r2_score(yv,m.predict(Xv)))
    return np.mean(s)

print("Tuning LGB..."); sys.stdout.flush()
bl = optuna.create_study(direction='maximize'); bl.optimize(tune_lgb, n_trials=15); bl=bl.best_params; print(f"  LGB done"); sys.stdout.flush()
print("Tuning XGB..."); sys.stdout.flush()
bx = optuna.create_study(direction='maximize'); bx.optimize(tune_xgb, n_trials=15); bx=bx.best_params; print(f"  XGB done"); sys.stdout.flush()
print("Tuning CB..."); sys.stdout.flush()
bc = optuna.create_study(direction='maximize'); bc.optimize(tune_cb, n_trials=15); bc=bc.best_params; print(f"  CB done"); sys.stdout.flush()

# ==========================================
# Training with tuned params
# ==========================================
N_FOLDS = 5; N_SEEDS = 3
print(f"\nTraining: {N_FOLDS}f x {N_SEEDS}s"); sys.stdout.flush()

oof_lgb = np.zeros(len(train_main)); oof_xgb = np.zeros(len(train_main)); oof_cb = np.zeros(len(train_main))
t_lgb = np.zeros(len(test)); t_xgb = np.zeros(len(test)); t_cb = np.zeros(len(test))

for seed in range(N_SEEDS):
    print(f"  S{seed+1}", end=" "); sys.stdout.flush()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed*42+7)
    for ti, vi in kf.split(train_main, y):
        dtr = train_fe.iloc[ti].copy(); dvl = train_fe.iloc[vi].copy(); dte = test_fe.copy()
        yt, yv = y[ti], y[vi]
        for col in te_cols:
            if col in dtr.columns:
                enc = ce.TargetEncoder(cols=[col], smoothing=30)
                dtr[col] = enc.fit_transform(dtr[col], yt)
                dvl[col] = enc.transform(dvl[col])
                dte[col] = enc.transform(dte[col])
        drop = ['Index','demand'] + obj_cols
        nc = [c for c in dtr.columns if c not in drop]
        X_tr = dtr[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        X_vl = dvl[nc].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        av = [c for c in nc if c in dte.columns]
        X_te = dte[av].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

        m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=bl['lr'], num_leaves=bl['nl'],
            max_depth=bl['md'], subsample=0.8, colsample_bytree=0.8,
            reg_alpha=bl['ra'], reg_lambda=bl['rl'], min_child_samples=bl['mcs'],
            verbose=-1, n_jobs=-1, random_state=seed)
        m.fit(X_tr, yt, eval_set=[(X_vl, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_lgb[vi] += m.predict(X_vl)/N_SEEDS; t_lgb += m.predict(X_te)/(N_FOLDS*N_SEEDS)

        m = xgb.XGBRegressor(n_estimators=2000, learning_rate=bx['lr'], max_depth=bx['md'],
            subsample=0.8, colsample_bytree=0.8, reg_alpha=bx['ra'], reg_lambda=bx['rl'],
            min_child_weight=bx['mcw'], tree_method='hist', random_state=seed, n_jobs=-1,
            early_stopping_rounds=50)
        m.fit(X_tr, yt, eval_set=[(X_vl, yv)], verbose=False)
        oof_xgb[vi] += m.predict(X_vl)/N_SEEDS; t_xgb += m.predict(X_te)/(N_FOLDS*N_SEEDS)

        m = CatBoostRegressor(iterations=2000, learning_rate=bc['lr'], depth=bc['d'],
            l2_leaf_reg=bc['l2'], random_seed=seed, verbose=0, od_type='Iter', od_wait=50)
        m.fit(X_tr, yt, eval_set=(X_vl, yv), use_best_model=True)
        oof_cb[vi] += m.predict(X_vl)/N_SEEDS; t_cb += m.predict(X_te)/(N_FOLDS*N_SEEDS)
    print(); sys.stdout.flush()

true = np.expm1(y)
s_l = r2_score(true, np.expm1(np.clip(oof_lgb, 0, None)))
s_x = r2_score(true, np.expm1(np.clip(oof_xgb, 0, None)))
s_c = r2_score(true, np.expm1(np.clip(oof_cb, 0, None)))
print(f"\nOOF: LGB={s_l*100:.4f}% XGB={s_x*100:.4f}% CB={s_c*100:.4f}%")

# ==========================================
# Ensemble
# ==========================================
def obj_fn(w):
    b = w[0]*np.expm1(oof_lgb)+w[1]*np.expm1(oof_xgb)+w[2]*np.expm1(oof_cb)
    return -r2_score(true, np.clip(b, 0, None))
res = minimize(obj_fn, [1/3,1/3,1/3], method='SLSQP', bounds=[(0,1)]*3,
               constraints={'type':'eq','fun':lambda w:1-sum(w)})
bw = res.x
print(f"Weights: L={bw[0]:.4f} X={bw[1]:.4f} C={bw[2]:.4f}")

weighted = np.clip(bw[0]*np.expm1(oof_lgb)+bw[1]*np.expm1(oof_xgb)+bw[2]*np.expm1(oof_cb), 0, None)
ws = r2_score(true, weighted)

stk = np.column_stack([oof_lgb, oof_xgb, oof_cb]); stk_t = np.column_stack([t_lgb, t_xgb, t_cb])
moof = np.zeros(len(train_main)); mtest = np.zeros(len(test))
for ti, vi in KFold(5, shuffle=True, random_state=42).split(stk, y):
    m = Ridge(alpha=1.0); m.fit(stk[ti], y[ti])
    moof[vi] = m.predict(stk[vi]); mtest += m.predict(stk_t) / 5
ss = r2_score(true, np.expm1(moof))
bl_oof = np.clip(0.5*np.expm1(moof)+0.5*weighted, 0, None)
bs = r2_score(true, bl_oof)

print(f"Weighted={ws*100:.4f}% Stacking={ss*100:.4f}% Blend={bs*100:.4f}%")
best = max(ws, ss, bs)
if bs >= ws and bs >= ss:
    final = np.clip(0.5*np.expm1(mtest)+0.5*(bw[0]*np.expm1(t_lgb)+bw[1]*np.expm1(t_xgb)+bw[2]*np.expm1(t_cb)), 0, None)
    print(f">>> Blend {bs*100:.4f}%")
elif ss > ws:
    final = np.clip(np.expm1(mtest), 0, None)
    print(f">>> Stacking {ss*100:.4f}%")
else:
    final = np.clip(bw[0]*np.expm1(t_lgb)+bw[1]*np.expm1(t_xgb)+bw[2]*np.expm1(t_cb), 0, None)
    print(f">>> Weighted {ws*100:.4f}%")

print(f">>> FINAL: {best*100:.4f}%")
sub = pd.DataFrame({'Index': test['Index'], 'demand': final})
sub.to_csv('submission_882.csv', index=False)
print("Saved: submission_882.csv")
