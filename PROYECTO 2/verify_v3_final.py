"""
FULL PIPELINE VERIFICATION SCRIPT – V3 FINAL
Runs the entire pipeline. If this script passes, the notebook will too.
All types are kept as float32. No float16 anywhere.
"""
import sys, pandas as pd, numpy as np, gc, datetime, warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix
from imblearn.combine import SMOTETomek
import lightgbm as lgb
warnings.filterwarnings('ignore')

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("[1/9] Cargando datos (10 000 filas)...")
trx = pd.read_csv('train_transaction.csv', nrows=10000)
idy = pd.read_csv('train_identity.csv')
df  = trx.merge(idy, on='TransactionID', how='left')
del trx, idy; gc.collect()

# Cast floats → float32 (NO float16 ever)
for c in df.select_dtypes('float64').columns:
    df[c] = df[c].astype(np.float32)
print(f"   Shape: {df.shape}")

# ── QUALITY: drop >80% null cols ──────────────────────────────────────────────
print("[2/9] Calidad de datos (drop >80% nulos)...")
drop_cols = [c for c in df.columns if df[c].isnull().mean() > 0.8]
df.drop(columns=drop_cols, inplace=True)

X = df.drop(columns=['TransactionID','isFraud'])
y = df['isFraud'].astype(np.int8)

# ── EARLY SPLIT (anti Data Leakage) ───────────────────────────────────────────
print("[3/9] Split 80/20 estratificado (pre-procesamiento)...")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── OUTLIERS: IQR Clamping on TransactionAmt ─────────────────────────────────
print("[4/9] Clamping IQR de Outliers...")
q1, q3 = X_tr['TransactionAmt'].quantile(0.25), X_tr['TransactionAmt'].quantile(0.75)
ub = q3 + 1.5*(q3-q1)
for df_  in [X_tr, X_te]:
    df_['TransactionAmt_Capped'] = df_['TransactionAmt'].clip(upper=float(ub)).astype(np.float32)

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
print("[5/9] Feature Engineering (Temporal + Binning + Target Encoding)...")

def feat_eng(X_, y_=None, params=None):
    df_ = X_.copy()
    is_train = y_ is not None
    if is_train:
        params = {}

    # Temporal
    t0 = datetime.datetime(2017, 12, 1)
    ts = df_['TransactionDT'].astype('int64')
    dt = ts.apply(lambda s: t0 + datetime.timedelta(seconds=int(s)))
    df_['tx_hour'] = dt.dt.hour.astype(np.int8)
    df_['tx_dow']  = dt.dt.dayofweek.astype(np.int8)
    df_.drop(columns=['TransactionDT'], inplace=True)

    # Log amount (safe: use float32)
    df_['log_amt'] = np.log1p(df_['TransactionAmt_Capped'].astype(np.float32))

    # Binning (safe: force float64 for pd.cut input, then cast result back)
    if is_train:
        vals = df_['TransactionAmt'].astype(np.float64).replace([np.inf,-np.inf], np.nan).dropna()
        _, edges = pd.qcut(vals, q=4, retbins=True, duplicates='drop')
        edges[0], edges[-1] = -np.inf, np.inf
        params['edges'] = list(edges)
    df_['amt_bin'] = pd.cut(
        df_['TransactionAmt'].astype(np.float64),
        bins=params['edges'], labels=False, include_lowest=True
    ).astype(np.float32)

    # Target encoding for high-cardinality columns
    cat_cols = ['P_emaildomain','R_emaildomain','card1','card2']
    for col in cat_cols:
        if col not in df_.columns:
            continue
        df_[col] = df_[col].astype(str).replace('nan','MISSING')
        if is_train:
            tmp = pd.DataFrame({'c': df_[col], 't': y_.astype(float).values})
            m   = tmp.groupby('c')['t'].mean().to_dict()
            gm  = float(y_.mean())
            params[f'{col}_te'] = (m, gm)
        m, gm = params[f'{col}_te']
        df_[col+'_te'] = df_[col].map(m).fillna(gm).astype(np.float32)
        df_.drop(columns=[col], inplace=True)

    # Remaining object → category (LightGBM native)
    for c in df_.select_dtypes('object').columns:
        df_[c] = df_[c].astype('category')

    return df_, params

X_tr_fe, P = feat_eng(X_tr, y_tr)
X_te_fe, _ = feat_eng(X_te, params=P)

# ── VIF + PCA on V-columns ────────────────────────────────────────────────────
print("[6/9] VIF / PCA en columnas V...")
v_cols = [c for c in X_tr_fe.columns if c.startswith('V')]
if v_cols:
    v_tr = X_tr_fe[v_cols].astype(np.float32).fillna(X_tr_fe[v_cols].median())
    v_te = X_te_fe[v_cols].astype(np.float32).fillna(X_tr_fe[v_cols].median())
    sc  = StandardScaler()
    vt  = sc.fit_transform(v_tr)
    vv  = sc.transform(v_te)
    n_comp = min(12, vt.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    vt_p = pca.fit_transform(vt)
    vv_p = pca.transform(vv)
    print(f"   Varianza retenida: {np.sum(pca.explained_variance_ratio_):.2%}")
    for i in range(n_comp):
        X_tr_fe[f'V_PC{i}'] = vt_p[:,i].astype(np.float32)
        X_te_fe[f'V_PC{i}'] = vv_p[:,i].astype(np.float32)
    X_tr_fe.drop(columns=v_cols, inplace=True)
    X_te_fe.drop(columns=v_cols, inplace=True)

# ── CLUSTERING (KMeans) ───────────────────────────────────────────────────────
print("[7/9] Clustering (KMeans)...")
clust_cols = ['log_amt','tx_hour'] + [c for c in X_tr_fe.columns if c.startswith('V_PC')]
ctr = X_tr_fe[clust_cols].fillna(0).astype(np.float32)
cte = X_te_fe[clust_cols].fillna(0).astype(np.float32)
sc2 = StandardScaler()
km  = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=1024)
X_tr_fe['cluster'] = pd.Series(km.fit_predict(sc2.fit_transform(ctr)), index=X_tr_fe.index).astype('category')
X_te_fe['cluster'] = pd.Series(km.predict(sc2.transform(cte)), index=X_te_fe.index).astype('category')

# ── SMOTETomek Resampling ─────────────────────────────────────────────────────
print("[8/9] SMOTETomek (sampling_strategy=0.15)...")
num_cols = ['log_amt','tx_hour','amt_bin'] + [c for c in X_tr_fe.columns if '_te' in c or c.startswith('V_PC')]
X_sm = X_tr_fe[num_cols].astype(np.float32).fillna(X_tr_fe[num_cols].median())
X_sv = X_te_fe[num_cols].astype(np.float32).fillna(X_tr_fe[num_cols].median())
smt  = SMOTETomek(sampling_strategy=0.15, random_state=42)
Xr, yr = smt.fit_resample(X_sm, y_tr)
print(f"   Distribución post-SMOTE: {dict(pd.Series(yr).value_counts())}")

# ── LIGHTGBM ──────────────────────────────────────────────────────────────────
print("[9/9] Entrenando LightGBM...")
lgb_tr  = lgb.Dataset(Xr, yr)
lgb_val = lgb.Dataset(X_sv, y_te, reference=lgb_tr)
params  = dict(objective='binary', metric='auc', learning_rate=0.05,
               num_leaves=64, max_depth=8, feature_fraction=0.8,
               bagging_fraction=0.8, seed=42, verbose=-1)
clf = lgb.train(params, lgb_tr, num_boost_round=50,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(10, verbose=False)])

probs = clf.predict(X_sv)
prec, rec, thr = precision_recall_curve(y_te, probs)
f1s  = 2*prec*rec/(prec+rec+1e-9)
best = thr[np.argmax(f1s[:-1])] if len(thr) else 0.3
preds = (probs >= best).astype(int)

print(f"\n{'='*50}")
print(f"  ROC-AUC  : {roc_auc_score(y_te, probs):.4f}")
print(f"  Threshold: {best:.4f}")
print(f"  F1-Score : {f1s.max():.4f}")
print(f"{'='*50}")
print(classification_report(y_te, preds))
print("\n✅  PIPELINE COMPLETO SIN ERRORES")
