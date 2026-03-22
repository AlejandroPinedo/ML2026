import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
import warnings

from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_recall_curve, f1_score, roc_auc_score,
    confusion_matrix, classification_report, make_scorer
)
from imblearn.combine import SMOTETomek
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max: df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype(np.float32)
    return df

print("Cargando y uniendo datasets (SAMPLE 10K para Testeo Rapido)...")
train_transaction = pd.read_csv('train_transaction.csv', nrows=10000)
train_identity = pd.read_csv('train_identity.csv', nrows=10000)
train = train_transaction.merge(train_identity, on='TransactionID', how='left')
del train_transaction, train_identity
gc.collect()

train = reduce_mem_usage(train)

null_percent = train.isnull().sum() / len(train)
cols_to_drop = null_percent[null_percent > 0.8].index.tolist()
train.drop(columns=cols_to_drop, inplace=True)

X = train.drop(columns=['TransactionID', 'isFraud'])
y = train['isFraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Q1 = X_train['TransactionAmt'].quantile(0.25)
Q3 = X_train['TransactionAmt'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

X_train['TransactionAmt_Capped'] = np.where(X_train['TransactionAmt'] > upper_bound, upper_bound, X_train['TransactionAmt'])
X_test['TransactionAmt_Capped'] = np.where(X_test['TransactionAmt'] > upper_bound, upper_bound, X_test['TransactionAmt'])

def apply_feature_engineering(X_curr, y_curr=None, fit_params=None):
    df_feat = X_curr.copy()
    is_train = y_curr is not None
    if is_train: fit_params = {}
    
    import datetime
    start_date = datetime.datetime.strptime("2017-12-01", "%Y-%m-%d")
    dates = df_feat['TransactionDT'].apply(lambda dt: start_date + datetime.timedelta(seconds=dt))
    df_feat['Transaction_Hour'] = dates.dt.hour
    df_feat.drop(columns=['TransactionDT'], inplace=True)
    
    if is_train:
        clean_amt = df_feat['TransactionAmt'].astype('float32').replace([np.inf, -np.inf], np.nan).dropna()
        _, bins = pd.qcut(clean_amt, q=4, retbins=True, duplicates='drop')
        bins[0] = -np.inf
        bins[-1] = np.inf
        fit_params['amt_bins'] = bins
    df_feat['Amt_Bin'] = pd.cut(df_feat['TransactionAmt'].astype('float32'), 
                                bins=fit_params['amt_bins'], labels=False, include_lowest=True).astype('float32')
    df_feat['LogTransactionAmt'] = np.log1p(df_feat['TransactionAmt_Capped'].astype('float32'))
    
    high_card_cat = ['P_emaildomain', 'R_emaildomain', 'card1', 'card2']
    for col in high_card_cat:
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna('MISSING')
            if is_train:
                temp_df = pd.DataFrame({col: df_feat[col], 'target': y_curr})
                means = temp_df.groupby(col)['target'].mean().to_dict()
                global_mean = y_curr.mean()
                fit_params[f'{col}_te'] = (means, global_mean)
                
            means, global_mean = fit_params[f'{col}_te']
            df_feat[col + '_TE'] = df_feat[col].map(means).fillna(global_mean)
            df_feat.drop(columns=[col], inplace=True)

    for c in df_feat.select_dtypes(include=['object']).columns:
        df_feat[c] = df_feat[c].astype('category')
        
    return df_feat, fit_params

X_train_fe, prep_params = apply_feature_engineering(X_train, y_train)
X_test_fe, _ = apply_feature_engineering(X_test, fit_params=prep_params)

v_cols = [c for c in X_train_fe.columns if c.startswith('V')]
if len(v_cols) > 0:
    v_train = X_train_fe[v_cols].copy()
    v_test = X_test_fe[v_cols].copy()
    medians = v_train.median()
    v_train.fillna(medians, inplace=True)
    v_test.fillna(medians, inplace=True)
    scaler_pca = StandardScaler()
    v_train_sc = scaler_pca.fit_transform(v_train)
    v_test_sc = scaler_pca.transform(v_test)
    pca = PCA(n_components=12, random_state=42)
    v_pca_train = pca.fit_transform(v_train_sc)
    v_pca_test = pca.transform(v_test_sc)
    
    for i in range(12):
        X_train_fe[f'V_PCA_{i}'] = v_pca_train[:, i]
        X_test_fe[f'V_PCA_{i}'] = v_pca_test[:, i]
        
    X_train_fe.drop(columns=v_cols, inplace=True)
    X_test_fe.drop(columns=v_cols, inplace=True)

cluster_cols = ['LogTransactionAmt', 'Transaction_Hour'] + [c for c in X_train_fe.columns if 'V_PCA_' in c]
clust_train = X_train_fe[cluster_cols].copy()
clust_test = X_test_fe[cluster_cols].copy()
c_medians = clust_train.median()
clust_train.fillna(c_medians, inplace=True)
clust_test.fillna(c_medians, inplace=True)
scaler_clust = StandardScaler()
c_train_sc = scaler_clust.fit_transform(clust_train)
c_test_sc = scaler_clust.transform(clust_test)
kmeans = MiniBatchKMeans(n_clusters=8, random_state=42, batch_size=2048)
X_train_fe['Behavior_Cluster'] = kmeans.fit_predict(c_train_sc).astype('category')
X_test_fe['Behavior_Cluster'] = kmeans.predict(c_test_sc).astype('category')

num_cols_smote = ['LogTransactionAmt', 'Transaction_Hour', 'Amt_Bin'] + [c for c in X_train_fe.columns if '_TE' in c or 'V_PCA_' in c]
X_train_smote = X_train_fe[num_cols_smote].copy()
X_test_smote = X_test_fe[num_cols_smote].copy()
smote_impute = X_train_smote.median()
X_train_smote.fillna(smote_impute, inplace=True)
X_test_smote.fillna(smote_impute, inplace=True)

smt = SMOTETomek(sampling_strategy=0.15, random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_train_smote, y_train)

lgb_train = lgb.Dataset(X_resampled, y_resampled)
lgb_val = lgb.Dataset(X_test_smote, y_test, reference=lgb_train)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 31,              
    'max_depth': 6,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'seed': 42,
    'verbose': -1
}
callbacks = [lgb.early_stopping(stopping_rounds=5, verbose=False)]
opt_clf = lgb.train(lgb_params, lgb_train, num_boost_round=10, valid_sets=[lgb_train, lgb_val], callbacks=callbacks)

print("TODO EL PIPELINE Y MODELO COMPILO SIN ERRORES!")
