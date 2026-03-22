import pandas as pd
import numpy as np

df = pd.DataFrame({
    'TransactionAmt': [10.5, 50.0, np.nan, 100.0, 500.0, 50.0, 50.0, np.inf, 10.5, 999.0],
    'TransactionDT': [86400, 86401, 86402, 86403, 86404, 86405, 86406, 86407, 86408, 86409],
    'TransactionAmt_Capped': [10.5, 50.0, np.nan, 100.0, 150.0, 50.0, 50.0, 150.0, 10.5, 150.0],
    'P_emaildomain': ['gmail.com', 'yahoo.com', np.nan, 'gmail.com', 'hotmail.com', 'gmail.com', 'gmail.com', np.nan, 'yahoo.com', 'hotmail.com'],
    'R_emaildomain': [np.nan, 'gmail.com', 'yahoo.com', 'yahoo.com', 'yahoo.com', np.nan, np.nan, 'gmail.com', np.nan, 'gmail.com'],
    'card1': [1000, 2000, 3000, 1000, 2000, 1000, 1000, 1000, 2000, 3000],
    'card2': [100.0, 200.0, 300.0, np.nan, 200.0, 100.0, 100.0, 100.0, 200.0, 300.0],
    'isFraud': [0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
})
for col in df.columns:
    if df[col].dtype == np.float64:
        df[col] = df[col].astype(np.float32)

X_train = df.drop(columns=['isFraud'])
y_train = df['isFraud']
X_test = X_train.copy()

def apply_feature_engineering(X_curr, y_curr=None, fit_params=None):
    df_feat = X_curr.copy()
    is_train = y_curr is not None
    if is_train: fit_params = {}
    
    # 1. Variables Temporales
    import datetime
    start_date = datetime.datetime.strptime("2017-12-01", "%Y-%m-%d")
    dates = df_feat['TransactionDT'].apply(lambda dt: start_date + datetime.timedelta(seconds=dt))
    df_feat['Transaction_Hour'] = dates.dt.hour
    df_feat.drop(columns=['TransactionDT'], inplace=True)
    
    # 2. Binning y Transformaciones (Garantizamos float32 para indexación pd.cut en VESTA)
    if is_train:
        clean_amt = df_feat['TransactionAmt'].astype('float32').replace([np.inf, -np.inf], np.nan).dropna()
        _, bins = pd.qcut(clean_amt, q=4, retbins=True, duplicates='drop')
        bins[0] = -np.inf
        bins[-1] = np.inf
        fit_params['amt_bins'] = bins
    df_feat['Amt_Bin'] = pd.cut(df_feat['TransactionAmt'].astype('float32'), 
                                bins=fit_params['amt_bins'], labels=False, include_lowest=True).astype('float32')
    df_feat['LogTransactionAmt'] = np.log1p(df_feat['TransactionAmt_Capped'].astype('float32'))
    
    # 3. Target Encoding para Alta Cardinalidad
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
            df_feat[col + '_TE'] = df_feat[col].map(means).fillna(global_mean) # fillna con global para no-vistos
            df_feat.drop(columns=[col], inplace=True)

    # El remanente categórico a tipo Categorical para LightGBM
    for c in df_feat.select_dtypes(include=['object']).columns:
        df_feat[c] = df_feat[c].astype('category')
        
    return df_feat, fit_params

print("Aplicando ingeniería de características...")
X_train_fe, prep_params = apply_feature_engineering(X_train, y_train)
X_test_fe, _ = apply_feature_engineering(X_test, fit_params=prep_params)
print("Ingeniería y Target Encoding completados sin ERRORES.")
