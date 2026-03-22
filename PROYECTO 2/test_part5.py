import pandas as pd
import numpy as np

# Mock data
np.random.seed(42)
df = pd.DataFrame({
    'TransactionAmt': [10.5, 50.0, np.nan, 100.0, 500.0],
    'TransactionDT': [86400, 86401, 86402, 86403, 86404],
    'P_emaildomain': ['gmail.com', 'yahoo.com', np.nan, 'gmail.com', 'hotmail.com'],
    'R_emaildomain': [np.nan, 'gmail.com', 'yahoo.com', 'yahoo.com', 'yahoo.com'],
    'card1': [1000, 2000, 3000, 1000, 2000],
    'card2': [100.0, 200.0, 300.0, np.nan, 200.0],
    'isFraud': [0, 1, 0, 0, 1]
})

X_train = df.drop(columns=['isFraud'])
y_train = df['isFraud']
X_test = X_train.copy()

def engineer_features(X_curr, y_curr=None, fit_params=None):
    df_feat = X_curr.copy()
    is_train = y_curr is not None
    
    if is_train:
        fit_params = {}
    
    # --- Temporales ---
    import datetime
    start_date = datetime.datetime.strptime("2017-12-01", "%Y-%m-%d")
    dates = df_feat['TransactionDT'].apply(lambda dt: start_date + datetime.timedelta(seconds=dt))
    df_feat['Transaction_Hour'] = dates.dt.hour
    df_feat['Transaction_DayOfWeek'] = dates.dt.dayofweek
    df_feat.drop(columns=['TransactionDT'], inplace=True)
    
    # --- Binning (Discretización) de TransactionAmt ---
    if is_train:
        # Extraemos los cuantiles que dictarán los "cortes" para binning
        bins = [-np.inf, np.percentile(df_feat['TransactionAmt'].dropna(), 25), 
                np.percentile(df_feat['TransactionAmt'].dropna(), 75), np.inf]
        fit_params['amt_bins'] = bins
    df_feat['Amt_Bin'] = pd.cut(df_feat['TransactionAmt'], bins=fit_params['amt_bins'], labels=[0,1,2]).astype('float16')
    df_feat['LogTransactionAmt'] = np.log1p(df_feat['TransactionAmt'])
    
    # --- Target Encoding (Solo calcular promedios en TRAIN) ---
    high_card_cat = ['P_emaildomain', 'R_emaildomain', 'card1', 'card2']
    for col in high_card_cat:
        if col in df_feat.columns:
            # Llenar ausencias provisionalmente
            df_feat[col] = df_feat[col].fillna(df_feat[col].mode()[0] if not df_feat[col].mode().empty else 'MISSING')
            
            if is_train:
                # Calculamos media agrupando
                temp_df = pd.DataFrame({col: df_feat[col], 'target': y_curr})
                means = temp_df.groupby(col)['target'].mean().to_dict()
                global_mean = y_curr.mean()
                fit_params[f'{col}_te'] = (means, global_mean)
                
            means, global_mean = fit_params[f'{col}_te']
            df_feat[col + '_TE'] = df_feat[col].map(means).fillna(global_mean) # fillna(global_mean) con valores nunca vistos en test
            df_feat.drop(columns=[col], inplace=True)

    # Las demás catebóricas (nominales de muy rala cardinalidad) a 'category' nativo para LightGBM
    for c in df_feat.select_dtypes(include=['object']).columns:
        df_feat[c] = df_feat[c].astype('category')
        
    return df_feat, fit_params

print("Aplicando ingeniería de características previniendo fuga...")
X_train_fe, pipeline_params = engineer_features(X_train, y_train)
X_test_fe, _ = engineer_features(X_test, fit_params=pipeline_params)
print("Hecho.")
