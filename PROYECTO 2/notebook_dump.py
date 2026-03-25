
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
import warnings

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    precision_recall_curve, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ---

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memoria reducida de {start_mem:.2f} MB a {end_mem:.2f} MB')
    return df

print("Cargando datasets...")
train_transaction = pd.read_csv('train_transaction.csv')
train_identity = pd.read_csv('train_identity.csv')

train = train_transaction.merge(train_identity, on='TransactionID', how='left')
del train_transaction, train_identity
gc.collect()

train = reduce_mem_usage(train)
print(f"Dimensiones del Dataset Combinado: {train.shape}")

# ---

plt.figure(figsize=(6, 4))
ax = sns.countplot(data=train, x='isFraud')
plt.title('Distribución de isFraud (Fraude)')
plt.ylabel('Cantidad')
plt.xlabel('isFraud')
for p in ax.patches:
    ax.annotate(f'{p.get_height()} ({(p.get_height()/len(train))*100:.2f}%)', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')
plt.show()

# ---

df = train.copy()

# 1. Transformación Logarítmica
df['LogTransactionAmt'] = np.log1p(df['TransactionAmt'])

# 2. Características Temporales
import datetime
start_date = datetime.datetime.strptime("2017-12-01", "%Y-%m-%d")
df['Date'] = df['TransactionDT'].apply(lambda dt: start_date + datetime.timedelta(seconds=dt))
df['Transaction_Hour'] = df['Date'].dt.hour
df['Transaction_DayOfWeek'] = df['Date'].dt.dayofweek
df.drop(columns=['Date', 'TransactionDT'], inplace=True)

# 3. Agregaciones de Grupo (Mean / Std por tarjeta)
card_amt_mean = df.groupby('card1')['TransactionAmt'].mean().to_dict()
df['Card1_Amt_Mean'] = df['card1'].map(card_amt_mean)
# Desviación estandarizada relacional:
df['Card1_Amt_Std_Dev'] = df['TransactionAmt'] / df['Card1_Amt_Mean']

# 4. Frequency Encoding (Codificación por distribución acumulada)
categorical_features = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
for col in categorical_features:
    if col in df.columns:
        # Los nulos categóricos se mapean como "MISSING" para contabilizar dicha ausencia como una categoría probabilística más.
        df[col] = df[col].astype(str).fillna('MISSING')
        freq = df[col].value_counts(normalize=True).to_dict()
        df[col + '_Freq'] = df[col].map(freq)
        df.drop(columns=[col], inplace=True)

print("Ingeniería de Características Completada.")

# ---

null_percent = df.isnull().sum() / len(df)
cols_to_drop = null_percent[null_percent > 0.8].index.tolist()
df.drop(columns=cols_to_drop, inplace=True)
print(f"Se descartaron {len(cols_to_drop)} columnas (por contener >80% valores nulos).")
gc.collect()

# ---

v_cols = [c for c in df.columns if c.startswith('V')]

if len(v_cols) > 0:
    v_df = df[v_cols].copy()
    
    # Imputación puramente local para PCA (el df original los mantiene)
    v_df.fillna(v_df.median(), inplace=True)
    
    # Estandarización estrictamente requerida
    scaler = StandardScaler()
    v_scaled = scaler.fit_transform(v_df)
    
    pca = PCA(n_components=15, random_state=42)
    v_pca = pca.fit_transform(v_scaled)
    
    # Exploración visual de varianza explicada
    plt.figure(figsize=(8,4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada Acumulada')
    plt.title('Varianza Explicada (Subset Vesta V-Features)')
    plt.grid()
    plt.show()
    
    # Destrucción controlada de las V-columns puras y pegado de Componentes
    df.drop(columns=v_cols, inplace=True)
    
    for i in range(15):
        df[f'V_PCA_{i+1}'] = v_pca[:, i]
    
    print(f"Dimensionalidad destruida. Reducidas {len(v_cols)} columnas 'V' ruidosas en {15} componentes limpios (PCA).")
    del v_df, v_scaled, v_pca
    gc.collect()

# ---

# Sub-conjunto robusto de clustering (montos, perfiles PCA y horarios)
cluster_cols = ['TransactionAmt', 'LogTransactionAmt', 'Transaction_Hour'] + [c for c in df.columns if 'V_PCA_' in c]
cluster_df = df[cluster_cols].copy()

# Tratamiento de nulos de clustering
cluster_df.fillna(cluster_df.median(), inplace=True)

scaler_clust = StandardScaler()
clust_scaled = scaler_clust.fit_transform(cluster_df)

# Definimos K=5 grupos macro de fraude/cliente
kmeans = MiniBatchKMeans(n_clusters=5, random_state=42, batch_size=2048)
df['Cluster_ID'] = kmeans.fit_predict(clust_scaled)

# Ploteamos influencia y correlación
fraud_rates = df.groupby('Cluster_ID')['isFraud'].mean() * 100
plt.figure(figsize=(6, 4))
sns.barplot(x=fraud_rates.index, y=fraud_rates.values, color='lightcoral')
plt.title('Penetración del Fraude por Segmentación Comportamental (KMeans)')
plt.ylabel('% de Fraude Confirmado')
plt.xlabel('Identificador del Cluster')
plt.show()

# Conversión semántica (lightgbm adora los 'Category')
df['Cluster_ID'] = df['Cluster_ID'].astype('category')

del cluster_df, clust_scaled
gc.collect()

# ---

X = df.drop(columns=['TransactionID', 'isFraud'])

# Castear todos los features "objects" a categoricos nativos (DeviceType, DeviceInfo)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

y = df['isFraud']

# División Test-Train garantizando estrato proporcional (3.49%) (Stratify)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Forma Matriz Entrenamiento: {X_train.shape}")
print(f"Forma Matriz Validación Interna: {X_val.shape}")

# ---

ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
print(f"Coeficiente Multiplicador 'scale_pos_weight': {ratio:.2f}")

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Diccionario de Parámetros Avanzados LightGBM
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,
    'num_leaves': 128,              # Limite de ramificaciones incrementado para mapeos minuciosos
    'max_depth': -1,
    'scale_pos_weight': ratio,      # Auto-ajuste Fuerte del Desbalance
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'verbose': -1
}

# Empleando una métrica de Parada Temprana
callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]

# Ajustando al modelo iterativamente
print("Comenzando Entrenamiento Supervisado Profundo...")
clf = lgb.train(
    params, 
    lgb_train, 
    num_boost_round=1000, 
    valid_sets=[lgb_train, lgb_val],
    callbacks=callbacks
)

# ---

# 1. Obtención de Probabilidades en Crudo (Raw Probabilities)
y_pred_prob = clf.predict(X_val)

roc_auc = roc_auc_score(y_val, y_pred_prob)

# 2. Reconstrucción de la Curva Precision-Recall 
precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_prob)

# 3. Formulación de los hiper-F1
fscores = (2 * precisions * recalls) / (precisions + recalls + 1e-10) # 1e-10 previene zero dev
optimal_idx = np.argmax(fscores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
optimal_f1 = fscores[optimal_idx]

print(f"-> ROC-AUC del modelo              : {roc_auc:.4f}")
print(f"-> Umbral de Decisión Optimo (T*)  : {optimal_threshold:.4f}")
print(f"-> Mejor F1-Score Proyectable      : {optimal_f1:.4f}")

plt.figure(figsize=(10,5))
plt.plot(thresholds, fscores[:-1], "g-", alpha=0.8, label="F1-Score Trajectory")
plt.plot(thresholds, precisions[:-1], "b--", alpha=0.5, label="Precision")
plt.plot(thresholds, recalls[:-1], "r--", alpha=0.5, label="Recall")
plt.axvline(x=optimal_threshold, color='black', linestyle=':', linewidth=2, label=f'Umbral Crítico T* = {optimal_threshold:.3f}')
plt.xlabel("Límite Probabilístico de Decisión (Threshold)")
plt.ylabel("Scores")
plt.title("Threshold Moving: Maximización de la Métrica F1")
plt.legend()
plt.grid()
plt.show()

# ---

# Refinamos las etiquetas binarizadoras con nuestra T*
y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)

cm = confusion_matrix(y_val, y_pred_optimal)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', cbar=False)
plt.title('Matriz de Confusión (Threshold Optimo T*)')
plt.ylabel('Etiquetas Auténticas')
plt.xlabel('Diagnóstico Computacional')
plt.show()

print("============= REPORTE FINAL DE AUDITORÍA CLASIFICATORIA =============")
print(classification_report(y_val, y_pred_optimal))

# ---
