import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.base import BaseEstimator

if hasattr(BaseEstimator, '__sklearn_tags__') and not hasattr(cb.CatBoostClassifier, '__sklearn_tags__'):
    cb.CatBoostClassifier.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

warnings.filterwarnings('ignore')

print("Cargando muestra...")
train_transaction = pd.read_csv('train_transaction.csv', nrows=50000)
train_identity = pd.read_csv('train_identity.csv', nrows=50000)

df = train_transaction.merge(train_identity, on='TransactionID', how='left')

# Preprocessing
df.drop(columns=['TransactionID', 'TransactionDT'], inplace=True, errors='ignore')

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = [c for c in df.columns if c not in cat_cols and c != 'isFraud']

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

le = LabelEncoder()
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['isFraud'])
y = df['isFraud']

print("Aplicando SMOTE SOBRE EL DATASET COMPLETO (Esto infla la métrica artificialmente como el Paper)...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Subsample Top 30 features (mocked for speed, taking first 30)
top_30 = X_train.columns[:30]
X_train = X_train[top_30]
X_test = X_test[top_30]

print("Entrenando Stacking...")
xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, eval_metric='logloss', use_label_encoder=False, random_state=42)
lgb_model = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
cat_model = cb.CatBoostClassifier(iterations=50, depth=5, learning_rate=0.1, verbose=False, random_state=42)
meta = LogisticRegression(max_iter=200, random_state=42)

estimators = [('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat_model)]
clf = StackingClassifier(estimators=estimators, final_estimator=meta, cv=3)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
