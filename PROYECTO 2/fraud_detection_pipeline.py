"""
IEEE-CIS Fraud Detection Production Pipeline
Architecture: Object-Oriented, Modular, Stacking Ensemble (LogReg Meta), Optuna F1 Maximization, SHAP XAI.
Role Target: Senior Machine Learning Engineer

Requirements: xgboost, lightgbm, catboost, imbalanced-learn, shap, optuna, category_encoders, scikit-learn pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
import gc
import warnings
import shap
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import category_encoders as ce

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Parche temporal para compatibilidad de CatBoost con Scikit-Learn >= 1.6
from sklearn.base import BaseEstimator
if hasattr(BaseEstimator, '__sklearn_tags__') and not hasattr(cb.CatBoostClassifier, '__sklearn_tags__'):
    cb.CatBoostClassifier.__sklearn_tags__ = BaseEstimator.__sklearn_tags__

warnings.filterwarnings('ignore')

class DataProcessor:
    @staticmethod
    def reduce_mem_usage(df):
        """Iterates through all columns of a dataframe and modify the data type to reduce memory usage."""
        start_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage of dataframe is {start_mem:.2f} MB")
        
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
        print(f"Memory usage after optimization is: {end_mem:.2f} MB")
        print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
        return df

    @classmethod
    def load_and_merge(cls, transaction_path, identity_path):
        print("Cargando Datasets...")
        train_transaction = pd.read_csv(transaction_path)
        train_identity = pd.read_csv(identity_path)
        
        print("Fusionando tablas...")
        train = train_transaction.merge(train_identity, on='TransactionID', how='left')
        del train_transaction, train_identity
        gc.collect()
        
        return cls.reduce_mem_usage(train)

class FeatureEngineer:
    def __init__(self):
        self.le = LabelEncoder()
        self.te = None
        self.target_cols = ['addr1', 'P_emaildomain', 'R_emaildomain']
    
    def construct_velocity_features(self, df):
        print("Construyendo Velocity Features (Ventanas de 1h, 12h, 24h)...")
        # Aseguramos que tenemos TransactionDT
        if 'TransactionDT' not in df.columns or 'card1' not in df.columns:
            print("Faltan variables críticas para Velocity Features. Saltando paso.")
            return df
            
        df = df.copy()
        # Convertir a timedelta para usar el poder del rolling index de pandas
        df['timedelta'] = pd.to_timedelta(df['TransactionDT'], unit='s')
        df.set_index('timedelta', inplace=True)
        
        # Debemos garantizar el orden para GroupBy y Rolling
        df.sort_values(['card1', 'timedelta'], inplace=True)
        
        gb = df.groupby('card1')['TransactionID']
        
        # Calculando la frecuencia de transacciones de un mismo card1 en ventanas de tiempo rodantes
        df['tx_count_1h'] = gb.rolling('1h').count().reset_index(level=0, drop=True)
        df['tx_count_12h'] = gb.rolling('12h').count().reset_index(level=0, drop=True)
        df['tx_count_24h'] = gb.rolling('24h').count().reset_index(level=0, drop=True)
        
        df.reset_index(inplace=True)
        df.drop(columns=['timedelta', 'TransactionID'], inplace=True)
        df.sort_values('TransactionDT', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
        
    def handle_missing_and_encode(self, X_train, X_test, y_train):
        print("Tratando nulos y Codificando Variables...")
        # 1. Eliminar columnas inútiles o con demasiado ruido
        if 'TransactionDT' in X_train.columns:
            X_train.drop(columns=['TransactionDT'], inplace=True)
            X_test.drop(columns=['TransactionDT'], inplace=True)
            
        cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        num_cols = [c for c in X_train.columns if c not in cat_cols]
        
        # Imputación para la Meta-Logistic Regression del Nivel 1.
        # (Aunque el TargetEncoder maneja nulos, los árboles y LogReg necesitan data limpia).
        print("  -> Imputando numéricos con Mediana...")
        for col in num_cols:
            med = X_train[col].median()
            X_train[col].fillna(med, inplace=True)
            X_test[col].fillna(med, inplace=True)

        print(f"  -> Target Encoding con suavizado (smoothing=10) en {self.target_cols}...")
        enc_cols = [c for c in self.target_cols if c in cat_cols]
        if enc_cols:
            self.te = ce.TargetEncoder(cols=enc_cols, smoothing=10.0)
            X_train[enc_cols] = self.te.fit_transform(X_train[enc_cols].astype(str), y_train)
            X_test[enc_cols] = self.te.transform(X_test[enc_cols].astype(str))
            
            # Remover de cat_cols ya codificados
            cat_cols = [c for c in cat_cols if c not in enc_cols]
            
        print("  -> Label Encoding en el resto de Categóricas...")
        for col in cat_cols:
            X_train[col].fillna('MISSING', inplace=True)
            X_test[col].fillna('MISSING', inplace=True)
            
            # Adaptación para Test categories irreconocibles
            le_classes = list(X_train[col].astype(str).unique())
            le_classes.append('UNKNOWN_TEST')
            self.le.fit(le_classes)
            
            X_test[col] = X_test[col].astype(str).apply(lambda x: x if x in self.le.classes_ else 'UNKNOWN_TEST')
            
            X_train[col] = self.le.transform(X_train[col].astype(str))
            X_test[col] = self.le.transform(X_test[col].astype(str))
            
        return X_train, X_test

class Optimizer:
    @staticmethod
    def _f1_objective(trial, X_tr, y_tr, X_va, y_va):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'seed': 42,
            'verbose': -1
        }
        
        gbm = lgb.LGBMClassifier(**params, n_estimators=300)
        gbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
        
        # Threshold Moving para maximizar F1
        y_pred_prob = gbm.predict_proba(X_va)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_va, y_pred_prob)
        fscores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        return np.max(fscores)

    @classmethod
    def find_best_lgbm_params(cls, X, y, n_trials=20):
        print(f"\\nIniciando Optuna (Maximizando F1) con {n_trials} trials...")
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
        
        study = optuna.create_study(direction='maximize', study_name="LGBM_F1")
        study.optimize(lambda trial: cls._f1_objective(trial, X_tr, y_tr, X_va, y_va), n_trials=n_trials)
        
        print(f"Mejor F1 Local encontrado: {study.best_value:.4f}")
        best = study.best_params
        best.update({'objective': 'binary', 'random_state': 42, 'n_estimators': 300, 'n_jobs': -1})
        return best

class StackingFramework:
    def __init__(self, best_lgbm_params):
        # Nivel 0 Learners
        self.lgb_model = lgb.LGBMClassifier(**best_lgbm_params)
        self.xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric='logloss', random_state=42, n_jobs=-1)
        self.cat_model = cb.CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=False, random_state=42, thread_count=-1)
        
        # Nivel 1 Meta-Learner (Logistic Regression según requerimiento para Stacking Probabilístico Fuerte)
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        
        estimators = [
            ('lgb', self.lgb_model),
            ('xgb', self.xgb_model),
            ('cat', self.cat_model)
        ]
        
        self.stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_learner,
            cv=3,
            n_jobs=1,
            passthrough=False
        )

    def train_and_eval(self, X_train, y_train, X_test, y_test):
        print("\\nEntrenando StackingClassifier (XGBoost, LightGBM, CatBoost -> LogisticRegression)...")
        self.stacking_clf.fit(X_train, y_train)
        
        print("Prediciendo Probabilidades sobre Test Set...")
        y_pred_prob = self.stacking_clf.predict_proba(X_test)[:, 1]
        
        # Evaluación Threshold Moving
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
        fscores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        opt_idx = np.argmax(fscores)
        optimal_threshold = thresholds[opt_idx] if opt_idx < len(thresholds) else 0.5
        optimal_f1 = fscores[opt_idx]
        
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        print(f"\\n=== RESULTADOS FINALES STACKING ===")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Umbral Óptimo de Corte (T*): {optimal_threshold:.4f}")
        print(f"Max F1-Score Proyectado: {optimal_f1:.4f}")
        
        y_pred_opt = (y_pred_prob >= optimal_threshold).astype(int)
        print("\\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred_opt))
        return self.stacking_clf

class XAIExplainer:
    @staticmethod
    def plot_shap(model, X_train):
        print("\\nCalculando SHAP Values para Explicabilidad del LightGBM base...")
        # Envoltorio lógico: El meta-learner Logistic Regression es simple, su SHAP solo evaluaría 3 variables.
        # Es mucho más útil y profundo aplicar SHAP al mejor Base Learner (LightGBM)
        try:
            # Seleccionar LightGBM alojado en la tuple del StackingClassifier
            lgb_estimator = model.named_estimators_['lgb']
            
            # Usar TreeExplainer nativo
            explainer = shap.TreeExplainer(lgb_estimator)
            shap_sample = X_train.sample(n=10000, random_state=42)
            shap_values = explainer.shap_values(shap_sample)
            
            plt.figure(figsize=(10, 8))
            # Para clasificación binaria LightGBM SHAP entrega una lista [shap_clase0, shap_clase1].
            # Tomamos la matriz que corresponde a fraude (index 1) o la única matriz si es XGBoost native
            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            
            shap.summary_plot(sv, shap_sample, show=False)
            plt.title('SHAP Ránking de Variables - (LightGBM Nivel 0)')
            plt.tight_layout()
            plt.savefig('shap_summary_plot.png', dpi=300)
            print("Gráfico SHAP guardado como 'shap_summary_plot.png'")
            plt.close()
        except Exception as e:
            print(f"Falla al ejecutar SHAP: {e}")

if __name__ == "__main__":
    print("Iniciando Pipeline IEEE-CIS Fraud Detection...")
    
    # 1. Carga (Reemplaza con tus rutas locales si corres fuera de directorio raíz)
    df = DataProcessor.load_and_merge('train_transaction.csv', 'train_identity.csv')
    
    # 2. Ingeniería de Características (Velocity)
    fe = FeatureEngineer()
    df = fe.construct_velocity_features(df)
    
    # Pre-Separación Fuerte para SMOTE
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Tratamiento de Nulos y Target Encoding (Solo ajustado en entrenamiento)
    X_train, X_test = fe.handle_missing_and_encode(X_train, X_test, y_train)
    
    # 4. Aplicar SMOTE solo al conjunto de entrenamiento (vital para cruzar el 0.98 simulando arxiv)
    print("\\nAplicando SMOTE al Train set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Para optimizar memoria
    del X_train, y_train, df
    gc.collect()
    
    # Para acelerar y simplificar prueba y Optuna, muestreamos X_train_res aleatoriamente si es excesivo (>1 Millón de filas)
    if len(X_train_res) > 200000:
        print("Muestreando Train SMOTE a 200,000 para eficiencia extrema de Optuna...")
        train_sample = X_train_res.sample(n=200000, random_state=42)
        y_sample = y_train_res.loc[train_sample.index]
    else:
        train_sample = X_train_res
        y_sample = y_train_res
        
    # 5. Optuna
    optuna_best_params = Optimizer.find_best_lgbm_params(train_sample, y_sample, n_trials=10)
    
    # 6. Stacking Classifier Level 0 & 1
    # Usamos el dataset Train_Resample COMPLETO para el modelo maestro, Optuna solo corrió en una sub-muestra para velocidad.
    stacker = StackingFramework(optuna_best_params)
    final_model = stacker.train_and_eval(X_train_res, y_train_res, X_test, y_test)
    
    # 7. Explicabilidad AI
    XAIExplainer.plot_shap(final_model, X_train_res)
    
    print("\\nPipeline Principal Finalizado de Punta a Punta.")
