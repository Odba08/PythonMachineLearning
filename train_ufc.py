import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("🥊 FASE 1 & 2: ENTRENAMIENTO DEL MODELO PREDICTIVO UFC (+EV) 🥊")

# 1. Cargar Dataset Histórico
df = pd.read_csv('data/ufc-master.csv')
print(f"Total peleas en ufc-master.csv: {len(df)}")

# Filtrar solo peleas concluidas con ganador Red o Blue
df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
df['target'] = (df['Winner'] == 'Red').astype(int)

# Definir las 15 características diferenciales clave
feature_cols = [
    'reach_dif',
    'height_dif',
    'age_dif',
    'sig_str_dif',
    'avg_td_dif',
    'avg_sub_att_dif',
    'win_streak_dif',
    'longest_win_streak_dif',
    'lose_streak_dif',
    'win_dif',
    'loss_dif',
    'total_round_dif',
    'total_title_bout_dif',
    'ko_dif',
    'sub_dif'
]

print(f"Características seleccionadas ({len(feature_cols)}):", feature_cols)

# Limpieza de nulos con medianas
for col in feature_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    else:
        df[col] = 0.0

X = df[feature_cols]
y = df['target']

print(f"Datos listos para entrenamiento: {X.shape[0]} combates.")
print(f"Distribución de victorias: Rojo={y.sum()} ({round(y.mean()*100, 1)}%) | Azul={(1-y).sum()} ({round((1-y.mean())*100, 1)}%)")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensemble robusto: GradientBoosting + RandomForest
base_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42)
base_rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=4, random_state=42)

ensemble = VotingClassifier(
    estimators=[('gb', base_gb), ('rf', base_rf)],
    voting='soft'
)

# Calibración de probabilidades (Isotonic / Sigmoid) para cálculo riguroso de +EV
calibrated_model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=5)
calibrated_model.fit(X_train_scaled, y_train)

# Evaluación
y_pred = calibrated_model.predict(X_test_scaled)
y_prob = calibrated_model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)

print("\n📊 RESULTADOS DE VALIDACIÓN DEL MODELO:")
print(f"  • Exactitud (Accuracy): {round(acc * 100, 2)}%")
print(f"  • ROC-AUC Score:        {round(auc, 4)}")
print(f"  • Brier Score:          {round(brier, 4)} (Métrica clave para calibración de cuotas)")

# Guardar modelos y artefactos
joblib.dump(calibrated_model, 'modelo_ufc_ganador.pkl')
joblib.dump(scaler, 'scaler_ufc.pkl')
joblib.dump(feature_cols, 'features_ufc.pkl')

print("\n✅ Archivos guardados exitosamente:")
print("  • modelo_ufc_ganador.pkl")
print("  • scaler_ufc.pkl")
print("  • features_ufc.pkl")
