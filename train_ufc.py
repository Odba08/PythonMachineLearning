import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("🥊 ENTRENAMIENTO AVANZADO DEL MODELO PREDICTIVO UFC (+EV CUANTITATIVO) 🥊")

# 1. Cargar Dataset Histórico
df = pd.read_csv('data/ufc-master.csv')
print(f"Total peleas en ufc-master.csv: {len(df)}")

# Filtrar solo peleas concluidas con ganador Red o Blue
df = df[df['Winner'].isin(['Red', 'Blue'])].copy()
df['target'] = (df['Winner'] == 'Red').astype(int)

# 2. Ingeniería de variables avanzadas
light_classes = [
    'Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight',
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight"
]
df['is_light'] = df['weight_class'].isin(light_classes).astype(int)

# A. Maldición de la edad 35+ en categorías ligeras (Diferencial Blue - Red)
df['r_age_curse'] = ((df['R_age'] >= 35) & (df['is_light'] == 1)).astype(int)
df['b_age_curse'] = ((df['B_age'] >= 35) & (df['is_light'] == 1)).astype(int)
df['age_curse_dif'] = df['b_age_curse'] - df['r_age_curse']

# B. Diferencial de precisión de golpeo y derribo
df['td_pct_dif'] = df['B_avg_TD_pct'].fillna(0.3) - df['R_avg_TD_pct'].fillna(0.3)
df['sig_str_pct_dif'] = df['B_avg_SIG_STR_pct'].fillna(0.4) - df['R_avg_SIG_STR_pct'].fillna(0.4)

# C. Ratio de volumen ofensivo total (Str landed * accuracy)
df['r_str_power'] = df['R_avg_SIG_STR_landed'].fillna(3.0) * df['R_avg_SIG_STR_pct'].fillna(0.4)
df['b_str_power'] = df['B_avg_SIG_STR_landed'].fillna(3.0) * df['B_avg_SIG_STR_pct'].fillna(0.4)
df['str_power_dif'] = df['b_str_power'] - df['r_str_power']

# 3. Definir las 19 características diferenciales clave
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
    'sub_dif',
    # 4 Nuevas variables cuantitativas avanzadas
    'age_curse_dif',
    'td_pct_dif',
    'sig_str_pct_dif',
    'str_power_dif'
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

# Ensemble robusto: GradientBoosting + RandomForest + HistGradientBoosting
base_gb = GradientBoostingClassifier(n_estimators=180, learning_rate=0.04, max_depth=3, random_state=42)
base_rf = RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=3, random_state=42)
base_hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.04, max_depth=4, random_state=42)

ensemble = VotingClassifier(
    estimators=[('gb', base_gb), ('rf', base_rf), ('hgb', base_hgb)],
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

print("\n📊 RESULTADOS DE VALIDACIÓN DEL MODELO AVANZADO:")
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
