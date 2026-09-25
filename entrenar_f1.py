import sys
import io
import warnings
warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import joblib

print("🏎️ Iniciando Entrenamiento del Motor de IA para F1 (Pole, Ganador y Podio)...")

# 1. Generar Dataset Sintético de Entrenamiento basado en la Física y Telemetría de F1
np.random.seed(42)
n_samples = 3000

grid_position = np.random.randint(1, 21, n_samples)
driver_elo = np.random.uniform(1400, 2200, n_samples)
constructor_elo = np.random.uniform(1400, 2200, n_samples)
fp3_pace_diff = np.random.uniform(-1.5, 1.5, n_samples)

elo_diff = (driver_elo + constructor_elo) / 2.0 - 1700.0

# Labels: Pole, Winner, Podium
prob_pole = 1.0 / (1.0 + np.exp(-( -1.5 * (grid_position - 1) + 0.005 * elo_diff - 2.0 * fp3_pace_diff )))
y_pole = (np.random.rand(n_samples) < prob_pole).astype(int)

prob_win = 1.0 / (1.0 + np.exp(-( -0.8 * (grid_position - 1) + 0.004 * elo_diff - 1.2 * fp3_pace_diff )))
y_win = (np.random.rand(n_samples) < prob_win).astype(int)

prob_podium = 1.0 / (1.0 + np.exp(-( -0.5 * (grid_position - 1) + 0.003 * elo_diff - 0.8 * fp3_pace_diff )))
y_podium = (np.random.rand(n_samples) < prob_podium).astype(int)

X = pd.DataFrame({
    'GridPosition': grid_position,
    'DriverElo': driver_elo,
    'ConstructorElo': constructor_elo,
    'FP3PaceDiff': fp3_pace_diff
})

# Escalador
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Entrenar Clasificadores Calibrados
print("⚙️ Entrenando Modelo de Pole Position (Q3)...")
model_pole = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=42), cv=3)
model_pole.fit(X_scaled, y_pole)

print("⚙️ Entrenando Modelo de Ganador de Carrera (P1)...")
model_win = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=42), cv=3)
model_win.fit(X_scaled, y_win)

print("⚙️ Entrenando Modelo de Podio (Top 3)...")
model_podium = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=42), cv=3)
model_podium.fit(X_scaled, y_podium)

# 3. Guardar cerebros .pkl
joblib.dump(scaler, 'scaler_f1.pkl')
joblib.dump(model_pole, 'modelo_pole_f1.pkl')
joblib.dump(model_win, 'modelo_win_f1.pkl')
joblib.dump(model_podium, 'modelo_podio_f1.pkl')

print("🎉 ¡ENTRENAMIENTO F1 COMPLETADO CON ÉXITO!")
print("Cerebros guardados: scaler_f1.pkl, modelo_pole_f1.pkl, modelo_win_f1.pkl, modelo_podio_f1.pkl")
