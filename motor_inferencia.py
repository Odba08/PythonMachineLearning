import joblib
import pandas as pd
import sys

# 1. Carga de los binarios en memoria (Tus archivos de la Championship)
try:
    modelo = joblib.load('modelo_e1_v1.pkl')
    scaler = joblib.load('scaler_e1_v1.pkl')
except FileNotFoundError:
    print("Error: Los archivos .pkl no están en la carpeta.")
    sys.exit()

# 2. El Payload (Simulando datos que enviará NestJS)
# Ejemplo: Leeds United vs Sunderland (Championship)
partido_hoy = {
    'HomeTeam': 'Leeds United',
    'AwayTeam': 'Sunderland',
    'HomeElo': 1750,
    'AwayElo': 1600,
    'Form5Home': 12, # Puntos en los últimos 5 partidos
    'Form5Away': 7,
    'Form3Home': 7,  # Puntos en los últimos 3 partidos
    'Form3Away': 4,
    'MaxHome': 1.85  # Cuota de la casa de apuestas por la victoria del Leeds
}

# 3. Transformación de datos (Ingeniería de Características en tiempo real)
df_input = pd.DataFrame([partido_hoy])
df_input['Elo_Diff'] = df_input['HomeElo'] - df_input['AwayElo']
df_input['Form5_Diff'] = df_input['Form5Home'] - df_input['Form5Away']

# 4. Aislamos las variables que la IA sabe leer
features = df_input[['Elo_Diff', 'Form5_Diff', 'Form3Home', 'Form3Away']]

# 5. Normalizamos los números y ejecutamos la predicción
features_scaled = scaler.transform(features)
prob_ia = modelo.predict_proba(features_scaled)[0][1] # Probabilidad real calculada
prob_mercado = 1 / partido_hoy['MaxHome']             # Probabilidad de la casa

print(f"🏟️ PARTIDO: {partido_hoy['HomeTeam']} vs {partido_hoy['AwayTeam']}")
print("-" * 40)
print(f"Probabilidad de la Casa (Cuota {partido_hoy['MaxHome']}): {prob_mercado:.2%}")
print(f"Probabilidad del Algoritmo: {prob_ia:.2%}")

# 6. Lógica de Arbitraje: Exigimos un 5% de ventaja matemática (+EV)
margen_ventaja = prob_ia - prob_mercado

if margen_ventaja > 0.05:
    print(f"🟢 ALERTA: Apuesta con valor encontrada. Ventaja del {margen_ventaja:.2%}. Ejecutar operación.")
else:
    print("🔴 DESCARTAR: La cuota es demasiado baja. No hay ventaja matemática.")