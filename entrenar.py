import sys
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("⚙️ Iniciando compilación en lote para 5 ligas (Multitarea)...")

# 1. Cargar datos
try:
    df = pd.read_csv('Matches.csv')
    df = df.dropna(subset=['HomeElo', 'AwayElo', 'Form5Home', 'Form5Away', 'Form3Home', 'Form3Away', 'FTHome', 'FTAway'])
except FileNotFoundError:
    print("❌ Error: No se encontró 'Matches.csv'.")
    exit()

# 2. Ingeniería de Características Base
df['Elo_Diff'] = df['HomeElo'] - df['AwayElo']
df['Form5_Diff'] = df['Form5Home'] - df['Form5Away']

# 3. Targets (1X2, Goles y Ambos Anotan - BTTS)
def definir_1x2(fila):
    if fila['FTHome'] > fila['FTAway']: return 2
    elif fila['FTHome'] == fila['FTAway']: return 1
    else: return 0

df['Target_1X2'] = df.apply(definir_1x2, axis=1)
df['Target_O25'] = ((df['FTHome'] + df['FTAway']) > 2.5).astype(int)
df['Target_BTTS'] = ((df['FTHome'] > 0) & (df['FTAway'] > 0)).astype(int)

features = ['Elo_Diff', 'Form5_Diff', 'Form3Home', 'Form3Away']

# 4. Diccionario de Ligas a compilar
ligas = {
    'E1': 'Championship',
    'E0': 'Premier',
    'SP1': 'LaLiga',
    'D1': 'Bundesliga',
    'I1': 'SerieA'
}

# 5. Entrenamiento iterativo con Calibración de Probabilidades
for codigo, nombre in ligas.items():
    print(f"\n🧠 Entrenando y calibrando microservicios para: {nombre}...")
    df_liga = df[df['Division'] == codigo].copy()
    
    if len(df_liga) < 50:
        print(f"⚠️ Pocos datos para {nombre} ({len(df_liga)} filas), saltando...")
        continue

    X = df_liga[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Modelo 1X2 Calibrado
    base_1x2 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    modelo_1x2 = CalibratedClassifierCV(estimator=base_1x2, cv=5)
    modelo_1x2.fit(X_scaled, df_liga['Target_1X2'])
    
    # Modelo Goles (Over/Under 2.5) Calibrado
    base_goles = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    modelo_goles = CalibratedClassifierCV(estimator=base_goles, cv=5)
    modelo_goles.fit(X_scaled, df_liga['Target_O25'])

    # Modelo BTTS (Ambos Anotan) Calibrado
    base_btts = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    modelo_btts = CalibratedClassifierCV(estimator=base_btts, cv=5)
    modelo_btts.fit(X_scaled, df_liga['Target_BTTS'])
    
    # Evaluación básica
    acc_1x2 = accuracy_score(df_liga['Target_1X2'], modelo_1x2.predict(X_scaled))
    acc_o25 = accuracy_score(df_liga['Target_O25'], modelo_goles.predict(X_scaled))
    acc_btts = accuracy_score(df_liga['Target_BTTS'], modelo_btts.predict(X_scaled))
    print(f"   📊 Métricas {nombre} -> Acc 1X2: {acc_1x2:.2%}, Acc O/U 2.5: {acc_o25:.2%}, Acc BTTS: {acc_btts:.2%}")

    # Exportación binaria
    joblib.dump(scaler, f'scaler_{nombre}.pkl')
    joblib.dump(modelo_1x2, f'modelo_1x2_{nombre}.pkl')
    joblib.dump(modelo_goles, f'modelo_goles_{nombre}.pkl')
    joblib.dump(modelo_btts, f'modelo_btts_{nombre}.pkl')
    print(f"✅ {nombre} compilada con éxito (1X2, Over2.5, BTTS).")

print("\n🚀 Proceso finalizado. Binarios de IA generados correctamente.")