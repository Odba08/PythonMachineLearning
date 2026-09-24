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

# 4. Diccionario de Ligas y Torneos a compilar (13 Competiciones Top)
ligas = {
    'E1': 'Championship',
    'E0': 'Premier',
    'SP1': 'LaLiga',
    'D1': 'Bundesliga',
    'I1': 'SerieA',
    'F1': 'Ligue1',
    'P1': 'Portugal',
    'N1': 'Eredivisie',
    'T1': 'SuperLig',
    'B1': 'Brasileirao',
    'EC': 'Champions',
    'ARG': 'Libertadores',
}

# 5. Entrenamiento iterativo con Calibración de Probabilidades
for codigo, nombre in ligas.items():
    print(f"\n🧠 Entrenando y calibrando microservicios para: {nombre}...")
    if nombre == 'Libertadores':
        df_liga = df[df['Division'].isin(['B1', 'BRA', 'ARG', 'COP'])].copy()
        if len(df_liga) < 50:
            df_liga = df.copy()
    else:
        df_liga = df[df['Division'] == codigo].copy()
    
    if len(df_liga) < 50:
        df_liga = df.copy()

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

# Entrenar modelo General (utilizado para Saudi y fallback de nuevas ligas)
print("\n🧠 Entrenando y calibrando microservicio General (Saudi Fallback)...")
X_gen = df[features]
scaler_gen = StandardScaler()
X_gen_scaled = scaler_gen.fit_transform(X_gen)

base_1x2_gen = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
modelo_1x2_gen = CalibratedClassifierCV(estimator=base_1x2_gen, cv=5)
modelo_1x2_gen.fit(X_gen_scaled, df['Target_1X2'])

base_goles_gen = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
modelo_goles_gen = CalibratedClassifierCV(estimator=base_goles_gen, cv=5)
modelo_goles_gen.fit(X_gen_scaled, df['Target_O25'])

base_btts_gen = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
modelo_btts_gen = CalibratedClassifierCV(estimator=base_btts_gen, cv=5)
modelo_btts_gen.fit(X_gen_scaled, df['Target_BTTS'])

joblib.dump(scaler_gen, 'scaler_Saudi.pkl')
joblib.dump(modelo_1x2_gen, 'modelo_1x2_Saudi.pkl')
joblib.dump(modelo_goles_gen, 'modelo_goles_Saudi.pkl')
joblib.dump(modelo_btts_gen, 'modelo_btts_Saudi.pkl')
print("✅ Saudi (General Model) compilada con éxito.")

print("\n🚀 Proceso finalizado. Binarios de IA generados correctamente para las 11 Ligas.")