import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

app = FastAPI(title="Motor Cuantitativo Multi-Liga")

# 1. Cargamos TODOS los modelos en memoria al arrancar el servidor
ligas_soportadas = ['Championship', 'Premier', 'LaLiga', 'Bundesliga', 'SerieA']
motores = {}

print("Cargando cerebros en memoria...")
for liga in ligas_soportadas:
    try:
        motores[liga] = {
            'scaler': joblib.load(f'scaler_{liga}.pkl'),
            'modelo_1x2': joblib.load(f'modelo_1x2_{liga}.pkl'),
            'modelo_goles': joblib.load(f'modelo_goles_{liga}.pkl'),
            'modelo_btts': joblib.load(f'modelo_btts_{liga}.pkl')
        }
    except FileNotFoundError:
        print(f"⚠️ Faltan archivos de la liga: {liga}")

# 2. DTO Actualizado: Ahora exigimos saber qué liga es
class PartidoData(BaseModel):
    Liga: str  # Ejs: "Premier", "LaLiga", "Championship"
    HomeTeam: str
    AwayTeam: str
    HomeElo: float
    AwayElo: float
    Form5Home: int
    Form5Away: int
    Form3Home: int
    Form3Away: int

import math
from scipy.stats import poisson

def calcular_poisson_marcadores(home_elo: float, away_elo: float, form5_diff: float):
    elo_diff = home_elo - away_elo
    lambda_home = max(0.3, min(4.2, 1.40 + (elo_diff / 350.0) + (form5_diff / 250.0)))
    mu_away = max(0.2, min(4.2, 1.10 - (elo_diff / 350.0) - (form5_diff / 250.0)))
    
    scores = []
    for h in range(5):
        for a in range(5):
            prob = poisson.pmf(h, lambda_home) * poisson.pmf(a, mu_away)
            scores.append({"marcador": f"{h}-{a}", "prob_pct": round(prob * 100, 2), "raw": prob})
            
    scores.sort(key=lambda x: x["raw"], reverse=True)
    top3 = [{"marcador": s["marcador"], "probabilidad": f"{s['prob_pct']}%"} for s in scores[:3]]
    return round(lambda_home, 2), round(mu_away, 2), top3

# 3. Endpoint Dinámico
@app.post("/analizar-completo")
def analizar_completo(partidos: List[PartidoData]):
    resultados = []
    
    for p in partidos:
        if p.Liga not in motores:
            resultados.append({"partido": f"{p.HomeTeam} vs {p.AwayTeam}", "error": f"Liga '{p.Liga}' no soportada."})
            continue
            
        # Seleccionamos el motor correcto para este partido
        motor = motores[p.Liga]
        
        df = pd.DataFrame([p.dict()])
        df['Elo_Diff'] = df['HomeElo'] - df['AwayElo']
        df['Form5_Diff'] = df['Form5Home'] - df['Form5Away']
        
        features = df[['Elo_Diff', 'Form5_Diff', 'Form3Home', 'Form3Away']]
        features_scaled = motor['scaler'].transform(features)
        
        # Predicciones
        prob_1x2 = motor['modelo_1x2'].predict_proba(features_scaled)[0]
        prob_goles = motor['modelo_goles'].predict_proba(features_scaled)[0]
        prob_btts = motor['modelo_btts'].predict_proba(features_scaled)[0]
        
        p_visitante, p_empate, p_local = prob_1x2[0], prob_1x2[1], prob_1x2[2]
        p_under, p_over = prob_goles[0], prob_goles[1]
        p_btts_no, p_btts_si = prob_btts[0], prob_btts[1]
        
        # Poisson Math Model
        xg_home, xg_away, marcadores_top = calcular_poisson_marcadores(p.HomeElo, p.AwayElo, df['Form5_Diff'].iloc[0])
        
        resultado = {
            "partido": f"{p.HomeTeam} vs {p.AwayTeam}",
            "liga": p.Liga,
            "xg_esperados": {
                "xg_local": xg_home,
                "xg_visitante": xg_away
            },
            "probabilidades_1X2": {
                "Victoria_Local": f"{round(p_local * 100, 2)}%",
                "Empate": f"{round(p_empate * 100, 2)}%",
                "Victoria_Visitante": f"{round(p_visitante * 100, 2)}%"
            },
            "doble_oportunidad": {
                "1X": f"{round((p_local + p_empate) * 100, 2)}%",
                "X2": f"{round((p_visitante + p_empate) * 100, 2)}%",
                "12": f"{round((p_local + p_visitante) * 100, 2)}%"
            },
            "mercado_goles": {
                "Over_2_5": f"{round(p_over * 100, 2)}%",
                "Under_2_5": f"{round(p_under * 100, 2)}%"
            },
            "ambos_anotan": {
                "Si": f"{round(p_btts_si * 100, 2)}%",
                "No": f"{round(p_btts_no * 100, 2)}%"
            },
            "marcadores_exactos": marcadores_top
        }
        resultados.append(resultado)
        
    return {"analisis_multiliga": resultados}