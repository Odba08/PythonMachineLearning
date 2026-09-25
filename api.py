import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

app = FastAPI(title="Motor Cuantitativo Multi-Liga")

# 1. Cargamos TODOS los modelos en memoria al arrancar el servidor
ligas_soportadas = [
    'Championship', 'Premier', 'LaLiga', 'Bundesliga', 'SerieA',
    'Ligue1', 'Portugal', 'Eredivisie', 'SuperLig', 'Brasileirao', 'Saudi',
    'Champions', 'Libertadores'
]
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
            "marcadores_exactos": marcadores_top
        }
        resultados.append(resultado)
        
    return {"analisis_multiliga": resultados}

# ------------------------------------------------------------------
# 4. MOTOR CUANTITATIVO F1 (FASTF1 + JOLPICA 2026)
# ------------------------------------------------------------------
import fastf1
import numpy as np
import urllib.request
import json
import time

f1_cache_time = 0
f1_cached_result = None

from datetime import datetime, timezone

def ejecutar_simulacion_f1_real(year=2026, gp='Azerbaijan'):
    global f1_cache_time, f1_cached_result
    ahora_ts = time.time()
    if f1_cached_result and (ahora_ts - f1_cache_time < 300):
        return f1_cached_result

    # 1. Detección dinámica de sesiones del fin de semana (FP1, FP2, FP3, Sprint, Qualy)
    event = fastf1.get_event(year, gp)
    ahora_utc = datetime.now(timezone.utc)
    
    sesiones_a_cargar = []
    qualy_completada = False
    qualy_session_obj = None

    for i in range(1, 6):
        s_name = event[f'Session{i}']
        s_date = event[f'Session{i}DateUtc'] if 'Session{i}DateUtc' in event else event[f'Session{i}Date']
        # Si la sesión ya comenzó según el calendario oficial FIA
        if s_date <= ahora_utc:
            sesiones_a_cargar.append((i, s_name))

    driver_telemetry = {}
    ultima_sesion = "FP1"

    for idx, s_name in sesiones_a_cargar:
        try:
            s = fastf1.get_session(year, gp, s_name)
            s.load(laps=True, telemetry=False, weather=False)
            if hasattr(s, 'laps') and len(s.laps) > 0:
                ultima_sesion = s_name
                if 'Qualifying' in s_name or s_name == 'Q':
                    qualy_completada = True
                    qualy_session_obj = s

                fastest_list = []
                for drv in s.drivers:
                    laps = s.laps.pick_drivers(drv)
                    if not laps.empty:
                        fl = laps.pick_fastest()
                        if fl is not None and not pd.isna(fl['LapTime']):
                            info = s.get_driver(drv)
                            fastest_list.append({
                                'code': info['Abbreviation'],
                                'name': info['FullName'],
                                'team': info['TeamName'],
                                'time': fl['LapTime'].total_seconds(),
                            })
                df_s = pd.DataFrame(fastest_list).sort_values('time').reset_index(drop=True)
                if not df_s.empty:
                    best_t = df_s.iloc[0]['time']
                    for pos, r in df_s.iterrows():
                        c = r['code']
                        if c not in driver_telemetry:
                            driver_telemetry[c] = {
                                'name': r['name'],
                                'team': r['team'],
                                'latest_pos': pos + 1,
                                'latest_delta': r['time'] - best_t,
                                'fp1_pos': 22, 'fp1_delta': 4.0,
                                'fp2_pos': 22, 'fp2_delta': 4.0,
                                'fp3_pos': 22, 'fp3_delta': 4.0,
                            }
                        driver_telemetry[c]['latest_pos'] = pos + 1
                        driver_telemetry[c]['latest_delta'] = r['time'] - best_t
                        if '1' in s_name:
                            driver_telemetry[c]['fp1_pos'] = pos + 1
                            driver_telemetry[c]['fp1_delta'] = r['time'] - best_t
                        elif '2' in s_name:
                            driver_telemetry[c]['fp2_pos'] = pos + 1
                            driver_telemetry[c]['fp2_delta'] = r['time'] - best_t
                        elif '3' in s_name:
                            driver_telemetry[c]['fp3_pos'] = pos + 1
                            driver_telemetry[c]['fp3_delta'] = r['time'] - best_t
        except Exception as e:
            print(f"Sesión {s_name} aún no disponible o sin telemetría: {e}")

    # 2. Standings Oficiales 2026 de Jolpica-F1
    req = urllib.request.Request("https://api.jolpi.ca/ergast/f1/2026/driverstandings.json", headers={'User-Agent': 'AntigravityBot/1.0'})
    with urllib.request.urlopen(req) as resp:
        standings_data = json.loads(resp.read().decode('utf-8'))
    standings = standings_data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']

    pilotos = []
    for s in standings:
        pos = int(s['position'])
        name = f"{s['Driver']['givenName']} {s['Driver']['familyName']}"
        code = s['Driver'].get('code')
        team = s['Constructors'][0]['name'] if s.get('Constructors') else 'Escudería'
        points = float(s['points'])
        wins = int(s['wins'])

        t_info = None
        for k, v in driver_telemetry.items():
            if k == code or s['Driver']['familyName'].lower() in v['name'].lower() or v['name'].lower() in name.lower():
                t_info = v
                break
        if not t_info:
            t_info = {'fp1_pos': pos, 'fp1_delta': pos * 0.15, 'fp2_pos': pos, 'fp2_delta': pos * 0.15, 'latest_pos': pos, 'latest_delta': pos * 0.15}

        # Ritmo representativo: si hay FP3 se le da prioridad máxima; si no, el mejor de FP1/FP2
        latest_delta = t_info.get('latest_delta', t_info.get('fp2_delta', 1.0))
        fp2_delta = t_info.get('fp2_delta', 1.0)
        fp1_delta = t_info.get('fp1_delta', 1.0)

        pilotos.append({
            'nombre': name,
            'escuderia': team,
            'pos_mundial': pos,
            'puntos': points,
            'victorias': wins,
            'fp1_pos': t_info.get('fp1_pos', 22),
            'fp2_pos': t_info.get('fp2_pos', 22),
            'latest_pos': t_info.get('latest_pos', pos),
            'latest_delta': round(latest_delta, 3),
            'fp1_delta': round(fp1_delta, 3),
            'fp2_delta': round(fp2_delta, 3),
        })

    # 3. Simulación Cuantitativa Monte Carlo (10,000 iteraciones)
    n_sims = 10000
    n_pilotos = len(pilotos)

    # Si la Qualy ya se corrió, la Pole es 100% certera para el P1
    if qualy_completada:
        qualy_base_pace = np.array([p['latest_pos'] for p in pilotos], dtype=float)
    else:
        qualy_base_pace = np.array([
            (p['latest_delta'] * 0.70 + p['fp1_delta'] * 0.20 - (p['puntos'] / 300.0) * 0.2)
            for p in pilotos
        ])
        qualy_base_pace -= qualy_base_pace.min()

    race_base_pace = np.array([
        (p['latest_delta'] * 0.50 + p['fp2_delta'] * 0.20 - (p['victorias'] * 0.12) - (p['puntos'] / 250.0) * 0.35)
        for p in pilotos
    ])
    race_base_pace -= race_base_pace.min()

    if qualy_completada:
        pole_counts = np.zeros(n_pilotos)
        # El P1 de qualy tiene el 100% de la Pole
        winner_pole_idx = int(np.argmin(qualy_base_pace))
        pole_counts[winner_pole_idx] = n_sims
    else:
        qualy_sims = qualy_base_pace[:, None] + np.random.normal(0, 0.18, (n_pilotos, n_sims))
        pole_counts = np.sum(qualy_sims == np.min(qualy_sims, axis=0), axis=1)

    race_sims = race_base_pace[:, None] + np.random.normal(0, 0.32, (n_pilotos, n_sims))
    race_ranks = np.argsort(race_sims, axis=0)

    win_counts = np.zeros(n_pilotos)
    podium_counts = np.zeros(n_pilotos)

    for sim in range(n_sims):
        winner_idx = race_ranks[0, sim]
        win_counts[winner_idx] += 1
        for top3_pos in range(3):
            podium_counts[race_ranks[top3_pos, sim]] += 1

    resultados = []
    for i, p in enumerate(pilotos):
        prob_pole = round((pole_counts[i] / n_sims) * 100, 2)
        prob_win = round((win_counts[i] / n_sims) * 100, 2)
        prob_podium = round((podium_counts[i] / n_sims) * 100, 2)

        resultados.append({
            'piloto': p['nombre'],
            'escuderia': p['escuderia'],
            'pos_mundial': p['pos_mundial'],
            'puntos': p['puntos'],
            'victorias': p['victorias'],
            'fp1_pos': p['fp1_pos'],
            'fp2_pos': p['fp2_pos'],
            'latest_pos': p['latest_pos'],
            'latest_delta': p['latest_delta'],
            'prob_pole': f"{prob_pole}%",
            'prob_victoria': f"{prob_win}%",
            'prob_podio': f"{prob_podium}%",
            'raw_pole': prob_pole,
            'raw_win': prob_win,
            'raw_podium': prob_podium
        })

    f1_cached_result = {
        "sesion_mas_reciente": ultima_sesion,
        "qualy_completada": qualy_completada,
        "analisis_f1": resultados
    }
    f1_cache_time = ahora_ts
    return f1_cached_result

@app.post("/analizar-f1")
def analizar_f1():
    return ejecutar_simulacion_f1_real(2026, 'Azerbaijan')

@app.get("/analizar-f1")
def analizar_f1_get():
    return ejecutar_simulacion_f1_real(2026, 'Azerbaijan')