import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    lider_actual = {'nombre': 'George Russell', 'equipo': 'Mercedes', 'sesion': 'Practice 2'}
    sesiones_procesadas = []

    for idx, s_name in sesiones_a_cargar:
        try:
            s = fastf1.get_session(year, gp, s_name)
            s.load(laps=True, telemetry=False, weather=False)
            if hasattr(s, 'laps') and len(s.laps) > 0:
                ultima_sesion = s_name
                sesiones_procesadas.append(s_name)
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
                    lider_actual = {
                        'nombre': str(df_s.iloc[0]['name']),
                        'equipo': str(df_s.iloc[0]['team']),
                        'sesion': s_name,
                        'tiempo': round(best_t, 3)
                    }
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
                        elif 'Sprint' in s_name:
                            driver_telemetry[c]['sprint_pos'] = pos + 1
                            driver_telemetry[c]['sprint_delta'] = r['time'] - best_t
                        elif 'Qualifying' in s_name or s_name == 'Q':
                            driver_telemetry[c]['qualy_pos'] = pos + 1
                            driver_telemetry[c]['qualy_delta'] = r['time'] - best_t
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
        "sesiones_cargadas": sesiones_procesadas if sesiones_procesadas else [ultima_sesion],
        "lider_sesion_reciente": lider_actual,
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

# ------------------------------------------------------------------
# 5. MOTOR PREDICTIVO UFC (+EV VALUE HUNTER & THE-ODDS-API)
# ------------------------------------------------------------------
try:
    ufc_model = joblib.load(os.path.join(BASE_DIR, 'modelo_ufc_ganador.pkl'))
    ufc_scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_ufc.pkl'))
    ufc_features = joblib.load(os.path.join(BASE_DIR, 'features_ufc.pkl'))
    print("Cerebro UFC cargado exitosamente en memoria.")
except Exception as e:
    print(f"Aviso cargando cerebro UFC: {e}")

# Funciones de sincronización con el repositorio de Greco (GitHub)
GRECO_RAW_URLS = {
    'tott': 'https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/master/ufc_fighter_tott.csv',
    'details': 'https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/master/ufc_fighter_details.csv'
}

def sincronizar_csvs_greco():
    actualizados = []
    for tag, url in GRECO_RAW_URLS.items():
        dest = os.path.join(BASE_DIR, "data", f"ufc_fighter_{tag}.csv")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                content = resp.read()
                with open(dest, 'wb') as f:
                    f.write(content)
                actualizados.append(dest)
        except Exception as e:
            print(f"Aviso actualizando {tag} de Greco: {e}")
    return actualizados

# Cargar Tale of the Tape y compilar estadísticas de carrera
ft_df = None
fighter_career_stats = {}

WEIGHT_CLASS_PRIORS = {
    'Heavyweight': {'ko': 0.515, 'sub': 0.126, 'dec': 0.359},
    'Light Heavyweight': {'ko': 0.453, 'sub': 0.175, 'dec': 0.372},
    'Middleweight': {'ko': 0.388, 'sub': 0.179, 'dec': 0.433},
    'Welterweight': {'ko': 0.326, 'sub': 0.171, 'dec': 0.503},
    'Lightweight': {'ko': 0.308, 'sub': 0.202, 'dec': 0.490},
    'Featherweight': {'ko': 0.295, 'sub': 0.168, 'dec': 0.537},
    'Bantamweight': {'ko': 0.262, 'sub': 0.195, 'dec': 0.543},
    'Flyweight': {'ko': 0.248, 'sub': 0.204, 'dec': 0.548},
    "Women's Bantamweight": {'ko': 0.222, 'sub': 0.175, 'dec': 0.603},
    "Women's Flyweight": {'ko': 0.175, 'sub': 0.198, 'dec': 0.627},
    "Women's Strawweight": {'ko': 0.142, 'sub': 0.199, 'dec': 0.659},
    "Women's Featherweight": {'ko': 0.160, 'sub': 0.240, 'dec': 0.600},
    'Catch Weight': {'ko': 0.280, 'sub': 0.200, 'dec': 0.520},
}
DEFAULT_PRIOR = {'ko': 0.320, 'sub': 0.180, 'dec': 0.500}

def norm_name(n):
    return str(n).replace('.', '').strip().lower()

def recargar_datos_peleadores():
    global ft_df, fighter_career_stats
    try:
        ft_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ufc_fighter_tott.csv'))
    except Exception as e:
        print(f"Aviso leyendo tott: {e}")
        ft_df = pd.DataFrame()

    try:
        df_master = pd.read_csv(os.path.join(BASE_DIR, 'data', 'ufc-master.csv'))
        fighter_career_stats = {}
        for idx, r in df_master.iterrows():
            for prefix, name_col in [('R_', 'R_fighter'), ('B_', 'B_fighter')]:
                raw_name = str(r[name_col]).strip()
                k = norm_name(raw_name)
                if k not in fighter_career_stats:
                    fighter_career_stats[k] = {
                        'name': raw_name,
                        'sig_str': float(r.get(f'{prefix}avg_SIG_STR_landed', 3.5)) if not pd.isna(r.get(f'{prefix}avg_SIG_STR_landed')) else 3.5,
                        'avg_td': float(r.get(f'{prefix}avg_TD_landed', 1.2)) if not pd.isna(r.get(f'{prefix}avg_TD_landed')) else 1.2,
                        'avg_sub': float(r.get(f'{prefix}avg_SUB_ATT', 0.5)) if not pd.isna(r.get(f'{prefix}avg_SUB_ATT')) else 0.5,
                        'win_streak': float(r.get(f'{prefix}current_win_streak', 0)) if not pd.isna(r.get(f'{prefix}current_win_streak')) else 0.0,
                        'longest_win_streak': float(r.get(f'{prefix}longest_win_streak', 1)) if not pd.isna(r.get(f'{prefix}longest_win_streak')) else 1.0,
                        'lose_streak': float(r.get(f'{prefix}current_lose_streak', 0)) if not pd.isna(r.get(f'{prefix}current_lose_streak')) else 0.0,
                        'wins': float(r.get(f'{prefix}wins', 5)) if not pd.isna(r.get(f'{prefix}wins')) else 5.0,
                        'losses': float(r.get(f'{prefix}losses', 2)) if not pd.isna(r.get(f'{prefix}losses')) else 2.0,
                        'total_rounds': float(r.get(f'{prefix}total_rounds_fought', 15)) if not pd.isna(r.get(f'{prefix}total_rounds_fought')) else 15.0,
                        'total_title_bouts': float(r.get(f'{prefix}total_title_bouts', 0)) if not pd.isna(r.get(f'{prefix}total_title_bouts')) else 0.0,
                        'ko_wins': float(r.get(f'{prefix}win_by_KO/TKO', 1)) if not pd.isna(r.get(f'{prefix}win_by_KO/TKO')) else 1.0,
                        'sub_wins': float(r.get(f'{prefix}win_by_Submission', 1)) if not pd.isna(r.get(f'{prefix}win_by_Submission')) else 1.0,
                        'dec_wins': float(
                            (r.get(f'{prefix}win_by_Decision_Unanimous', 0) or 0) +
                            (r.get(f'{prefix}win_by_Decision_Split', 0) or 0) +
                            (r.get(f'{prefix}win_by_Decision_Majority', 0) or 0)
                        ),
                    }
        print(f"Estadísticas de {len(fighter_career_stats)} peleadores compiladas en memoria.")
    except Exception as e:
        print(f"Aviso compilando estadísticas de carrera: {e}")

recargar_datos_peleadores()

def parse_height_cm(h_str):
    if not isinstance(h_str, str) or '--' in h_str:
        return 178.0
    try:
        parts = h_str.replace('"', '').split("'")
        feet = float(parts[0].strip())
        inches = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0.0
        return round(feet * 30.48 + inches * 2.54, 2)
    except:
        return 178.0

def parse_reach_cm(r_str, default_height=178.0):
    if not isinstance(r_str, str) or '--' in r_str:
        return default_height
    try:
        inches = float(r_str.replace('"', '').strip())
        return round(inches * 2.54, 2)
    except:
        return default_height

def parse_age(dob_str):
    if not isinstance(dob_str, str) or '--' in dob_str:
        return 30.0
    try:
        dob = datetime.strptime(dob_str.strip(), "%b %d, %Y")
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return float(age)
    except:
        return 30.0

def get_fighter_finishing_profile(name, weight_class):
    prior = WEIGHT_CLASS_PRIORS.get(weight_class, DEFAULT_PRIOR)
    k = norm_name(name)
    s = fighter_career_stats.get(k)
    if not s or s['wins'] <= 0:
        return prior['ko'], prior['sub'], prior['dec']

    w = s['wins']
    ko_r = s['ko_wins'] / w
    sub_r = s['sub_wins'] / w
    dec_r = s['dec_wins'] / w if s['dec_wins'] > 0 else max(0.0, 1.0 - (ko_r + sub_r))

    weight = min(w / 8.0, 0.70)
    p_ko = (1.0 - weight) * prior['ko'] + weight * ko_r
    p_sub = (1.0 - weight) * prior['sub'] + weight * sub_r
    p_dec = (1.0 - weight) * prior['dec'] + weight * dec_r

    tot = p_ko + p_sub + p_dec
    return p_ko / tot, p_sub / tot, p_dec / tot

def calcular_props_combate(f_red, f_blue, prob_red, prob_blue, weight_class):
    r_ko, r_sub, r_dec = get_fighter_finishing_profile(f_red, weight_class)
    b_ko, b_sub, b_dec = get_fighter_finishing_profile(f_blue, weight_class)

    p_r = prob_red / 100.0
    p_b = prob_blue / 100.0

    comb_ko = round((p_r * r_ko + p_b * b_ko) * 100.0, 1)
    comb_sub = round((p_r * r_sub + p_b * b_sub) * 100.0, 1)
    comb_dec = round(100.0 - (comb_ko + comb_sub), 1)

    va_distancia = comb_dec
    no_distancia = round(100.0 - va_distancia, 1)

    p_over_15 = round(va_distancia + (no_distancia * 0.40), 1)
    p_under_15 = round(100.0 - p_over_15, 1)

    p_over_25 = round(va_distancia + (no_distancia * 0.15), 1)
    p_under_25 = round(100.0 - p_over_25, 1)

    red_ko = round(prob_red * r_ko, 1)
    red_sub = round(prob_red * r_sub, 1)
    red_dec = round(prob_red * r_dec, 1)

    blue_ko = round(prob_blue * b_ko, 1)
    blue_sub = round(prob_blue * b_sub, 1)
    blue_dec = round(prob_blue * b_dec, 1)

    candidatos = []
    if va_distancia >= 55.0:
        candidatos.append((va_distancia, f"Combate a Decisión / Tarjetas ({va_distancia:.0f}%)"))
    if no_distancia >= 55.0:
        candidatos.append((no_distancia, f"Termina antes del límite ({no_distancia:.0f}%)"))
    if p_over_15 >= 70.0:
        candidatos.append((p_over_15, f"Más de 1.5 Asaltos ({p_over_15:.0f}%)"))
    if p_under_15 >= 50.0:
        candidatos.append((p_under_15, f"Menos de 1.5 Asaltos ({p_under_15:.0f}%)"))
    if p_over_25 >= 60.0:
        candidatos.append((p_over_25, f"Más de 2.5 Asaltos ({p_over_25:.0f}%)"))
    if p_under_25 >= 55.0:
        candidatos.append((p_under_25, f"Menos de 2.5 Asaltos ({p_under_25:.0f}%)"))

    for nombre, prob, metodo in [
        (f_red, red_sub, "por Sumisión"),
        (f_red, red_ko, "por KO/TKO"),
        (f_red, red_dec, "por Decisión"),
        (f_blue, blue_sub, "por Sumisión"),
        (f_blue, blue_ko, "por KO/TKO"),
        (f_blue, blue_dec, "por Decisión")
    ]:
        if prob >= 35.0:
            candidatos.append((prob, f"{nombre} {metodo} ({prob:.0f}%)"))

    candidatos.sort(key=lambda x: x[0], reverse=True)
    top_alt = candidatos[0][1] if candidatos else f"Combate a Decisión ({va_distancia:.0f}%)"

    return {
        'metodos': {
            'ko_tko': comb_ko,
            'sumision': comb_sub,
            'decision': comb_dec,
        },
        'distancia': {
            'va_distancia': va_distancia,
            'no_distancia': no_distancia,
        },
        'asaltos': {
            'over_15': p_over_15,
            'under_15': p_under_15,
            'over_25': p_over_25,
            'under_25': p_under_25,
        },
        'props_peleador': {
            'red_ko': red_ko,
            'red_sub': red_sub,
            'red_dec': red_dec,
            'blue_ko': blue_ko,
            'blue_sub': blue_sub,
            'blue_dec': blue_dec,
        },
        'jugada_alternativa': top_alt
    }

def predecir_combate_dinamico(f_red: str, f_blue: str, weight_class: str = 'Bantamweight'):
    k_r = norm_name(f_red)
    k_b = norm_name(f_blue)

    m_red = ft_df[ft_df['FIGHTER'].astype(str).str.replace('.', '', regex=False).str.strip().str.lower() == k_r] if ft_df is not None and not ft_df.empty else pd.DataFrame()
    m_blue = ft_df[ft_df['FIGHTER'].astype(str).str.replace('.', '', regex=False).str.strip().str.lower() == k_b] if ft_df is not None and not ft_df.empty else pd.DataFrame()

    h_r = parse_height_cm(m_red.iloc[0]['HEIGHT']) if not m_red.empty else 178.0
    r_r = parse_reach_cm(m_red.iloc[0]['REACH'], h_r) if not m_red.empty else h_r
    a_r = parse_age(m_red.iloc[0]['DOB']) if not m_red.empty else 30.0

    h_b = parse_height_cm(m_blue.iloc[0]['HEIGHT']) if not m_blue.empty else 178.0
    r_b = parse_reach_cm(m_blue.iloc[0]['REACH'], h_b) if not m_blue.empty else h_b
    a_b = parse_age(m_blue.iloc[0]['DOB']) if not m_blue.empty else 30.0

    default_s = {'sig_str': 3.5, 'avg_td': 1.2, 'avg_sub': 0.5, 'win_streak': 0, 'longest_win_streak': 1, 'lose_streak': 0, 'wins': 5, 'losses': 2, 'total_rounds': 15, 'total_title_bouts': 0, 'ko_wins': 1, 'sub_wins': 1, 'dec_wins': 2}
    s_r = fighter_career_stats.get(k_r, default_s)
    s_b = fighter_career_stats.get(k_b, default_s)

    diffs = {
        'reach_dif': r_r - r_b,
        'height_dif': h_r - h_b,
        'age_dif': a_r - a_b,
        'sig_str_dif': s_r['sig_str'] - s_b['sig_str'],
        'avg_td_dif': s_r['avg_td'] - s_b['avg_td'],
        'avg_sub_att_dif': s_r['avg_sub'] - s_b['avg_sub'],
        'win_streak_dif': s_r['win_streak'] - s_b['win_streak'],
        'longest_win_streak_dif': s_r['longest_win_streak'] - s_b['longest_win_streak'],
        'lose_streak_dif': s_r['lose_streak'] - s_b['lose_streak'],
        'win_dif': s_r['wins'] - s_b['wins'],
        'loss_dif': s_r['losses'] - s_b['losses'],
        'total_round_dif': s_r['total_rounds'] - s_b['total_rounds'],
        'total_title_bout_dif': s_r['total_title_bouts'] - s_b['total_title_bouts'],
        'ko_dif': s_r['ko_wins'] - s_b['ko_wins'],
        'sub_dif': s_r['sub_wins'] - s_b['sub_wins'],
    }

    x_df = pd.DataFrame([[diffs[c] for c in ufc_features]], columns=ufc_features)
    x_scaled = ufc_scaler.transform(x_df)
    proba = ufc_model.predict_proba(x_scaled)[0]
    prob_red = round(float(proba[1]) * 100.0, 1)
    prob_blue = round(float(proba[0]) * 100.0, 1)

    props = calcular_props_combate(f_red, f_blue, prob_red, prob_blue, weight_class)

    return prob_red, prob_blue, diffs, props

def american_to_decimal_ufc(odds):
    try:
        val = float(odds)
        if val > 0:
            return round(1.0 + (val / 100.0), 2)
        elif val < 0:
            return round(1.0 + (100.0 / abs(val)), 2)
        return 1.90
    except:
        return 1.90

@app.get("/predecir-ufc")
@app.post("/predecir-ufc")
def predecir_ufc():
    try:
        csv_path = os.path.join(BASE_DIR, 'data', 'upcoming.csv')
        df_up = pd.read_csv(csv_path)
    except Exception as e:
        return {"error": f"No se pudo cargar upcoming.csv: {e}"}

    combates = []
    for idx, r in df_up.iterrows():
        f_red = str(r['R_fighter']).strip()
        f_blue = str(r['B_fighter']).strip()
        wc = str(r.get('weight_class', 'Bantamweight'))

        prob_r, prob_b, diffs, props = predecir_combate_dinamico(f_red, f_blue, wc)

        raw_r = r.get('R_odds', 1.90)
        raw_b = r.get('B_odds', 1.90)
        try:
            val_r = float(raw_r)
            dec_r = round(val_r, 2) if 1.01 <= val_r <= 35.0 else american_to_decimal_ufc(val_r)
        except:
            dec_r = 1.90

        try:
            val_b = float(raw_b)
            dec_b = round(val_b, 2) if 1.01 <= val_b <= 35.0 else american_to_decimal_ufc(val_b)
        except:
            dec_b = 1.90

        implied_r = round((1.0 / dec_r) * 100.0, 1) if dec_r > 0 else 50.0
        implied_b = round((1.0 / dec_b) * 100.0, 1) if dec_b > 0 else 50.0

        edge_r = round(prob_r - implied_r, 1)
        edge_b = round(prob_b - implied_b, 1)

        combate = {
            'red_fighter': f_red,
            'blue_fighter': f_blue,
            'weight_class': str(r.get('weight_class', 'Combate MMA')),
            'cuota_red': dec_r,
            'cuota_blue': dec_b,
            'prob_red': prob_r,
            'prob_blue': prob_b,
            'edge_red': edge_r,
            'edge_blue': edge_b,
            'reach_dif': diffs.get('reach_dif', 0.0),
            'age_dif': diffs.get('age_dif', 0.0),
            'sig_str_dif': round(diffs.get('sig_str_dif', 0.0), 2),
            'avg_td_dif': round(diffs.get('avg_td_dif', 0.0), 2),
            'props': props,
        }

        if edge_r >= 3.0 and edge_r >= edge_b:
            combate['value_pick'] = f_red
            combate['value_side'] = 'RED'
            combate['value_odds'] = dec_r
            combate['value_prob'] = prob_r
            combate['value_edge'] = edge_r
            combate['has_value'] = True
        elif edge_b >= 3.0:
            combate['value_pick'] = f_blue
            combate['value_side'] = 'BLUE'
            combate['value_odds'] = dec_b
            combate['value_prob'] = prob_b
            combate['value_edge'] = edge_b
            combate['has_value'] = True
        else:
            combate['value_pick'] = None
            combate['value_side'] = None
            combate['value_odds'] = 0.0
            combate['value_prob'] = 0.0
            combate['value_edge'] = 0.0
            combate['has_value'] = False

        combates.append(combate)

    combates_ordenados = sorted(combates, key=lambda x: x.get('value_edge', -999) if x.get('has_value') else -999, reverse=True)

    return {
        "total_combates": len(combates),
        "combates_con_valor": sum(1 for c in combates if c.get('has_value')),
        "analisis_ufc": combates_ordenados
    }

class MatchupOddsItem(BaseModel):
    home_team: str
    away_team: str
    odds_home: float
    odds_away: float
    commence_time: str = ""
    weight_class: str = "Bantamweight"

@app.post("/analizar-peleas-odds")
def analizar_peleas_the_odds(peleas: List[MatchupOddsItem]):
    resultados = []
    for p in peleas:
        prob_home, prob_away, diffs, props = predecir_combate_dinamico(p.home_team, p.away_team, p.weight_class)

        implied_home = round((1.0 / p.odds_home) * 100.0, 1) if p.odds_home > 1.0 else 50.0
        implied_away = round((1.0 / p.odds_away) * 100.0, 1) if p.odds_away > 1.0 else 50.0

        edge_home = round(prob_home - implied_home, 1)
        edge_away = round(prob_away - implied_away, 1)

        item = {
            'fighter_home': p.home_team,
            'fighter_away': p.away_team,
            'odds_home': p.odds_home,
            'odds_away': p.odds_away,
            'prob_home': prob_home,
            'prob_away': prob_away,
            'edge_home': edge_home,
            'edge_away': edge_away,
            'commence_time': p.commence_time,
            'reach_dif': diffs['reach_dif'],
            'age_dif': diffs['age_dif'],
            'sig_str_dif': diffs['sig_str_dif'],
            'avg_td_dif': diffs['avg_td_dif'],
            'props': props,
        }

        if edge_home >= 3.0 and edge_home >= edge_away:
            item['value_pick'] = p.home_team
            item['value_side'] = 'HOME'
            item['value_odds'] = p.odds_home
            item['value_prob'] = prob_home
            item['value_edge'] = edge_home
            item['has_value'] = True
        elif edge_away >= 3.0:
            item['value_pick'] = p.away_team
            item['value_side'] = 'AWAY'
            item['value_odds'] = p.odds_away
            item['value_prob'] = prob_away
            item['value_edge'] = edge_away
            item['has_value'] = True
        else:
            item['value_pick'] = None
            item['value_side'] = None
            item['value_odds'] = 0.0
            item['value_prob'] = 0.0
            item['value_edge'] = 0.0
            item['has_value'] = False

        resultados.append(item)

    resultados_ordenados = sorted(resultados, key=lambda x: x.get('value_edge', -999) if x.get('has_value') else -999, reverse=True)

    return {
        "total_analizados": len(resultados),
        "total_con_valor": sum(1 for r in resultados if r.get('has_value')),
        "combates": resultados_ordenados
    }

@app.post("/sincronizar-ufc-greco")
@app.get("/sincronizar-ufc-greco")
def endpoint_sync_greco():
    archivos = sincronizar_csvs_greco()
    recargar_datos_peleadores()
    return {
        "status": "success",
        "mensaje": f"Archivos sincronizados desde GitHub Greco ({len(archivos)}) y memoria recargada.",
        "archivos": archivos
    }