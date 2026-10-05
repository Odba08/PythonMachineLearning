import os
import sys
import json
import time
import urllib.request
from datetime import datetime, timezone
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import fastf1

# Silenciar logs ruidosos de FastF1
import logging
fastf1_logger = logging.getLogger('fastf1')
fastf1_logger.setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

cache_dir = os.path.join(os.environ.get('TEMP', '.'), 'fastf1')
try:
    fastf1.Cache.enable_cache(cache_dir)
except Exception:
    pass

def ejecutar_simulacion():
    year = 2026
    sched = fastf1.get_event_schedule(year)
    ahora_dt = pd.to_datetime(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
    valid = sched[sched['Session5DateUtc'].notna()]
    current = valid[valid['Session5DateUtc'] >= ahora_dt]
    event_row = current.iloc[0] if not current.empty else sched.iloc[18]
    round_num = int(event_row['RoundNumber'])

    event = fastf1.get_event(year, round_num)
    ahora_utc = datetime.now(timezone.utc)

    sesiones_a_cargar = []
    qualy_completada = False

    for i in range(1, 6):
        s_name = event[f'Session{i}']
        s_date = event[f'Session{i}DateUtc'] if 'Session{i}DateUtc' in event else event[f'Session{i}Date']
        if s_date <= ahora_utc:
            sesiones_a_cargar.append((i, s_name))

    driver_telemetry = {}
    ultima_sesion = "FP1"
    lider_actual = {'nombre': 'Charles Leclerc', 'equipo': 'Ferrari', 'sesion': 'Practice 2', 'tiempo': 97.528}
    sesiones_procesadas = []

    for idx, s_name in sesiones_a_cargar:
        try:
            s = fastf1.get_session(year, round_num, s_name)
            s.load(laps=True, telemetry=False, weather=False)
            if hasattr(s, 'laps') and len(s.laps) > 0:
                ultima_sesion = s_name
                sesiones_procesadas.append(s_name)
                if 'Qualifying' in s_name or s_name == 'Q':
                    qualy_completada = True

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
                    best_t = float(df_s.iloc[0]['time'])
                    lider_actual = {
                        'nombre': str(df_s.iloc[0]['name']),
                        'equipo': str(df_s.iloc[0]['team']),
                        'sesion': s_name,
                        'tiempo': round(best_t, 3)
                    }
                    for pos, r in df_s.iterrows():
                        c = r['code']
                        delta = float(r['time'] - best_t)
                        if c not in driver_telemetry:
                            driver_telemetry[c] = {
                                'name': r['name'],
                                'team': r['team'],
                                'latest_pos': pos + 1,
                                'latest_delta': delta,
                                'fp1_pos': 22, 'fp1_delta': 4.0,
                                'fp2_pos': 22, 'fp2_delta': 4.0,
                                'fp3_pos': 22, 'fp3_delta': 4.0,
                            }
                        driver_telemetry[c]['latest_pos'] = pos + 1
                        driver_telemetry[c]['latest_delta'] = delta
                        if '1' in s_name:
                            driver_telemetry[c]['fp1_pos'] = pos + 1
                            driver_telemetry[c]['fp1_delta'] = delta
                        elif '2' in s_name:
                            driver_telemetry[c]['fp2_pos'] = pos + 1
                            driver_telemetry[c]['fp2_delta'] = delta
                        elif '3' in s_name:
                            driver_telemetry[c]['fp3_pos'] = pos + 1
                            driver_telemetry[c]['fp3_delta'] = delta
        except Exception:
            pass

    # Standings Oficiales Jolpica
    try:
        req = urllib.request.Request("https://api.jolpi.ca/ergast/f1/2026/driverstandings.json", headers={'User-Agent': 'ApexBot/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            standings_data = json.loads(resp.read().decode('utf-8'))
        standings = standings_data['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
    except Exception:
        standings = []

    pilotos = []
    if standings:
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

            pilotos.append({
                'nombre': name,
                'escuderia': team,
                'pos_mundial': pos,
                'puntos': points,
                'victorias': wins,
                'fp1_pos': t_info.get('fp1_pos', 22),
                'fp2_pos': t_info.get('fp2_pos', 22),
                'latest_pos': t_info.get('latest_pos', pos),
                'latest_delta': round(t_info.get('latest_delta', 1.0), 3),
                'fp1_delta': round(t_info.get('fp1_delta', 1.0), 3),
                'fp2_delta': round(t_info.get('fp2_delta', 1.0), 3),
            })
    else:
        # Fallback de pilotos si la API externa de Jolpica no responde
        pilotos_default = [
            ('Andrea Kimi Antonelli', 'Mercedes', 1, 302.0, 8),
            ('George Russell', 'Mercedes', 2, 236.0, 3),
            ('Lewis Hamilton', 'Ferrari', 3, 199.0, 1),
            ('Lando Norris', 'McLaren', 4, 185.0, 2),
            ('Charles Leclerc', 'Ferrari', 5, 172.0, 1),
            ('Max Verstappen', 'Red Bull', 6, 160.0, 1),
            ('Oscar Piastri', 'McLaren', 7, 140.0, 0),
            ('Carlos Sainz', 'Williams', 8, 85.0, 0),
            ('Fernando Alonso', 'Aston Martin', 9, 65.0, 0),
            ('Alexander Albon', 'Williams', 10, 42.0, 0),
        ]
        for name, team, pos, points, wins in pilotos_default:
            t_info = None
            for k, v in driver_telemetry.items():
                if name.lower() in v['name'].lower():
                    t_info = v
                    break
            if not t_info:
                t_info = {'fp1_pos': pos, 'fp1_delta': pos * 0.15, 'fp2_pos': pos, 'fp2_delta': pos * 0.15, 'latest_pos': pos, 'latest_delta': pos * 0.15}
            pilotos.append({
                'nombre': name,
                'escuderia': team,
                'pos_mundial': pos,
                'puntos': points,
                'victorias': wins,
                'fp1_pos': t_info.get('fp1_pos', pos),
                'fp2_pos': t_info.get('fp2_pos', pos),
                'latest_pos': t_info.get('latest_pos', pos),
                'latest_delta': round(t_info.get('latest_delta', 1.0), 3),
                'fp1_delta': round(t_info.get('fp1_delta', 1.0), 3),
                'fp2_delta': round(t_info.get('fp2_delta', 1.0), 3),
            })

    # Simulación Monte Carlo
    n_sims = 10000
    n_pilotos = len(pilotos)

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
        prob_pole = round(float((pole_counts[i] / n_sims) * 100), 2)
        prob_win = round(float((win_counts[i] / n_sims) * 100), 2)
        prob_podium = round(float((podium_counts[i] / n_sims) * 100), 2)

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

    is_bahrain_in_malaysia = 'Bahrain' in str(event.get('EventName', '')) or 'Malaysia' in str(event.get('OfficialEventName', ''))
    gp_info = {
        'nombre': 'Gran Premio de Bahréin (en Malasia / Sepang)' if is_bahrain_in_malaysia else str(event.get('EventName', 'Gran Premio de F1')),
        'circuito': 'Circuito Internacional de Sepang (Kuala Lumpur)' if is_bahrain_in_malaysia else str(event.get('Location', 'Circuito F1')),
        'fecha': str(event.get('EventDate', '2026-10-04')).split(' ')[0],
        'round': round_num,
        'official_name': str(event.get('OfficialEventName', ''))
    }

    by_win = sorted(resultados, key=lambda x: x['raw_win'], reverse=True)
    by_pole = sorted(resultados, key=lambda x: x['raw_pole'], reverse=True)
    by_podium = sorted(resultados, key=lambda x: x['raw_podium'], reverse=True)

    win_prob = round(by_win[0]['raw_win']) if by_win else 50
    cuota_est = round((100 / win_prob) * 0.95, 2) if win_prob > 0 else 2.10

    output = {
        'gp': gp_info,
        'sesion_mas_reciente': ultima_sesion,
        'sesiones_cargadas': sesiones_procesadas if sesiones_procesadas else [ultima_sesion],
        'lider_sesion_reciente': lider_actual,
        'qualy_completada': qualy_completada,
        'predicciones_top': {
            'pole_position': {
                'piloto': by_pole[0]['piloto'] if by_pole else 'George Russell',
                'probabilidad': round(by_pole[0]['raw_pole']) if by_pole else 40
            },
            'probabilidad_victoria': {
                'piloto': by_win[0]['piloto'] if by_win else 'Andrea Kimi Antonelli',
                'probabilidad': win_prob,
                'cuota_estimada': cuota_est
            },
            'top3_podio': [
                {'piloto': p['piloto'], 'probabilidad': round(p['raw_podium'])}
                for p in by_podium[:4]
            ]
        },
        'analisis_f1': resultados
    }

    return output

if __name__ == '__main__':
    res = ejecutar_simulacion()
    print("###START_JSON###")
    print(json.dumps(res))
    print("###END_JSON###")
