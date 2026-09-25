import os
import sys
import io
import pandas as pd
import fastf1

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Activar caché para FastF1
cache_dir = os.path.join(os.path.expanduser('~'), '.fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def analizar_ritmo_practica(year=2024, gp_name='Azerbaijan', session_type='Q'):
    print(f"\n🏎️ --- ANÁLISIS AUTOMÁTICO DE TELEMETRÍA FIA (FASTF1) ---")
    print(f"GP: {gp_name} {year} | Sesión: {session_type}\n")
    
    try:
        session = fastf1.get_session(year, gp_name, session_type)
        session.load(laps=True, telemetry=True, weather=False)
        
        # Extraer la vuelta más rápida de cada piloto
        fastest_laps = []
        for drv in session.drivers:
            drv_laps = session.laps.pick_driver(drv)
            if not drv_laps.empty:
                fastest_lap = drv_laps.pick_fastest()
                if fastest_lap is not None and not pd.isna(fastest_lap['LapTime']):
                    drv_info = session.get_driver(drv)
                    fastest_laps.append({
                        'Abbr': drv_info['Abbreviation'],
                        'Piloto': drv_info['FullName'],
                        'Team': drv_info['TeamName'],
                        'LapTimeSec': fastest_lap['LapTime'].total_seconds(),
                        'Compound': fastest_lap['Compound'],
                        'Sector1Sec': fastest_lap['Sector1Time'].total_seconds() if fastest_lap['Sector1Time'] else 0,
                        'Sector2Sec': fastest_lap['Sector2Time'].total_seconds() if fastest_lap['Sector2Time'] else 0,
                        'Sector3Sec': fastest_lap['Sector3Time'].total_seconds() if fastest_lap['Sector3Time'] else 0,
                    })

        df = pd.DataFrame(fastest_laps)
        if not df.empty:
            df = df.sort_values(by='LapTimeSec').reset_index(drop=True)
            pole_time = df.iloc[0]['LapTimeSec']
            df['DeltaVsPole'] = df['LapTimeSec'] - pole_time
            
            print(f"⏱️ Clasificación / Ritmo de Práctica Obtenido:")
            for idx, r in df.iterrows():
                gap = f"+{r['DeltaVsPole']:.3f}s" if r['DeltaVsPole'] > 0 else "POLE / LÍDER"
                print(f"  P{idx+1}: {r['Piloto']} ({r['Team']}) - {r['LapTimeSec']:.3f}s ({gap}) [Neumático: {r['Compound']}]")
            
            return df
        else:
            print("⚠️ No se registraron tiempos válidos en esta sesión.")
            return None
    except Exception as e:
        print(f"❌ Error analizando telemetría: {e}")
        return None

if __name__ == '__main__':
    analizar_ritmo_practica(2024, 'Azerbaijan', 'Q')
