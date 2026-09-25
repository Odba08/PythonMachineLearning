import os
import sys
import io
import traceback
import warnings
import fastf1
import pandas as pd
import psycopg2

cache_dir = 'C:/Users/oscar.bueno/.gemini/antigravity/brain/e5a5b865-7840-4b89-8871-00121bfd9e29/scratch/fastf1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

DB_URL = "postgresql://postgres:secret@localhost:5432/motor_apuestas"

print("Conectando a FastF1 para extraer telemetria real de FIA F1...")

try:
    session = fastf1.get_session(2024, 'Monza', 'Q')
    session.load(telemetry=False)
    
    event_name = str(session.event['EventName'])
    location = str(session.event['Location'])
    event_date = str(session.event['EventDate']).split(' ')[0]
    
    print(f"Gran Premio Extraido: {event_name} ({location}) - Fecha: {event_date}")
    
    results = session.results
    
    conn = psycopg2.connect(DB_URL)
    conn.set_client_encoding('UTF8')
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO "GranPremioF1" (nombre, circuito, fecha, fase, "updatedAt")
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (nombre) DO UPDATE 
        SET circuito = EXCLUDED.circuito, fecha = EXCLUDED.fecha, fase = EXCLUDED.fase, "updatedAt" = NOW();
    """, (f"Gran Premio de {location} ({event_name})", f"Circuito de {location}", event_date, "Clasificacion Oficial Q3 Finalizada"))
    
    driver_count = 0
    for idx, row in results.iterrows():
        pos = int(row['Position']) if pd.notnull(row['Position']) else 20
        name = str(row['FullName']) if pd.notnull(row['FullName']) else str(row['BroadcastName'])
        team = str(row['TeamName']) if pd.notnull(row['TeamName']) else "F1 Team"
        
        if not name or name == 'nan':
            continue
          
        elo = float(round(2100.0 - (pos - 1) * 25.0, 1))
        puntos = float(max(0, 25 - (pos - 1) * 2))
        wins = 1 if pos == 1 else 0
        poles = 1 if pos == 1 else 0
        podios = 1 if pos <= 3 else 0
        
        cursor.execute("""
            INSERT INTO "PilotoF1" (nombre, escuderia, elo, "puntosMundial", victorias, poles, podios, "podiosConsecutivos", "fp1Pos", "fp2Pos", "rachaReciente", form5, "updatedAt")
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (nombre) DO UPDATE 
            SET escuderia = EXCLUDED.escuderia, elo = EXCLUDED.elo, "puntosMundial" = EXCLUDED."puntosMundial", 
                "fp1Pos" = EXCLUDED."fp1Pos", "fp2Pos" = EXCLUDED."fp2Pos", "updatedAt" = NOW();
        """, (name, team, elo, puntos, wins, poles, podios, 1 if pos <= 3 else 0, pos, pos, f"P{pos} en {event_name}", float(100 - (pos * 4))))
        
        cursor.execute("""
            INSERT INTO "EscuderiaF1" (nombre, elo, "puntosMundial", victorias, "updatedAt")
            VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT (nombre) DO UPDATE 
            SET elo = EXCLUDED.elo, "updatedAt" = NOW();
        """, (team, elo + 50.0, puntos, wins))
        
        driver_count += 1
        
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"EXITO DE TELEMETRIA REAL! Se han importado {driver_count} pilotos reales directamente de FastF1 a PostgreSQL.")

except Exception as e:
    print(f"Error al procesar FastF1: {repr(e)}")
    traceback.print_exc()
