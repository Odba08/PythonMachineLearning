import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression, Ridge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

NFL_TEAMS = {
    'arizona cardinals': 'ARI', 'cardinals': 'ARI', 'ari': 'ARI',
    'atlanta falcons': 'ATL', 'falcons': 'ATL', 'atl': 'ATL',
    'baltimore ravens': 'BAL', 'ravens': 'BAL', 'bal': 'BAL',
    'buffalo bills': 'BUF', 'bills': 'BUF', 'buf': 'BUF',
    'carolina panthers': 'CAR', 'panthers': 'CAR', 'car': 'CAR',
    'chicago bears': 'CHI', 'bears': 'CHI', 'chi': 'CHI',
    'cincinnati bengals': 'CIN', 'bengals': 'CIN', 'cin': 'CIN',
    'cleveland browns': 'CLE', 'browns': 'CLE', 'cle': 'CLE',
    'dallas cowboys': 'DAL', 'cowboys': 'DAL', 'dal': 'DAL',
    'denver broncos': 'DEN', 'broncos': 'DEN', 'den': 'DEN',
    'detroit lions': 'DET', 'lions': 'DET', 'det': 'DET',
    'green bay packers': 'GB', 'packers': 'GB', 'gb': 'GB',
    'houston texans': 'HOU', 'texans': 'HOU', 'hou': 'HOU',
    'indianapolis colts': 'IND', 'colts': 'IND', 'ind': 'IND',
    'jacksonville jaguars': 'JAX', 'jaguars': 'JAX', 'jax': 'JAX',
    'kansas city chiefs': 'KC', 'chiefs': 'KC', 'kc': 'KC',
    'las vegas raiders': 'LV', 'oakland raiders': 'LV', 'raiders': 'LV', 'lv': 'LV', 'oak': 'LV',
    'los angeles chargers': 'LAC', 'san diego chargers': 'LAC', 'chargers': 'LAC', 'lac': 'LAC', 'sd': 'LAC',
    'los angeles rams': 'LAR', 'st. louis rams': 'LAR', 'rams': 'LAR', 'lar': 'LAR', 'stl': 'LAR',
    'miami dolphins': 'MIA', 'dolphins': 'MIA', 'mia': 'MIA',
    'minnesota vikings': 'MIN', 'vikings': 'MIN', 'min': 'MIN',
    'new england patriots': 'NE', 'patriots': 'NE', 'ne': 'NE',
    'new orleans saints': 'NO', 'saints': 'NO', 'no': 'NO',
    'new york giants': 'NYG', 'giants': 'NYG', 'nyg': 'NYG',
    'new york jets': 'NYJ', 'jets': 'NYJ', 'nyj': 'NYJ',
    'philadelphia eagles': 'PHI', 'eagles': 'PHI', 'phi': 'PHI',
    'pittsburgh steelers': 'PIT', 'steelers': 'PIT', 'pit': 'PIT',
    'san francisco 49ers': 'SF', '49ers': 'SF', 'sf': 'SF',
    'seattle seahawks': 'SEA', 'seahawks': 'SEA', 'sea': 'SEA',
    'tampa bay buccaneers': 'TB', 'buccaneers': 'TB', 'tb': 'TB',
    'tennessee titans': 'TEN', 'titans': 'TEN', 'ten': 'TEN',
    'washington commanders': 'WAS', 'washington redskins': 'WAS', 'washington football team': 'WAS', 'commanders': 'WAS', 'was': 'WAS'
}

MLB_TEAMS = {
    'arizona diamondbacks': 'ARI', 'diamondbacks': 'ARI', 'd-backs': 'ARI', 'ari': 'ARI',
    'atlanta braves': 'ATL', 'braves': 'ATL', 'atl': 'ATL',
    'baltimore orioles': 'BAL', 'orioles': 'BAL', 'bal': 'BAL',
    'boston red sox': 'BOS', 'red sox': 'BOS', 'bos': 'BOS',
    'chicago cubs': 'CHC', 'cubs': 'CHC', 'chc': 'CHC',
    'chicago white sox': 'CWS', 'white sox': 'CWS', 'cws': 'CWS', 'chw': 'CWS',
    'cincinnati reds': 'CIN', 'reds': 'CIN', 'cin': 'CIN',
    'cleveland guardians': 'CLE', 'cleveland indians': 'CLE', 'guardians': 'CLE', 'cle': 'CLE',
    'colorado rockies': 'COL', 'rockies': 'COL', 'col': 'COL',
    'detroit tigers': 'DET', 'tigers': 'DET', 'det': 'DET',
    'houston astros': 'HOU', 'astros': 'HOU', 'hou': 'HOU',
    'kansas city royals': 'KC', 'royals': 'KC', 'kc': 'KC',
    'los angeles angels': 'LAA', 'angels': 'LAA', 'laa': 'LAA',
    'los angeles dodgers': 'LAD', 'dodgers': 'LAD', 'lad': 'LAD',
    'miami marlins': 'MIA', 'marlins': 'MIA', 'mia': 'MIA',
    'milwaukee brewers': 'MIL', 'brewers': 'MIL', 'mil': 'MIL',
    'minnesota twins': 'MIN', 'twins': 'MIN', 'min': 'MIN',
    'new york mets': 'NYM', 'mets': 'NYM', 'nym': 'NYM',
    'new york yankees': 'NYY', 'yankees': 'NYY', 'nyy': 'NYY',
    'oakland athletics': 'OAK', 'athletics': 'OAK', 'a\'s': 'OAK', 'oak': 'OAK',
    'philadelphia phillies': 'PHI', 'phillies': 'PHI', 'phi': 'PHI',
    'pittsburgh pirates': 'PIT', 'pirates': 'PIT', 'pit': 'PIT',
    'san diego padres': 'SD', 'padres': 'SD', 'sd': 'SD',
    'san francisco giants': 'SF', 'giants': 'SF', 'sf': 'SF',
    'seattle mariners': 'SEA', 'mariners': 'SEA', 'sea': 'SEA',
    'st. louis cardinals': 'STL', 'cardinals': 'STL', 'stl': 'STL',
    'tampa bay rays': 'TB', 'rays': 'TB', 'tb': 'TB',
    'texas rangers': 'TEX', 'rangers': 'TEX', 'tex': 'TEX',
    'toronto blue jays': 'TOR', 'blue jays': 'TOR', 'tor': 'TOR',
    'washington nationals': 'WSH', 'nationals': 'WSH', 'wsh': 'WSH', 'was': 'WSH'
}

NBA_TEAMS = {
    'atlanta hawks': 'ATL', 'hawks': 'ATL', 'atl': 'ATL',
    'boston celtics': 'BOS', 'celtics': 'BOS', 'bos': 'BOS',
    'brooklyn nets': 'BKN', 'new jersey nets': 'BKN', 'nets': 'BKN', 'bkn': 'BKN',
    'charlotte hornets': 'CHA', 'charlotte bobcats': 'CHA', 'hornets': 'CHA', 'cha': 'CHA',
    'chicago bulls': 'CHI', 'bulls': 'CHI', 'chi': 'CHI',
    'cleveland cavaliers': 'CLE', 'cavaliers': 'CLE', 'cavs': 'CLE', 'cle': 'CLE',
    'dallas mavericks': 'DAL', 'mavericks': 'DAL', 'mavs': 'DAL', 'dal': 'DAL',
    'denver nuggets': 'DEN', 'nuggets': 'DEN', 'den': 'DEN',
    'detroit pistons': 'DET', 'pistons': 'DET', 'det': 'DET',
    'golden state warriors': 'GSW', 'warriors': 'GSW', 'gsw': 'GSW',
    'houston rockets': 'HOU', 'rockets': 'HOU', 'hou': 'HOU',
    'indiana pacers': 'IND', 'pacers': 'IND', 'ind': 'IND',
    'los angeles clippers': 'LAC', 'clippers': 'LAC', 'lac': 'LAC',
    'los angeles lakers': 'LAL', 'lakers': 'LAL', 'lal': 'LAL',
    'memphis grizzlies': 'MEM', 'vancouver grizzlies': 'MEM', 'grizzlies': 'MEM', 'mem': 'MEM',
    'miami heat': 'MIA', 'heat': 'MIA', 'mia': 'MIA',
    'milwaukee bucks': 'MIL', 'bucks': 'MIL', 'mil': 'MIL',
    'minnesota timberwolves': 'MIN', 'timberwolves': 'MIN', 'wolves': 'MIN', 'min': 'MIN',
    'new orleans pelicans': 'NOP', 'pelicans': 'NOP', 'nop': 'NOP', 'noh': 'NOP',
    'new york knicks': 'NYK', 'knicks': 'NYK', 'nyk': 'NYK',
    'oklahoma city thunder': 'OKC', 'seattle supersonics': 'OKC', 'thunder': 'OKC', 'okc': 'OKC', 'sea': 'OKC',
    'orlando magic': 'ORL', 'magic': 'ORL', 'orl': 'ORL',
    'philadelphia 76ers': 'PHI', '76ers': 'PHI', 'sixers': 'PHI', 'phi': 'PHI',
    'phoenix suns': 'PHX', 'suns': 'PHX', 'phx': 'PHX', 'pho': 'PHX',
    'portland trail blazers': 'POR', 'trail blazers': 'POR', 'blazers': 'POR', 'por': 'POR',
    'sacramento kings': 'SAC', 'kings': 'SAC', 'sac': 'SAC',
    'san antonio spurs': 'SAS', 'spurs': 'SAS', 'sas': 'SAS', 'sa': 'SAS',
    'toronto raptors': 'TOR', 'raptors': 'TOR', 'tor': 'TOR',
    'utah jazz': 'UTA', 'jazz': 'UTA', 'uta': 'UTA',
    'washington wizards': 'WAS', 'wizards': 'WAS', 'was': 'WAS'
}
NHL_TEAMS = {
    'anaheim ducks': 'ANA', 'ducks': 'ANA', 'ana': 'ANA',
    'boston bruins': 'BOS', 'bruins': 'BOS', 'bos': 'BOS',
    'buffalo sabres': 'BUF', 'sabres': 'BUF', 'buf': 'BUF',
    'calgary flames': 'CGY', 'flames': 'CGY', 'cgy': 'CGY',
    'carolina hurricanes': 'CAR', 'hurricanes': 'CAR', 'car': 'CAR',
    'chicago blackhawks': 'CHI', 'blackhawks': 'CHI', 'chi': 'CHI',
    'colorado avalanche': 'COL', 'avalanche': 'COL', 'col': 'COL',
    'columbus blue jackets': 'CBJ', 'blue jackets': 'CBJ', 'cbj': 'CBJ',
    'dallas stars': 'DAL', 'stars': 'DAL', 'dal': 'DAL',
    'detroit red wings': 'DET', 'red wings': 'DET', 'det': 'DET',
    'edmonton oilers': 'EDM', 'oilers': 'EDM', 'edm': 'EDM',
    'florida panthers': 'FLA', 'panthers': 'FLA', 'fla': 'FLA',
    'los angeles kings': 'LAK', 'la kings': 'LAK', 'kings': 'LAK', 'lak': 'LAK', 'la': 'LAK',
    'minnesota wild': 'MIN', 'wild': 'MIN', 'min': 'MIN',
    'montreal canadiens': 'MTL', 'canadiens': 'MTL', 'mtl': 'MTL',
    'nashville predators': 'NSH', 'predators': 'NSH', 'nsh': 'NSH',
    'new jersey devils': 'NJD', 'devils': 'NJD', 'njd': 'NJD', 'nj': 'NJD',
    'new york islanders': 'NYI', 'islanders': 'NYI', 'nyi': 'NYI',
    'new york rangers': 'NYR', 'rangers': 'NYR', 'nyr': 'NYR',
    'ottawa senators': 'OTT', 'senators': 'OTT', 'ott': 'OTT',
    'philadelphia flyers': 'PHI', 'flyers': 'PHI', 'phi': 'PHI',
    'pittsburgh penguins': 'PIT', 'penguins': 'PIT', 'pit': 'PIT',
    'san jose sharks': 'SJS', 'sharks': 'SJS', 'sjs': 'SJS', 'sj': 'SJS',
    'seattle kraken': 'SEA', 'kraken': 'SEA', 'sea': 'SEA',
    'st. louis blues': 'STL', 'blues': 'STL', 'stl': 'STL',
    'tampa bay lightning': 'TBL', 'lightning': 'TBL', 'tbl': 'TBL', 'tb': 'TBL',
    'toronto maple leafs': 'TOR', 'maple leafs': 'TOR', 'leafs': 'TOR', 'tor': 'TOR',
    'utah hockey club': 'UTA', 'utah': 'UTA', 'uta': 'UTA', 'arizona coyotes': 'UTA', 'coyotes': 'UTA',
    'vancouver canucks': 'VAN', 'canucks': 'VAN', 'van': 'VAN',
    'vegas golden knights': 'VGK', 'golden knights': 'VGK', 'vgk': 'VGK', 'vegas': 'VGK',
    'washington capitals': 'WSH', 'capitals': 'WSH', 'wsh': 'WSH', 'was': 'WSH',
    'winnipeg jets': 'WPG', 'jets': 'WPG', 'wpg': 'WPG'
}


def normalize_name(name, lookup):
    cleaned = str(name).strip().lower()
    return lookup.get(cleaned, cleaned.upper()[:3])

def train_nfl():
    print('--> Entrenando NFL...')
    url = 'https://github.com/nflverse/nfldata/raw/master/data/games.csv'
    df = pd.read_csv(url)
    df = df[df['home_score'].notna() & df['away_score'].notna()].sort_values('gameday').copy()
    elo = {team: 1500.0 for team in set(df['home_team'].unique()) | set(df['away_team'].unique())}
    HOME_ADV = 48.0
    records = []
    for _, row in df.iterrows():
        h = row['home_team']
        a = row['away_team']
        h_score = float(row['home_score'])
        a_score = float(row['away_score'])
        margin = h_score - a_score
        h_win = 1 if margin > 0 else 0 if margin < 0 else 0.5
        h_elo = elo.get(h, 1500.0)
        a_elo = elo.get(a, 1500.0)
        elo_diff = (h_elo + HOME_ADV) - a_elo
        expected_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        records.append({
            'season': row.get('season', 2024),
            'home': h,
            'away': a,
            'elo_diff': elo_diff,
            'home_win': 1 if margin > 0 else 0,
            'margin': margin,
            'total_pts': h_score + a_score
        })
        k = 20.0 * (((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * abs(elo_diff)))
        elo[h] = h_elo + k * (h_win - expected_h)
        elo[a] = a_elo + k * ((1 - h_win) - (1 - expected_h))

    df_train = pd.DataFrame(records)
    recent = df_train[df_train['season'] >= 2012].copy()
    X = recent[['elo_diff']]
    model_win = LogisticRegression().fit(X, recent['home_win'])
    model_margin = Ridge().fit(X, recent['margin'])
    model_total = Ridge().fit(X, recent['total_pts'])

    joblib.dump(model_win, os.path.join(BASE_DIR, 'modelo_nfl_win.pkl'))
    joblib.dump(model_margin, os.path.join(BASE_DIR, 'modelo_nfl_margin.pkl'))
    joblib.dump(model_total, os.path.join(BASE_DIR, 'modelo_nfl_total.pkl'))
    with open(os.path.join(BASE_DIR, 'nfl_elo_ratings.json'), 'w') as f:
        json.dump({k: round(v, 1) for k, v in elo.items()}, f, indent=2)
    print(f'NFL listo con {len(recent)} juegos.')

def train_mlb():
    print('--> Entrenando MLB...')
    games = []
    for season in [2023, 2024]:
        url = f'https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={season}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                for d in data.get('dates', []):
                    for g in d.get('games', []):
                        if g.get('status', {}).get('abstractGameState') == 'Final':
                            h_team = g['teams']['home']['team']['name']
                            a_team = g['teams']['away']['team']['name']
                            h_score = g['teams']['home'].get('score')
                            a_score = g['teams']['away'].get('score')
                            if h_score is not None and a_score is not None:
                                games.append({
                                    'home': normalize_name(h_team, MLB_TEAMS),
                                    'away': normalize_name(a_team, MLB_TEAMS),
                                    'h_score': float(h_score),
                                    'a_score': float(a_score)
                                })
        except Exception as e:
            print(f'Aviso MLB {season}: {e}')

    df = pd.DataFrame(games)
    elo = {team: 1500.0 for team in set(df['home'].unique()) | set(df['away'].unique())}
    runs_scored = {team: 10.0 for team in elo}
    runs_allowed = {team: 10.0 for team in elo}
    HOME_ADV = 24.0
    records = []
    for _, row in df.iterrows():
        h = row['home']
        a = row['away']
        h_score = row['h_score']
        a_score = row['a_score']
        margin = h_score - a_score
        h_win = 1 if margin > 0 else 0
        h_elo = elo.get(h, 1500.0)
        a_elo = elo.get(a, 1500.0)
        elo_diff = (h_elo + HOME_ADV) - a_elo
        h_rs = runs_scored.get(h, 50.0)
        h_ra = runs_allowed.get(h, 50.0)
        a_rs = runs_scored.get(a, 50.0)
        a_ra = runs_allowed.get(a, 50.0)
        h_pyth = (h_rs ** 1.83) / ((h_rs ** 1.83) + (h_ra ** 1.83) + 1e-5)
        a_pyth = (a_rs ** 1.83) / ((a_rs ** 1.83) + (a_ra ** 1.83) + 1e-5)
        pyth_diff = h_pyth - a_pyth
        records.append({
            'elo_diff': elo_diff,
            'pyth_diff': pyth_diff,
            'home_win': h_win,
            'runline_cover': 1 if margin >= 2 else 0,
            'total_runs': h_score + a_score
        })
        expected_h = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        elo[h] = h_elo + 4.0 * (h_win - expected_h)
        elo[a] = a_elo + 4.0 * ((1 - h_win) - (1 - expected_h))
        runs_scored[h] += h_score
        runs_allowed[h] += a_score
        runs_scored[a] += a_score
        runs_allowed[a] += h_score

    df_train = pd.DataFrame(records)
    X = df_train[['elo_diff', 'pyth_diff']]
    model_win = LogisticRegression().fit(X, df_train['home_win'])
    model_runline = LogisticRegression().fit(X, df_train['runline_cover'])
    model_total = Ridge().fit(X, df_train['total_runs'])
    joblib.dump(model_win, os.path.join(BASE_DIR, 'modelo_mlb_win.pkl'))
    joblib.dump(model_runline, os.path.join(BASE_DIR, 'modelo_mlb_runline.pkl'))
    joblib.dump(model_total, os.path.join(BASE_DIR, 'modelo_mlb_total.pkl'))
    with open(os.path.join(BASE_DIR, 'mlb_elo_ratings.json'), 'w') as f:
        json.dump({k: round(v, 1) for k, v in elo.items()}, f, indent=2)
    print(f'MLB listo con {len(df_train)} juegos.')

def train_nba():
    print('--> Entrenando NBA...')
    url = 'https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-elo/nbaallelo.csv'
    df = pd.read_csv(url, low_memory=False)
    df = df[(df['year_id'] >= 2005) & (df['_iscopy'] == 0)].copy()
    records = []
    for _, row in df.iterrows():
        h = normalize_name(row['team_id'], NBA_TEAMS)
        a = normalize_name(row['opp_id'], NBA_TEAMS)
        h_pts = float(row['pts'])
        a_pts = float(row['opp_pts'])
        h_elo = float(row['elo_i'])
        a_elo = float(row['opp_elo_i'])
        HOME_ADV = 100.0
        elo_diff = (h_elo + HOME_ADV) - a_elo
        margin = h_pts - a_pts
        records.append({
            'elo_diff': elo_diff,
            'home_win': 1 if margin > 0 else 0,
            'margin': margin,
            'total_pts': h_pts + a_pts,
            'h_team': h,
            'h_elo_end': float(row['elo_n'])
        })
    df_train = pd.DataFrame(records)
    X = df_train[['elo_diff']]
    model_win = LogisticRegression().fit(X, df_train['home_win'])
    model_margin = Ridge().fit(X, df_train['margin'])
    model_total = Ridge().fit(X, df_train['total_pts'])
    joblib.dump(model_win, os.path.join(BASE_DIR, 'modelo_nba_win.pkl'))
    joblib.dump(model_margin, os.path.join(BASE_DIR, 'modelo_nba_margin.pkl'))
    joblib.dump(model_total, os.path.join(BASE_DIR, 'modelo_nba_total.pkl'))
    latest_elos = {}
    for team in set(df_train['h_team'].unique()):
        sub = df_train[df_train['h_team'] == team]
        if not sub.empty:
            latest_elos[team] = round(sub.iloc[-1]['h_elo_end'], 1)
    with open(os.path.join(BASE_DIR, 'nba_elo_ratings.json'), 'w') as f:
        json.dump(latest_elos, f, indent=2)
    print(f'NBA listo con {len(df_train)} juegos.')

DEFAULT_NHL_ELOS = {
    'FLA': 1665.0, 'EDM': 1655.4, 'DAL': 1635.0, 'NYR': 1630.4, 'CAR': 1624.6, 'COL': 1621.6,
    'BOS': 1611.2, 'VAN': 1604.6, 'WPG': 1590.0, 'TOR': 1585.0, 'VGK': 1581.2, 'TBL': 1573.4,
    'LAK': 1558.4, 'NSH': 1549.2, 'DET': 1520.0, 'WSH': 1515.0, 'NJD': 1510.4, 'PIT': 1502.8,
    'STL': 1495.0, 'MIN': 1490.8, 'NYI': 1484.6, 'PHI': 1476.8, 'BUF': 1473.8, 'CGY': 1468.0,
    'SEA': 1467.0, 'OTT': 1460.0, 'MTL': 1450.4, 'UTA': 1447.4, 'CBJ': 1431.2, 'ANA': 1420.0,
    'CHI': 1406.4, 'SJS': 1380.4
}

def train_nhl():
    print('--> Entrenando NHL (Hockey Sobre Hielo)...')
    elos = DEFAULT_NHL_ELOS.copy()
    try:
        req = urllib.request.Request('https://api-web.nhle.com/v1/standings/now', headers={'User-Agent': 'ApexBot/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            for entry in data.get('standings', []):
                code = entry.get('teamAbbrev', {}).get('default', '')
                if code in elos:
                    diff = float(entry.get('goalDifferential', 0))
                    elos[code] += (diff * 0.4)
    except Exception as e:
        print(f'Aviso conectando a API oficial NHL: {e}. Usando matriz base de poderio.')

    HOME_ADV = 35.0
    np.random.seed(42)
    records = []
    teams = list(elos.keys())

    for _ in range(12000):
        h, a = np.random.choice(teams, size=2, replace=False)
        h_elo = elos[h]
        a_elo = elos[a]
        elo_diff = (h_elo + HOME_ADV) - a_elo
        p_win = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        h_win = 1 if np.random.rand() < p_win else 0
        exp_margin = (elo_diff / 180.0)
        actual_margin = exp_margin + np.random.normal(0, 1.85)
        exp_total = 6.0 + (abs(elo_diff) / 500.0) * 0.3
        actual_total = max(2.0, exp_total + np.random.normal(0, 1.9))
        records.append({
            'elo_diff': elo_diff,
            'home_win': h_win,
            'margin': round(actual_margin, 2),
            'total_goals': round(actual_total, 2)
        })

    df_train = pd.DataFrame(records)
    X = df_train[['elo_diff']]
    model_win = LogisticRegression().fit(X, df_train['home_win'])
    model_margin = Ridge().fit(X, df_train['margin'])
    model_total = Ridge().fit(X, df_train['total_goals'])

    joblib.dump(model_win, os.path.join(BASE_DIR, 'modelo_nhl_win.pkl'))
    joblib.dump(model_margin, os.path.join(BASE_DIR, 'modelo_nhl_margin.pkl'))
    joblib.dump(model_total, os.path.join(BASE_DIR, 'modelo_nhl_total.pkl'))

    with open(os.path.join(BASE_DIR, 'nhl_elo_ratings.json'), 'w') as f:
        json.dump({k: round(v, 1) for k, v in sorted(elos.items(), key=lambda x: x[1], reverse=True)}, f, indent=2)
    print(f'NHL listo con {len(df_train)} partidos simulados.')


if __name__ == '__main__':
    train_nfl()
    train_mlb()
    train_nba()
    train_nhl()
    print('Entrenamiento US Sports Completado Exitosamente!')
