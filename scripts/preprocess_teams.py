
import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FOLDER = os.path.join(BASE_DIR, "data", "la-liga")

SEASONS_MAP = {
    "0506": "2005-06",
    "0607": "2006-07",
    "0708": "2007-08",
    "0809": "2008-09",
    "0910": "2009-10",
    "1011": "2010-11",
    "1112": "2011-12",
    "1213": "2012-13",
    "1314": "2013-14",
    "1415": "2014-15",
    "1516": "2015-16",
    "1617": "2016-17",
    "1718": "2017-18",
    "1819": "2018-19",
    "1920": "2019-20",
    "2021": "2020-21",
    "2122": "2021-22",
    "2223": "2022-23",
    "2324": "2023-24",
    "2425": "2024-25"
}


def preparar_all_teams(df):
    # Datos de equipo local
    home = df.rename(columns={
        "HomeTeam": "team",
        "FTHG": "goals_for",
        "FTAG": "goals_against",
        "HS": "shots",
        "HST": "shots_on_target",
        "HF": "fouls",
        "HC": "corners",
        "HY": "yellow",
        "HR": "red"
    })
    home["local"] = True

    # Datos de equipo visitante
    away = df.rename(columns={
        "AwayTeam": "team",
        "FTAG": "goals_for",
        "FTHG": "goals_against",
        "AS": "shots",
        "AST": "shots_on_target",
        "AF": "fouls",
        "AC": "corners",
        "AY": "yellow",
        "AR": "red"
    })
    away["local"] = False

    # Concatenar ambos equipos (local y visitante)
    all_teams = pd.concat([home, away], ignore_index=True)
    all_teams["matches"] = 1  # Aseguramos que cada fila cuenta como un partido

    return all_teams
def add_matchday_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("Date")
    df["matchday"] = df.groupby("season")["Date"].transform(lambda x: pd.factorize(x)[0] + 1)
    return df

def load_all_seasons():
    all_dfs = []
    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".csv") and filename.startswith("season-"):
            df = pd.read_csv(os.path.join(DATA_FOLDER, filename), parse_dates=["Date"], dayfirst=True)
            season_code = filename.replace("season-", "").replace(".csv", "")
            df["season"] = f"20{season_code[:2]}-{season_code[2:]}"  # adaptar si usas otro formato
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

def process_season_table(df: pd.DataFrame, season: str = None) -> pd.DataFrame:
    if season:
        df = df[df["season"] == season]

    # Paso 1: Preparar la tabla de equipos por partido
    all_teams = preparar_all_teams(df)

    # Paso 2: Calcular los puntos por partido
    def compute_points(row):
        if row["goals_for"] > row["goals_against"]:
            return 3  # Victoria
        elif row["goals_for"] == row["goals_against"]:
            return 1  # Empate
        else:
            return 0  # Derrota

    # Crear los puntos, victorias, empates y derrotas
    all_teams["points"] = all_teams.apply(compute_points, axis=1)
    all_teams["wins"] = all_teams["goals_for"] > all_teams["goals_against"]
    all_teams["draws"] = all_teams["goals_for"] == all_teams["goals_against"]
    all_teams["losses"] = all_teams["goals_for"] < all_teams["goals_against"]

    # Paso 3: Agrupar por equipo y temporada y calcular estadísticas
    grouped = all_teams.groupby(["season", "team"]).agg(
        matches=("matches", "sum"),
        wins=("wins", "sum"),
        draws=("draws", "sum"),
        losses=("losses", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
        points=("points", "sum"),
        shots=("shots", "sum"),
        shots_on_target=("shots_on_target", "sum"),
        fouls=("fouls", "sum"),
        corners=("corners", "sum"),
        yellow=("yellow", "sum"),
        red=("red", "sum")
    ).reset_index()

    # Paso 4: Añadir métricas adicionales
    grouped["goal_diff"] = grouped["goals_for"] - grouped["goals_against"]
    grouped["points_per_game"] = grouped["points"] / grouped["matches"]
    grouped["avg_goals_for"] = grouped["goals_for"] / grouped["matches"]
    grouped["avg_goals_against"] = grouped["goals_against"] / grouped["matches"]
    grouped["shot_accuracy"] = grouped["shots_on_target"] / grouped["shots"].replace(0, 1)
    grouped["goals_per_shot"] = grouped["goals_for"] / grouped["shots"].replace(0, 1)
    grouped["cards_total"] = grouped["yellow"] + grouped["red"]

    # Nuevas métricas ofensivas
    grouped["shots_per_game"] = grouped["shots"] / grouped["matches"]
    grouped["shots_on_target_per_game"] = grouped["shots_on_target"] / grouped["matches"]
    grouped["conversion_rate"] = grouped["goals_for"] / grouped["shots_on_target"].replace(0, 1)
    grouped["goals_per_shot"] = grouped["goals_for"] / grouped["shots"].replace(0, 1)
    grouped["goals_per_shot_on_target"] = grouped["goals_for"] / grouped["shots_on_target"].replace(0, 1)

    # Nuevas métricas defensivas
    grouped["fouls_per_game"] = grouped["fouls"] / grouped["matches"]
    grouped["cards_per_game"] = grouped["cards_total"] / grouped["matches"]
    grouped["cards_per_foul"] = grouped["cards_total"] / grouped["fouls"].replace(0, 1)

    # Índices compuestos
    grouped["risk_index"] = (grouped["fouls"] + grouped["yellow"] + grouped["red"]) / grouped["matches"]
    grouped["intensity_index"] = (grouped["shots"] + grouped["fouls"] + grouped["corners"]) / grouped["matches"]

    # Paso 5: Clasificación por puntos
    grouped = grouped.sort_values(by=["points", "goal_diff", "goals_for"], ascending=False)
    grouped["position"] = grouped["points"].rank(method="first", ascending=False).astype(int)

    return grouped
