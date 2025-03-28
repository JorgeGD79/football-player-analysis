import pandas as pd
import os
from scripts.preprocess_teams import load_all_seasons

CURRENT_SEASON = "2024-25"
df = load_all_seasons()
df = df[df["season"] == CURRENT_SEASON].copy()
df = df.sort_values("Date")

# Calcular jornada por equipo y temporada
df["home_matchday"] = df.groupby(["season", "HomeTeam"]).cumcount() + 1
df["away_matchday"] = df.groupby(["season", "AwayTeam"]).cumcount() + 1
df["matchday"] = df[["home_matchday", "away_matchday"]].max(axis=1)

rows = []

def get_stats(team, local_col, visitor_col, gf_col, ga_col):
    # Filtrar los partidos previos
    local_games = prev_matches[prev_matches[local_col] == team]
    visitor_games = prev_matches[prev_matches[visitor_col] == team]
    games_played = len(local_games) + len(visitor_games)

    # Goles a favor y goles en contra
    gf = local_games[gf_col].sum() + visitor_games[gf_col].sum()
    ga = local_games[ga_col].sum() + visitor_games[ga_col].sum()

    # Calcular victorias, empates y derrotas
    wins = ((local_games[gf_col] > local_games[ga_col]).sum() +
            (visitor_games[gf_col] > visitor_games[ga_col]).sum())
    draws = ((local_games[gf_col] == local_games[ga_col]).sum() +
             (visitor_games[gf_col] == visitor_games[ga_col]).sum())
    losses = games_played - wins - draws  # Derrotas

    # Calcular puntos
    pts = wins * 3 + draws

    return {
        "avg_gf": gf / games_played if games_played else 0,
        "avg_ga": ga / games_played if games_played else 0,
        "goal_diff": (gf - ga) / games_played if games_played else 0,
        "pts_per_game": pts / games_played if games_played else 0,
        "games": games_played
    }

for idx, row in df.iterrows():
    matchday = row["matchday"]
    if matchday < 3:
        continue

    home = row["HomeTeam"]
    away = row["AwayTeam"]
    date = row["Date"]
    prev_matches = df[df["Date"] < date]

    # Obtener estadísticas de los equipos
    stats_home = get_stats(home, "HomeTeam", "AwayTeam", "FTHG", "FTAG")
    stats_away = get_stats(away, "AwayTeam", "HomeTeam", "FTAG", "FTHG")

    # Calcular clasificación estimada por goles a favor
    teams = prev_matches["HomeTeam"].unique().tolist()
    team_stats = []
    for t in teams:
        s = get_stats(t, "HomeTeam", "AwayTeam", "FTHG", "FTAG")
        team_stats.append((t, s["avg_gf"]))
    sorted_teams = sorted(team_stats, key=lambda x: x[1], reverse=True)
    ranking = {team: pos + 1 for pos, (team, _) in enumerate(sorted_teams)}

    # Asignar posición al equipo
    home_position = ranking.get(home, len(teams) + 1)
    away_position = ranking.get(away, len(teams) + 1)

    # Determinar el resultado del partido
    result = "Empate"
    if row["FTHG"] > row["FTAG"]:
        result = "Local"
    elif row["FTHG"] < row["FTAG"]:
        result = "Visitante"

    # Crear un diccionario con los datos del partido
    row_dict = {
        "matchday": matchday,
        "home_team": home,
        "away_team": away,
        "home_avg_gf": stats_home["avg_gf"],
        "home_avg_ga": stats_home["avg_ga"],
        "away_avg_gf": stats_away["avg_gf"],
        "away_avg_ga": stats_away["avg_ga"],
        "home_goal_diff": stats_home["goal_diff"],
        "away_goal_diff": stats_away["goal_diff"],
        "goal_diff_delta": stats_home["goal_diff"] - stats_away["goal_diff"],
        "gf_delta": stats_home["avg_gf"] - stats_away["avg_gf"],
        "ga_delta": stats_home["avg_ga"] - stats_away["avg_ga"],
        "home_position": home_position,
        "away_position": away_position,
        "position_delta": home_position - away_position,
        "home_avg_pts": stats_home["pts_per_game"],
        "away_avg_pts": stats_away["pts_per_game"],
        "pts_delta": stats_home["pts_per_game"] - stats_away["pts_per_game"],
        "result": result
    }
    rows.append(row_dict)

# Guardar dataset en un archivo CSV
train_df = pd.DataFrame(rows)
os.makedirs("models", exist_ok=True)
train_df.to_csv("../models/training_dataset_2024.csv", index=False)
print("✅ Dataset de entrenamiento guardado en models/training_dataset_2024.csv")
