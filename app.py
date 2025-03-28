import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import storage
from google.oauth2 import service_account
import io
import json

from scripts.preprocess_teams import load_all_seasons, process_season_table, add_matchday_column

st.set_page_config(page_title="Análisis LaLiga", layout="wide")


# --- Datos
@st.cache_data
def load_data():
    df_raw = load_all_seasons()
    return df_raw


def load_from_gcs(bucket_name, file_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))


# Recalcular jornadas por orden de partido por equipo y fecha por temporada
def recalculate_matchday(df):
    df = df.copy()
    df = df.sort_values("Date")
    df["home_matchday"] = df.groupby(["season", "HomeTeam"]).cumcount() + 1
    df["away_matchday"] = df.groupby(["season", "AwayTeam"]).cumcount() + 1
    df["matchday"] = df[["home_matchday", "away_matchday"]].max(axis=1)
    return df


df_raw = load_data()
df_raw = recalculate_matchday(df_raw)
available_seasons = sorted(df_raw["season"].unique())
selected_season = st.sidebar.selectbox("📅 Temporada", available_seasons, index=len(available_seasons) - 1)
df_stats = process_season_table(df_raw, season=selected_season)

# --- Menú lateral
view = st.sidebar.radio(":file_folder: Sección", [
    "📊 Clasificación",
    "🏆 Puntos por equipo",
    "🎯 Perfil de equipo (Radar)",
    "🌜 Tarjetas por equipo",
    "⚔️ Comparador de Equipos",
    "📅 Estadísticas de Partidos",
    "🔮 Análisis de Pronósticos",
    "🔮 Predicción de Partidos"
])

# --- Vista 1: Clasificación
if view == "📊 Clasificación":
    st.title(f"📊 Clasificación – {selected_season}")
    df_display = df_stats.sort_values(by=["points", "goal_diff", "goals_for"], ascending=False).reset_index(drop=True)
    df_display.index += 1
    st.dataframe(df_display, use_container_width=True)

# --- Vista 2: Puntos por equipo
elif view == "🏆 Puntos por equipo":
    st.title(f"🏆 Puntos por equipo – {selected_season}")
    fig = px.bar(df_stats.sort_values("points", ascending=True),
                 x="points", y="team", orientation="h",
                 title="Puntos por equipo",
                 labels={"points": "Puntos", "team": "Equipo"})
    st.plotly_chart(fig, use_container_width=True)

# --- Vista 3: Radar de rendimiento
elif view == "🎯 Perfil de equipo (Radar)":
    st.title(f"🌟 Perfil comparativo – {selected_season}")
    team_list = df_stats["team"].sort_values().unique()
    selected_team = st.selectbox("Selecciona un equipo", team_list)

    radar_cols = ["avg_goals_for", "avg_goals_against", "shot_accuracy", "goals_per_shot", "points_per_game"]

    team_row = df_stats[df_stats["team"] == selected_team]
    avg_row = df_stats[radar_cols].mean()

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=team_row[radar_cols].values.flatten(),
        theta=radar_cols,
        fill='toself',
        name=selected_team
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_row.values,
        theta=radar_cols,
        fill='toself',
        name='Promedio Liga'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title=f"🌟 Perfil de rendimiento – {selected_team}"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# --- Vista 4: Tarjetas por equipo
elif view == "🌜 Tarjetas por equipo":
    st.title(f"📊 Tarjetas – {selected_season}")
    fig_cards = px.bar(df_stats.sort_values("cards_total", ascending=False),
                       x="team", y=["yellow", "red"],
                       title="Tarjetas por equipo",
                       labels={"value": "Tarjetas", "team": "Equipo"},
                       barmode="stack")
    st.plotly_chart(fig_cards, use_container_width=True)

# --- Vista 5: Comparador de Equipos
elif view == "⚔️ Comparador de Equipos":
    st.title(f"⚔️ Comparador de Equipos – {selected_season}")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Equipo A", df_stats["team"].unique())
    with col2:
        team_b = st.selectbox("Equipo B", df_stats["team"].unique(), index=1)

    selected_metrics = [
        "points_per_game", "avg_goals_for", "avg_goals_against",
        "shots_per_game", "shots_on_target_per_game", "conversion_rate",
        "fouls_per_game", "cards_per_game", "cards_per_foul",
        "intensity_index", "risk_index"
    ]

    radar_labels = [
        "Puntos/Partido", "Goles a Favor", "Goles en Contra",
        "Tiros", "Tiros a Puerta", "Conversión",
        "Faltas", "Tarjetas", "Tarjetas/Falta",
        "Intensidad", "Riesgo"
    ]

    team_a_data = df_stats[df_stats["team"] == team_a][selected_metrics].values.flatten()
    team_b_data = df_stats[df_stats["team"] == team_b][selected_metrics].values.flatten()

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatterpolar(
        r=team_a_data,
        theta=radar_labels,
        fill='toself',
        name=team_a
    ))
    fig_compare.add_trace(go.Scatterpolar(
        r=team_b_data,
        theta=radar_labels,
        fill='toself',
        name=team_b
    ))

    fig_compare.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title=f"📊 Comparativa de Rendimiento: {team_a} vs {team_b}"
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    df_comparison = df_stats[df_stats["team"].isin([team_a, team_b])][["team"] + selected_metrics].set_index("team").T
    df_comparison.columns = [f"{team_a}", f"{team_b}"]
    st.dataframe(df_comparison.style.format("{:.2f}"), use_container_width=True)

# --- Vista 6: Estadísticas de Partidos
elif view == "📅 Estadísticas de Partidos":
    st.title(f"📅 Estadísticas de Partidos – {selected_season}")
    df_matches = df_raw[df_raw["season"] == selected_season].copy()
    df_matches["Resultado"] = df_matches["FTHG"].astype(str) + " - " + df_matches["FTAG"].astype(str)
    df_matches["Partido"] = df_matches["HomeTeam"] + " vs " + df_matches["AwayTeam"]

    all_teams = sorted(df_matches["HomeTeam"].unique())
    team_filter = st.selectbox("Filtrar por equipo (opcional)", ["Todos"] + all_teams)

    if team_filter != "Todos":
        df_matches = df_matches[(df_matches["HomeTeam"] == team_filter) | (df_matches["AwayTeam"] == team_filter)]

    columns_to_show = [
        "Date", "matchday", "Partido", "Resultado", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HY", "AY", "HR", "AR"
    ]

    df_display = df_matches[columns_to_show].rename(columns={
        "Date": "Fecha", "matchday": "Jornada", "FTHG": "Goles Local", "FTAG": "Goles Visitante",
        "HS": "Tiros Local", "AS": "Tiros Visitante",
        "HST": "Tiros Puerta Local", "AST": "Tiros Puerta Visitante",
        "HY": "Amarillas Local", "AY": "Amarillas Visitante",
        "HR": "Rojas Local", "AR": "Rojas Visitante"
    })

    st.dataframe(df_display.sort_values("Fecha"), use_container_width=True)

    st.subheader("📈 Evolución de goles por jornada")
    goals_by_round = df_matches.groupby("matchday")[["FTHG", "FTAG"]].sum()
    goals_by_round["Total"] = goals_by_round["FTHG"] + goals_by_round["FTAG"]

    fig_goals = px.line(goals_by_round.reset_index(),
                        x="matchday", y=["FTHG", "FTAG", "Total"],
                        markers=True,
                        labels={"matchday": "Jornada", "value": "Goles"},
                        title="Goles por jornada (local, visitante y total)")
    st.plotly_chart(fig_goals, use_container_width=True)

    avg_goals = goals_by_round.mean().round(2)
    st.metric("⚽ Media goles local", avg_goals["FTHG"])
    st.metric("Media goles visitante", avg_goals["FTAG"])
    st.metric("Media total", avg_goals["Total"])

# --- Vista 7: Análisis de Pronósticos
elif view == "🔮 Análisis de Pronósticos":
    st.title(f"🔮 Análisis de Pronósticos – {selected_season}")
    df_matches = df_raw[df_raw["season"] == selected_season].copy()


    def label_result(row):
        if row["FTHG"] > row["FTAG"]:
            return "Local"
        elif row["FTHG"] < row["FTAG"]:
            return "Visitante"
        else:
            return "Empate"


    df_matches["Resultado"] = df_matches.apply(label_result, axis=1)

    result_counts = df_matches["Resultado"].value_counts(normalize=True).round(3) * 100
    st.subheader("📊 Distribución de resultados")
    st.bar_chart(result_counts)

    df_matches["Total_Goles"] = df_matches["FTHG"] + df_matches["FTAG"]
    over_metrics = {
        "Over 1.5": (df_matches["Total_Goles"] > 1.5).mean(),
        "Over 2.5": (df_matches["Total_Goles"] > 2.5).mean(),
        "Over 3.5": (df_matches["Total_Goles"] > 3.5).mean()
    }

    st.subheader("📈 Probabilidad de Over (goles por partido)")
    for k, v in over_metrics.items():
        st.metric(k, f"{v * 100:.1f}%")

    avg_home_goals = df_matches["FTHG"].mean()
    avg_away_goals = df_matches["FTAG"].mean()
    st.subheader("⚽ Promedio de goles por partido")
    st.metric("Local", f"{avg_home_goals:.2f}")
    st.metric("Visitante", f"{avg_away_goals:.2f}")
    st.metric("Total", f"{avg_home_goals + avg_away_goals:.2f}")

    st.subheader("🧮 Análisis por equipo")
    all_teams = sorted(set(df_matches["HomeTeam"]).union(set(df_matches["AwayTeam"])))
    selected_team = st.selectbox("Selecciona un equipo", all_teams)

    df_team = df_matches[(df_matches["HomeTeam"] == selected_team) | (df_matches["AwayTeam"] == selected_team)].copy()
    df_team["goles_favor"] = df_team.apply(lambda row: row["FTHG"] if row["HomeTeam"] == selected_team else row["FTAG"],
                                           axis=1)
    df_team["goles_contra"] = df_team.apply(
        lambda row: row["FTAG"] if row["HomeTeam"] == selected_team else row["FTHG"], axis=1)
    df_team["resultado"] = df_team.apply(lambda row: "Victoria" if row["goles_favor"] > row["goles_contra"] else (
        "Empate" if row["goles_favor"] == row["goles_contra"] else "Derrota"), axis=1)

    total_partidos = len(df_team)
    victorias = (df_team["resultado"] == "Victoria").sum()
    empates = (df_team["resultado"] == "Empate").sum()
    derrotas = (df_team["resultado"] == "Derrota").sum()
    goles_favor = df_team["goles_favor"].sum()
    goles_contra = df_team["goles_contra"].sum()

    st.metric("Partidos Jugados", total_partidos)
    st.metric("Victorias", victorias)
    st.metric("Empates", empates)
    st.metric("Derrotas", derrotas)
    st.metric("Goles a Favor", goles_favor)
    st.metric("Goles en Contra", goles_contra)

    # Métricas adicionales: desglose local/visitante
    home_matches = df_matches[df_matches["HomeTeam"] == selected_team]
    away_matches = df_matches[df_matches["AwayTeam"] == selected_team]

    st.subheader("🏟️ Promedios según condición")
    st.metric("Goles a favor (local)", f"{home_matches['FTHG'].mean():.2f}")
    st.metric("Goles en contra (local)", f"{home_matches['FTAG'].mean():.2f}")
    st.metric("Goles a favor (visitante)", f"{away_matches['FTAG'].mean():.2f}")
    st.metric("Goles en contra (visitante)", f"{away_matches['FTHG'].mean():.2f}")

    fig_team_results = px.bar(df_team, x="matchday", y=["goles_favor", "goles_contra"],
                              title=f"Goles por jornada – {selected_team}",
                              labels={"value": "Goles", "matchday": "Jornada"})
    st.plotly_chart(fig_team_results, use_container_width=True)

elif view == "🔮 Predicción de Partidos":
    st.title("🔮 Predicción de Resultado de Partido")

    import joblib
    import pandas as pd

    # Cargar modelo entrenado
    model = joblib.load("models/result_predictor_rf.pkl")

    # Cargar dataset de stats
    df_stats = pd.read_csv("models/training_dataset_2024.csv")

    equipos = sorted(set(df_stats["home_team"]).union(set(df_stats["away_team"])))
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Equipo local", equipos)
    with col2:
        away_team = st.selectbox("🚌 Equipo visitante", equipos, index=1)

    if home_team == away_team:
        st.warning("⚠️ Selecciona equipos diferentes.")
    else:
        home_data = df_stats[df_stats["home_team"] == home_team].iloc[-1]
        away_data = df_stats[df_stats["away_team"] == away_team].iloc[-1]

        # Mostrar tabla comparativa
        st.markdown("### 📋 Estadísticas comparativas:")
        stats_df = pd.DataFrame({
            "Equipo": [home_team, away_team],
            "GF": [home_data["home_avg_gf"], away_data["away_avg_gf"]],
            "GC": [home_data["home_avg_ga"], away_data["away_avg_ga"]],
            "DG": [home_data["home_goal_diff"], away_data["away_goal_diff"]],
            "Pts/Partido": [home_data["home_avg_pts"], away_data["away_avg_pts"]],
            "Posición": [home_data["home_position"], away_data["away_position"]]
        }).set_index("Equipo")
        st.table(stats_df)

        # Crear features para la predicción
        features = {
            "matchday": max(home_data["matchday"], away_data["matchday"]),
            "home_avg_gf": home_data["home_avg_gf"],
            "home_avg_ga": home_data["home_avg_ga"],
            "home_goal_diff": home_data["home_goal_diff"],
            "away_avg_gf": away_data["away_avg_gf"],
            "away_avg_ga": away_data["away_avg_ga"],
            "away_goal_diff": away_data["away_goal_diff"],
            "goal_diff_delta": home_data["home_goal_diff"] - away_data["away_goal_diff"],
            "gf_delta": home_data["home_avg_gf"] - away_data["away_avg_gf"],
            "ga_delta": home_data["home_avg_ga"] - away_data["away_avg_ga"],
            "home_position": home_data["home_position"],
            "away_position": away_data["away_position"],
            "position_delta": home_data["home_position"] - away_data["away_position"],
            "home_avg_pts": home_data["home_avg_pts"],
            "away_avg_pts": away_data["away_avg_pts"],
            "pts_delta": home_data["home_avg_pts"] - away_data["away_avg_pts"]
        }

        input_df = pd.DataFrame([features])

        if st.button("🔎 Predecir resultado"):
            pred_result = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]

            st.success(f"📌 Predicción: **{pred_result}**")
            st.markdown(f"**Probabilidad de victoria para {home_team}:** {proba[0] * 100:.2f}%")
            st.markdown(f"**Probabilidad de empate:** {proba[1] * 100:.2f}%")
            st.markdown(f"**Probabilidad de victoria para {away_team}:** {proba[2] * 100:.2f}%")
