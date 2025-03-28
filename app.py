import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from google.cloud import storage
from google.oauth2 import service_account
import io
import json

from scripts.preprocess_teams import load_all_seasons, process_season_table, add_matchday_column

st.set_page_config(page_title="LaLiga Analysis", layout="wide")

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
selected_season = st.sidebar.selectbox("📅 Season", available_seasons, index=len(available_seasons) - 1)
df_stats = process_season_table(df_raw, season=selected_season)

view = st.sidebar.radio(":file_folder: Section", [
    "🏠 Home",
    "📊 Standings",
    "🏆 Points per Team",
    "🎯 Team Profile (Radar)",
    "🌜 Cards per Team",
    "⚔️ Team Comparison",
    "📅 Match Statistics",
    "🔮 Prediction Analysis",
    "🧬 Team Clustering",
    "🌜 Cards per Team",
    "⚔️ Team Comparison",
    "📅 Match Statistics",
    "🔮 Prediction Analysis"
])

if view == "🏠 Home":
    st.title("🏟️ Welcome to the LaLiga Performance Dashboard")
    st.markdown("""
    This interactive tool allows you to explore, analyze, and compare teams in **LaLiga** across multiple seasons.

    ### 🔍 What can you do here?
    - View full standings and team rankings.
    - Analyze team performance through radar charts.
    - Compare any two teams side by side.
    - Explore detailed match statistics and goal trends.
    - Understand prediction metrics like goal distributions and Over/Under probabilities.

    ### 📅 Available Seasons
    You can select the season from the left sidebar to update all views accordingly.

    ### 📊 Powered by:
    - **Streamlit** for interactive visualization.
    - **Plotly** for advanced charts.
    - **Google Cloud Storage** for secure data hosting.

    ---
    Use the menu on the left to start exploring!
    """)

elif view == "📊 Standings":
    st.title(f"📊 Standings – {selected_season}")
    df_display = df_stats.sort_values(by=["points", "goal_diff", "goals_for"], ascending=False).reset_index(drop=True)
    df_display.index += 1
    st.dataframe(df_display, use_container_width=True)

elif view == "🏆 Points per Team":
    st.title(f"🏆 Points per Team – {selected_season}")
    fig = px.bar(df_stats.sort_values("points", ascending=True),
                 x="points", y="team", orientation="h",
                 title="Points per Team",
                 labels={"points": "Points", "team": "Team"})
    st.plotly_chart(fig, use_container_width=True)

elif view == "🎯 Team Profile (Radar)":
    st.title(f"🌟 Comparative Profile – {selected_season}")
    team_list = df_stats["team"].sort_values().unique()
    selected_team = st.selectbox("Select a team", team_list)

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
        name='League Average'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title=f"🌟 Performance Profile – {selected_team}"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

elif view == "🌜 Cards per Team":
    st.title(f"📊 Cards – {selected_season}")
    fig_cards = px.bar(df_stats.sort_values("cards_total", ascending=False),
                       x="team", y=["yellow", "red"],
                       title="Cards per Team",
                       labels={"value": "Cards", "team": "Team"},
                       barmode="stack")
    st.plotly_chart(fig_cards, use_container_width=True)

elif view == "⚔️ Team Comparison":
    st.title(f"⚔️ Team Comparison – {selected_season}")

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", df_stats["team"].unique())
    with col2:
        team_b = st.selectbox("Team B", df_stats["team"].unique(), index=1)

    selected_metrics = [
        "points_per_game", "avg_goals_for", "avg_goals_against",
        "shots_per_game", "shots_on_target_per_game", "conversion_rate",
        "fouls_per_game", "cards_per_game", "cards_per_foul",
        "intensity_index", "risk_index"
    ]

    radar_labels = [
        "Points/Game", "Goals For", "Goals Against",
        "Shots", "Shots on Target", "Conversion",
        "Fouls", "Cards", "Cards/Foul",
        "Intensity", "Risk"
    ]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_stats = pd.DataFrame(scaler.fit_transform(df_stats[selected_metrics]), columns=selected_metrics, index=df_stats["team"])
    team_a_data = scaled_stats.loc[team_a].values
    team_b_data = scaled_stats.loc[team_b].values

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
        title=f"📊 Performance Comparison: {team_a} vs {team_b}"
    )

    st.plotly_chart(fig_compare, use_container_width=True)

    df_comparison = df_stats[df_stats["team"].isin([team_a, team_b])][["team"] + selected_metrics].set_index("team").T
    df_comparison.columns = [f"{team_a}", f"{team_b}"]
    st.dataframe(df_comparison.style.format("{:.2f}"), use_container_width=True)

elif view == "📅 Match Statistics":
    st.title(f"📅 Match Statistics – {selected_season}")
    df_matches = df_raw[df_raw["season"] == selected_season].copy()
    df_matches["Result"] = df_matches["FTHG"].astype(str) + " - " + df_matches["FTAG"].astype(str)
    df_matches["Match"] = df_matches["HomeTeam"] + " vs " + df_matches["AwayTeam"]

    all_teams = sorted(df_matches["HomeTeam"].unique())
    team_filter = st.selectbox("Filter by team (optional)", ["All"] + all_teams)

    if team_filter != "All":
        df_matches = df_matches[(df_matches["HomeTeam"] == team_filter) | (df_matches["AwayTeam"] == team_filter)]

    columns_to_show = [
        "Date", "matchday", "Match", "Result", "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HY", "AY", "HR", "AR"
    ]

    df_display = df_matches[columns_to_show].rename(columns={
        "Date": "Date", "matchday": "Matchday", "FTHG": "Goals Home", "FTAG": "Goals Away",
        "HS": "Shots Home", "AS": "Shots Away",
        "HST": "Shots on Target Home", "AST": "Shots on Target Away",
        "HY": "Yellows Home", "AY": "Yellows Away",
        "HR": "Reds Home", "AR": "Reds Away"
    })

    st.dataframe(df_display.sort_values("Date"), use_container_width=True)

    st.subheader("📈 Goals evolution per matchday")
    goals_by_round = df_matches.groupby("matchday")[["FTHG", "FTAG"]].sum()
    goals_by_round["Total"] = goals_by_round["FTHG"] + goals_by_round["FTAG"]

    fig_goals = px.line(goals_by_round.reset_index(),
                        x="matchday", y=["FTHG", "FTAG", "Total"],
                        markers=True,
                        labels={"matchday": "Matchday", "value": "Goals"},
                        title="Goals per Matchday (home, away and total)")
    st.plotly_chart(fig_goals, use_container_width=True)

    avg_goals = goals_by_round.mean().round(2)
    st.metric("⚽ Avg home goals", avg_goals["FTHG"])
    st.metric("Avg away goals", avg_goals["FTAG"])
    st.metric("Avg total goals", avg_goals["Total"])

elif view == "🧬 Team Clustering":
    st.title(f"🧬 Team Clustering – {selected_season}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Select relevant features for clustering
    cluster_features = [
    "avg_goals_for",  # more means more offensive
    "avg_goals_against",  # less means more defensive
    "shots_per_game"  # more means more aggressive
]

    X = df_stats[cluster_features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_stats["cluster"] = clusters

    # Optionally adjust cluster centroids for interpretability
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=cluster_features)

    # Compute custom score: more offensive and more shots, less goals against
    combined_score = (
        cluster_centers["avg_goals_for"] +
        cluster_centers["shots_per_game"] -
        cluster_centers["avg_goals_against"]
    )
    sorted_clusters = combined_score.sort_values(ascending=False).index.tolist()

    cluster_labels = {}
    for i, cluster_id in enumerate(sorted_clusters):
        if i == 0:
            cluster_labels[cluster_id] = "Aggressive"
        elif i == 1:
            cluster_labels[cluster_id] = "Balanced"
        else:
            cluster_labels[cluster_id] = "Defensive"

    df_stats["Cluster Type"] = df_stats["cluster"].map(cluster_labels)

    st.subheader("📋 Cluster assignment")
    st.dataframe(df_stats[["team", "Cluster Type"] + cluster_features].sort_values("Cluster Type"), use_container_width=True)

    st.subheader("📊 Cluster Heatmap")
    clustered_data = df_stats.groupby("Cluster Type")[cluster_features].mean().T

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(clustered_data, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("🧠 Cluster Descriptions")
    st.markdown("""
    - **Aggressive**: Teams with high offensive stats, shots, and goal attempts, possibly higher risk.
    - **Defensive**: Teams focused on minimizing goals against, often with fewer shots and conservative styles.
    - **Balanced**: Teams that maintain an equilibrium between attack and defense.
    """)

elif view == "🔮 Prediction Analysis":
    st.title(f"🔮 Prediction Analysis – {selected_season}")
    df_matches = df_raw[df_raw["season"] == selected_season].copy()

    def label_result(row):
        if row["FTHG"] > row["FTAG"]:
            return "Home"
        elif row["FTHG"] < row["FTAG"]:
            return "Away"
        else:
            return "Draw"

    df_matches["Result"] = df_matches.apply(label_result, axis=1)

    result_counts = df_matches["Result"].value_counts(normalize=True).round(3) * 100
    st.subheader("📊 Result distribution")
    st.bar_chart(result_counts)

    df_matches["Total_Goals"] = df_matches["FTHG"] + df_matches["FTAG"]
    over_metrics = {
        "Over 1.5": (df_matches["Total_Goals"] > 1.5).mean(),
        "Over 2.5": (df_matches["Total_Goals"] > 2.5).mean(),
        "Over 3.5": (df_matches["Total_Goals"] > 3.5).mean()
    }

    st.subheader("📈 Over probabilities (goals per match)")
    for k, v in over_metrics.items():
        st.metric(k, f"{v * 100:.1f}%")

    avg_home_goals = df_matches["FTHG"].mean()
    avg_away_goals = df_matches["FTAG"].mean()
    st.subheader("⚽ Average goals per match")
    st.metric("Home", f"{avg_home_goals:.2f}")
    st.metric("Away", f"{avg_away_goals:.2f}")
    st.metric("Total", f"{avg_home_goals + avg_away_goals:.2f}")
