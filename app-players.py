import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo
model = joblib.load("models/result_predictor_rf.pkl")

# Cargar dataset de entrenamiento
train_df = pd.read_csv("models/training_dataset_2024.csv")

# Equipos disponibles en el dataset
equipos = sorted(set(train_df["home_team"]).union(set(train_df["away_team"])))

# Títulos de la app
st.title("🔮 Predicción de Resultados de Fútbol")
st.markdown("Selecciona los equipos y predice el resultado.")

# Crear las opciones de selección de equipos
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("🏠 Equipo local", equipos)
with col2:
    away_team = st.selectbox("🚌 Equipo visitante", equipos, index=1)

if home_team == away_team:
    st.warning("⚠️ Selecciona equipos diferentes.")
else:
    # Filtrar las estadísticas de los equipos seleccionados
    home_data = train_df[train_df["home_team"] == home_team].iloc[-1]
    away_data = train_df[train_df["away_team"] == away_team].iloc[-1]

    # Mostrar la tabla comparativa de estadísticas
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

    # Crear las features para la predicción
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

    # Predicción del resultado
    if st.button("🔎 Predecir resultado"):
        pred_result = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        st.success(f"📌 Predicción: **{pred_result}**")
        st.markdown(f"**Probabilidad de victoria para {home_team}:** {proba[0] * 100:.2f}%")
        st.markdown(f"**Probabilidad de empate:** {proba[1] * 100:.2f}%")
        st.markdown(f"**Probabilidad de victoria para {away_team}:** {proba[2] * 100:.2f}%")
