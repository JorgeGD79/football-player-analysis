# 🏟️ LaLiga Performance Dashboard

An interactive web dashboard built with **Streamlit** that analyzes performance metrics of teams in **LaLiga** across multiple seasons.

> ⚠️ **Disclaimer**: This project is **not affiliated with, endorsed by, or connected to LaLiga or any of its official organizations**. It is an independent educational and technical initiative using publicly available data.

---

## 📊 Features

- **Standings**: League table by season.
- **Points per Team**: Bar charts comparing total points.
- **Team Radar Profile**: Offensive/defensive metrics compared to league averages.
- **Cards per Team**: Yellow and red card distributions.
- **Team Comparison**: Radar chart comparing two teams, normalized across all features.
- **Match Statistics**: Game-by-game stats, trends, and filters by team.
- **Prediction Analysis**: Distribution of results and over/under goal metrics.
- **Team Clustering**: Group teams as Aggressive, Balanced, or Defensive based on goal and shot stats.

---

## 📂 Data

The dashboard uses team and match statistics stored in **Google Cloud Storage (GCS)**, and processed locally using `pandas`.

---

## 🧰 Tech Stack

- **Streamlit** – interactive dashboard interface  
- **Plotly** – charts (bar, radar, line, polar)  
- **pandas** – data handling  
- **scikit-learn** – clustering, scaling  
- **Google Cloud Storage** – data storage and loading

---

## 🚀 Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/laliga-dashboard.git
   cd laliga-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your `secrets.toml` with your GCS credentials:
   ```toml
   GCS_CREDENTIALS_JSON = "..."
   GCS_BUCKET = "your-bucket-name"
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 📌 Author

Developed by Jorge.  
Open to feedback and contributions!