import pandas as pd
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset generado
DATA_PATH = "../models/training_dataset_2024.csv"
df = pd.read_csv(DATA_PATH)

# Verificar que las columnas necesarias están presentes
required_columns = [
    "matchday",
    "home_avg_gf", "home_avg_ga", "away_avg_gf", "away_avg_ga",
    "home_goal_diff", "away_goal_diff", "goal_diff_delta",
    "gf_delta", "ga_delta",
    "home_position", "away_position", "position_delta",
    "home_avg_pts", "away_avg_pts", "pts_delta",
    "result"
]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ El dataset no contiene las columnas necesarias: {missing_cols}")

# Features y target
X = df[[
    "matchday",
    "home_avg_gf", "home_avg_ga", "home_goal_diff",
    "away_avg_gf", "away_avg_ga", "away_goal_diff",
    "goal_diff_delta", "gf_delta", "ga_delta",
    "home_position", "away_position", "position_delta",
    "home_avg_pts", "away_avg_pts", "pts_delta"
]]
y = df["result"]

# Comparación con DummyClassifier
dummy = DummyClassifier(strategy="most_frequent")
dummy_scores = cross_val_score(dummy, X, y, cv=5, scoring="accuracy")
print("\n🎯 Accuracy DummyClassifier: {:.2f}%".format(dummy_scores.mean() * 100))

# GridSearch para hiperparámetros
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X, y)
best_model = grid_search.best_estimator_

print("\n✅ Mejor combinación de hiperparámetros:")
print(grid_search.best_params_)

# Validación cruzada
scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
cv_accuracy = scores.mean()
print("\n📊 Accuracy promedio (5-Fold CV): {:.2f}%".format(cv_accuracy * 100))

# Evaluación final con train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
plt.figure(figsize=(6, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión")
plt.savefig("../models/confusion_matrix.png")
print("\n📊 Matriz de confusión guardada en models/confusion_matrix.png")

# Importancia de variables
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index)
plt.title("Importancia de variables")
plt.tight_layout()
plt.savefig("../models/feature_importance.png")
print("\n📈 Gráfico de importancia de variables guardado en models/feature_importance.png")

# Guardar modelo y resultados
os.makedirs("models", exist_ok=True)
MODEL_PATH = "../models/result_predictor_rf.pkl"
joblib.dump(best_model, MODEL_PATH)
print(f"\n💾 Modelo guardado en {MODEL_PATH}")

# Guardar grid search completo
joblib.dump(grid_search, "../models/grid_search_full.pkl")

# Guardar métricas y parámetros
metrics_summary = {
    "cv_accuracy": round(cv_accuracy, 4),
    "dummy_accuracy": round(dummy_scores.mean(), 4),
    "best_params": grid_search.best_params_,
    "classification_report": report
}

with open("../models/metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)
print("\n📝 Métricas y parámetros guardados en models/metrics_summary.json")
