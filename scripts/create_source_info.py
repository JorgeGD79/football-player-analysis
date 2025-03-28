import os
import requests

# Temporadas disponibles (ajusta si hay más)
SEASONS = ["0506","0607","0708","0809","0910","1011","1112", "1213","1314","1415","1516","1617","1718","1819","1920", "2021","2122","2223","2324","2425"]
BASE_URL = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/ligue-1/"
TARGET_DIR = "../data/ligue-1"

def download_csvs():
    os.makedirs(TARGET_DIR, exist_ok=True)

    for code in SEASONS:
        filename = f"season-{code}.csv"
        url = BASE_URL + filename
        path = os.path.join(TARGET_DIR, filename)

        if not os.path.exists(path):
            print(f"📥 Descargando {filename}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Guardado en {path}")
            else:
                print(f"❌ Error al descargar {filename}: {response.status_code}")
        else:
            print(f"✔️ {filename} ya está descargado.")

if __name__ == "__main__":
    download_csvs()
