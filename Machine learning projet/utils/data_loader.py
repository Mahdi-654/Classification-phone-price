import pandas as pd

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, sep=";")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None
