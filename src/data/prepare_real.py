import pandas as pd
from pathlib import Path

def process_real_data():
    print("Chargement des vraies données du binôme...")
    df = pd.read_csv("data/raw/german_credit_survival.csv")

    # Nettoyage métier des comptes bancaires
    df["Saving accounts"] = df["Saving accounts"].fillna("none")
    df["Checking account"] = df["Checking account"].fillna("none")

    # Formatage des colonnes pour scikit-survival
    df["time"] = df["Duration"]
    df["event"] = df["default"]

    # Sauvegarde
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "german_credit_survival.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Données propres sauvegardées dans {out_path}")

if __name__ == "__main__":
    process_real_data()