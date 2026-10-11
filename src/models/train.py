"""
train.py — Module d'entraînement et d'évaluation des modèles de survie.

Ce script charge le dataset processé, gère l'encodage des variables catégorielles,
entraîne les modèles (CoxPH et Random Survival Forest) et exporte le meilleur.

Exécution :
    $ python -m src.models.train
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

def create_survival_y(df: pd.DataFrame, time_col: str = 'time', event_col: str = 'event') -> np.ndarray:
    """
    Transforme les colonnes de temps et d'événement en tableau structuré NumPy
    exigé par scikit-survival.
    """
    return np.array(
        list(zip(df[event_col].astype(bool), df[time_col])),
        dtype=[('event', '?'), ('time', '<f8')]
    )

def train_and_evaluate():
    # 1. Chargement des données préparées par prepare.py
    data_path = Path("data/processed/german_credit_survival.csv")
    if not data_path.exists():
        logger.error(f"Fichier introuvable : {data_path}. Lancez prepare.py d'abord.")
        return

    logger.info("Chargement des données traitées...")
    df = pd.read_csv(data_path)

    # 2. Séparation des features (X) et de la cible (y)
    X = df.drop(columns=['time', 'event', 'default', 'Risk'], errors='ignore')
    y = create_survival_y(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Préprocesseur : Indispensable pour éviter l'erreur sur les strings (ex: 'male')
    logger.info("Création du pipeline de preprocessing...")
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    # Le OneHotEncoder va convertir le texte en nombres, et le StandardScaler va normaliser les montants
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
        ])

    # --- Entraînement Modèle de Cox (Baseline Statistique) ---
    logger.info("Entraînement du modèle de CoxPH...")
    cox_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', CoxPHSurvivalAnalysis(alpha=0.1))
    ])
    cox_pipeline.fit(X_train, y_train)
    cox_pred = cox_pipeline.predict(X_test)
    cox_c_index = concordance_index_censored(y_test['event'], y_test['time'], cox_pred)[0]
    logger.info(f"Concordance Index (Cox) : {cox_c_index:.4f}")

    # --- Entraînement Random Survival Forest (Machine Learning) ---
    logger.info("Entraînement de la Random Survival Forest...")
    rsf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=42))
    ])
    rsf_pipeline.fit(X_train, y_train)
    rsf_pred = rsf_pipeline.predict(X_test)
    rsf_c_index = concordance_index_censored(y_test['event'], y_test['time'], rsf_pred)[0]
    logger.info(f"Concordance Index (RSF) : {rsf_c_index:.4f}")

    # --- Sauvegarde du meilleur modèle ---
    best_pipeline = rsf_pipeline if rsf_c_index > cox_c_index else cox_pipeline
    model_name = "Random Survival Forest" if rsf_c_index > cox_c_index else "CoxPH"
    logger.info(f"Modèle retenu pour la production : {model_name}")

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "survival_model.joblib"
    
    # Exporte le pipeline complet pour l'API
    joblib.dump(best_pipeline, model_path)
    logger.info(f"✅ Pipeline sauvegardé avec succès dans : {model_path}")

if __name__ == "__main__":
    train_and_evaluate()