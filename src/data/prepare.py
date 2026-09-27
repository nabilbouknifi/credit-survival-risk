"""
prepare.py — Pipeline principal de préparation des données.

Point d'entrée CLI pour exécuter le pipeline complet :
    $ python -m src.data.prepare

Ce script orchestre les modules loader et target_generator pour produire
le dataset final prêt à l'entraînement dans data/processed/.

Auteur : Pipeline Credit Survival Risk
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
import pandas as pd

from src.data.loader import load_raw_data, clean_data, get_feature_summary
from src.data.target_generator import compute_risk_score, simulate_survival_target

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    """Charge la configuration YAML du projet.

    Args:
        config_path: Chemin vers le fichier de configuration.

    Returns:
        Dictionnaire de configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(
            "Fichier de config introuvable (%s), utilisation des valeurs par défaut.",
            config_path,
        )
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration chargée depuis %s", config_path)
    return config


def run_pipeline(config: dict | None = None) -> pd.DataFrame:
    """Exécute le pipeline complet de préparation des données.

    Étapes :
        1. Chargement du CSV brut
        2. Nettoyage et encodage ordinal
        3. Calcul du score de risque latent
        4. Simulation Weibull de time-to-default
        5. Sauvegarde du dataset enrichi

    Args:
        config: Dictionnaire de configuration (optionnel).

    Returns:
        DataFrame final avec les colonnes time et event.
    """
    config = config or load_config()

    data_config = config.get("data", {})
    sim_config = config.get("target_simulation", {})

    # --- 1. Chargement ---
    raw_path = data_config.get("raw_path", "data/raw/german_credit_data.csv")
    logger.info("━" * 60)
    logger.info("ÉTAPE 1/4 — Chargement des données brutes")
    logger.info("━" * 60)
    df = load_raw_data(raw_path)

    # --- 2. Nettoyage ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 2/4 — Nettoyage et encodage ordinal")
    logger.info("━" * 60)
    df = clean_data(
        df,
        saving_ordinal=sim_config.get("saving_ordinal"),
        checking_ordinal=sim_config.get("checking_ordinal"),
        housing_risk=sim_config.get("housing_risk"),
    )

    # Affichage du résumé statistique
    summary = get_feature_summary(df)
    logger.info("Résumé statistique des features :\n%s", summary.to_string())

    # --- 3. Score de risque ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 3/4 — Calcul du score de risque latent")
    logger.info("━" * 60)
    risk_score = compute_risk_score(
        df, weights=sim_config.get("weights"),
    )

    # --- 4. Simulation Weibull ---
    logger.info("━" * 60)
    logger.info("ÉTAPE 4/4 — Simulation Weibull de time-to-default")
    logger.info("━" * 60)
    seed = data_config.get("random_seed", 42)
    df = simulate_survival_target(
        df,
        risk_score=risk_score,
        weibull_shape=sim_config.get("weibull_shape", 1.5),
        base_scale=sim_config.get("base_scale", 48.0),
        random_seed=seed,
    )

    # --- 5. Sauvegarde ---
    output_path = Path(data_config.get(
        "processed_path", "data/processed/german_credit_survival.csv"
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("✅ Dataset sauvegardé → %s (%d lignes)", output_path, len(df))

    # Résumé final
    logger.info("━" * 60)
    logger.info("PIPELINE TERMINÉ — Résumé des variables cibles")
    logger.info("━" * 60)
    logger.info("  time  → min=%d, max=%d, median=%.0f",
                df["time"].min(), df["time"].max(), df["time"].median())
    logger.info("  event → 0 (censuré): %d | 1 (défaut): %d",
                (df["event"] == 0).sum(), (df["event"] == 1).sum())
    logger.info("━" * 60)

    return df


# ------------------------------------------------------------------ #
#  Point d'entrée CLI                                                 #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    logger.info("Démarrage du pipeline de préparation des données...")
    try:
        df_final = run_pipeline()
        logger.info("🎯 Pipeline exécuté avec succès. %d observations prêtes.", len(df_final))
    except Exception as exc:
        logger.exception("❌ Erreur fatale dans le pipeline : %s", exc)
        sys.exit(1)
