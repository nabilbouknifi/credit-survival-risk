"""
loader.py — Chargement et nettoyage du dataset German Credit.

Responsabilités :
    1. Charger le CSV brut depuis data/raw/
    2. Normaliser les noms de colonnes (espaces → underscores)
    3. Traiter les valeurs manquantes ("NA" textuelles → np.nan)
    4. Encoder ordinalement les variables catégorielles pour la modélisation
    5. Retourner un DataFrame prêt pour la génération de la variable cible

Auteur : Pipeline Credit Survival Risk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Constantes par défaut (surchargées par config.yaml en production)  #
# ------------------------------------------------------------------ #

DEFAULT_SAVING_ORDINAL: dict[str, int] = {
    "NA": 0,
    "little": 1,
    "moderate": 2,
    "quite rich": 3,
    "rich": 4,
}

DEFAULT_CHECKING_ORDINAL: dict[str, int] = {
    "NA": 0,
    "little": 1,
    "moderate": 2,
    "rich": 3,
}

DEFAULT_HOUSING_RISK: dict[str, float] = {
    "rent": 0.8,
    "free": 0.5,
    "own": 0.2,
}


# ------------------------------------------------------------------ #
#  Fonctions publiques                                                #
# ------------------------------------------------------------------ #


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Charge le CSV brut et effectue les validations de base.

    Args:
        filepath: Chemin vers le fichier CSV German Credit.

    Returns:
        DataFrame brut avec index réinitialisé.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si des colonnes attendues sont absentes.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Fichier introuvable : {filepath.resolve()}")

    logger.info("Chargement de %s", filepath)
    df = pd.read_csv(filepath, index_col=0)

    # Colonnes minimales attendues
    expected_cols = {
        "Age", "Sex", "Job", "Housing",
        "Saving accounts", "Checking account",
        "Credit amount", "Duration", "Purpose",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    logger.info("Données chargées : %d lignes × %d colonnes", *df.shape)
    return df


def clean_data(
    df: pd.DataFrame,
    saving_ordinal: dict[str, int] | None = None,
    checking_ordinal: dict[str, int] | None = None,
    housing_risk: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Nettoie et encode le DataFrame pour l'analyse de survie.

    Étapes :
        1. Normalisation des noms de colonnes
        2. Remplacement des "NA" textuels par np.nan
        3. Encodage ordinal des variables catégorielles
        4. Imputation par la médiane pour les valeurs manquantes numériques

    Args:
        df: DataFrame brut issu de load_raw_data().
        saving_ordinal: Mapping ordinal pour Saving accounts.
        checking_ordinal: Mapping ordinal pour Checking account.
        housing_risk: Mapping de score de risque pour Housing.

    Returns:
        DataFrame nettoyé avec colonnes normalisées et encodées.
    """
    saving_ordinal = saving_ordinal or DEFAULT_SAVING_ORDINAL
    checking_ordinal = checking_ordinal or DEFAULT_CHECKING_ORDINAL
    housing_risk = housing_risk or DEFAULT_HOUSING_RISK

    df = df.copy()

    # 1. Normalisation des noms de colonnes
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    logger.info("Colonnes normalisées : %s", list(df.columns))

    # 2. Remplacement des "NA" textuels → vrai NaN, puis traitement
    #    (Le CSV encode les valeurs manquantes comme la chaîne "NA")
    for col in ["saving_accounts", "checking_account"]:
        df[col] = df[col].replace("NA", np.nan)

    # 3. Encodage ordinal — on conserve l'original en suffixe _raw
    df["saving_accounts_raw"] = df["saving_accounts"].copy()
    df["checking_account_raw"] = df["checking_account"].copy()
    df["housing_raw"] = df["housing"].copy()

    df["saving_accounts_ord"] = (
        df["saving_accounts"]
        .map(saving_ordinal)
        .fillna(saving_ordinal.get("NA", 0))  # NaN → score le plus risqué
        .astype(int)
    )

    df["checking_account_ord"] = (
        df["checking_account"]
        .map(checking_ordinal)
        .fillna(checking_ordinal.get("NA", 0))
        .astype(int)
    )

    df["housing_risk"] = (
        df["housing"]
        .map(housing_risk)
        .fillna(0.5)  # Fallback conservateur
        .astype(float)
    )

    # 4. Encodage binaire du sexe
    df["sex_male"] = (df["sex"] == "male").astype(int)

    # 5. Log-transform du montant de crédit (distribution asymétrique)
    df["log_credit_amount"] = np.log1p(df["credit_amount"])

    n_before = len(df)
    df = df.dropna(subset=["age", "duration", "credit_amount", "job"])
    n_after = len(df)
    if n_before != n_after:
        logger.warning(
            "Suppression de %d lignes avec NaN sur les colonnes critiques",
            n_before - n_after,
        )

    logger.info("Nettoyage terminé : %d lignes × %d colonnes", *df.shape)
    return df


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Retourne un résumé statistique des features clés.

    Utile pour le rapport de soutenance et le diagnostic rapide.
    """
    cols_of_interest = [
        "age", "credit_amount", "duration", "job",
        "saving_accounts_ord", "checking_account_ord", "housing_risk",
    ]
    existing = [c for c in cols_of_interest if c in df.columns]
    return df[existing].describe().round(2)
