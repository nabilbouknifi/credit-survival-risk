"""
target_generator.py — Simulation réaliste de la variable cible time-to-default.

Contexte académique :
    Le dataset German Credit ne contient pas de variable d'événement de défaut
    temporelle. Ce module simule un processus de survie réaliste via une
    distribution de Weibull, paramétré par un score de risque latent calculé
    à partir des covariables du prêt.

Modèle génératif :
    1. On calcule un score de risque r ∈ [0, 1] pour chaque emprunteur
       à partir de ses caractéristiques (montant, épargne, âge, logement...).
    2. On paramètre l'échelle λ d'une distribution Weibull(k, λ) :
       λ = base_scale × (1 - α × r), où α contrôle l'amplitude de l'effet.
       Plus r est élevé → λ petit → défaut plus rapide.
    3. On tire T* ~ Weibull(k, λ) pour chaque emprunteur.
    4. Censure administrative : si T* > Duration du prêt →
       l'emprunteur est censuré (event=0, time=Duration).
       Sinon → event=1, time=T*.

Références :
    - Collett, D. (2015). Modelling Survival Data in Medical Research.
    - Hosmer, D.W. et al. (2008). Applied Survival Analysis.

Auteur : Pipeline Credit Survival Risk
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_risk_score(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Calcule un score de risque latent ∈ [0, 1] pour chaque emprunteur.

    Le score combine les covariables normalisées selon les poids fournis.
    Un score élevé indique un profil à haut risque de défaut.

    Args:
        df: DataFrame nettoyé (issu de loader.clean_data).
        weights: Poids des composantes du score.
            Clés attendues : credit_amount, saving_accounts, checking_account,
                             age, housing, job.

    Returns:
        Série de scores de risque normalisés dans [0, 1].
    """
    if weights is None:
        weights = {
            "credit_amount": 0.30,
            "saving_accounts": 0.25,
            "checking_account": 0.20,
            "age": 0.10,
            "housing": 0.10,
            "job": 0.05,
        }

    # --- Normalisation min-max de chaque composante en [0, 1] ---
    # Montant du crédit : plus c'est élevé, plus c'est risqué
    credit_norm = _min_max_normalize(df["credit_amount"])

    # Épargne : score ordinal inversé (0 = pas d'épargne = risque max)
    saving_max = df["saving_accounts_ord"].max()
    saving_risk = 1.0 - (df["saving_accounts_ord"] / max(saving_max, 1))

    # Compte courant : même logique inversée
    checking_max = df["checking_account_ord"].max()
    checking_risk = 1.0 - (df["checking_account_ord"] / max(checking_max, 1))

    # Âge : les jeunes emprunteurs sont statistiquement plus risqués
    age_risk = 1.0 - _min_max_normalize(df["age"])

    # Logement : directement le score de risque issu du loader
    housing_risk = df["housing_risk"]

    # Job : les niveaux bas (0, 1) sont plus risqués que (2, 3)
    job_max = df["job"].max()
    job_risk = 1.0 - (df["job"] / max(job_max, 1))

    # --- Score composite pondéré ---
    risk_score = (
        weights["credit_amount"] * credit_norm
        + weights["saving_accounts"] * saving_risk
        + weights["checking_account"] * checking_risk
        + weights["age"] * age_risk
        + weights["housing"] * housing_risk
        + weights["job"] * job_risk
    )

    # Normalisation finale dans [0, 1]
    risk_score = _min_max_normalize(risk_score)

    logger.info(
        "Score de risque — min: %.3f, median: %.3f, max: %.3f",
        risk_score.min(),
        risk_score.median(),
        risk_score.max(),
    )

    return risk_score


def simulate_survival_target(
    df: pd.DataFrame,
    risk_score: pd.Series,
    weibull_shape: float = 1.5,
    base_scale: float = 48.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Simule les variables de survie (time, event) via un processus Weibull.

    Pour chaque emprunteur i :
        λ_i = base_scale × (1 - 0.7 × risk_score_i)
        T*_i ~ Weibull(k=weibull_shape, λ=λ_i)
        Si T*_i ≤ Duration_i : event=1, time=T*_i  (défaut observé)
        Si T*_i > Duration_i  : event=0, time=Duration_i (censuré)

    Args:
        df: DataFrame nettoyé contenant la colonne 'duration'.
        risk_score: Score de risque ∈ [0, 1] pour chaque ligne.
        weibull_shape: Paramètre de forme k de la Weibull (k > 1 → hazard croissant).
        base_scale: Échelle de base λ₀ en mois.
        random_seed: Graine pour reproductibilité.

    Returns:
        DataFrame enrichi avec les colonnes :
            - risk_score : Score de risque latent
            - time       : Durée observée (mois)
            - event      : 1 = défaut, 0 = censuré
    """
    rng = np.random.default_rng(random_seed)
    df = df.copy()

    # Calcul de l'échelle individuelle λ_i
    # Le facteur 0.7 contrôle l'amplitude : à risque max (score=1),
    # λ = base_scale × 0.3 → défaut rapide.
    # À risque min (score=0), λ = base_scale → survie longue.
    amplitude = 0.7
    individual_scale = base_scale * (1.0 - amplitude * risk_score.values)

    # Tirage Weibull : T* ~ Weibull(k, λ)
    # np.random uses the parameterization where scale = λ
    t_star = individual_scale * rng.weibull(weibull_shape, size=len(df))

    # Arrondi en mois entiers (réaliste pour des données bancaires)
    t_star = np.maximum(np.round(t_star), 1.0)

    # Censure administrative : T* vs Duration du prêt
    duration = df["duration"].values.astype(float)
    event = (t_star <= duration).astype(int)
    observed_time = np.where(event == 1, t_star, duration)

    # Ajout au DataFrame
    df["risk_score"] = risk_score.values
    df["time"] = observed_time.astype(int)
    df["event"] = event

    # --- Statistiques descriptives ---
    n_defaults = event.sum()
    n_censored = len(event) - n_defaults
    default_rate = n_defaults / len(event) * 100

    logger.info("=" * 50)
    logger.info("SIMULATION TIME-TO-DEFAULT (Weibull)")
    logger.info("=" * 50)
    logger.info("  Weibull shape (k)    : %.2f", weibull_shape)
    logger.info("  Base scale (λ₀)     : %.1f mois", base_scale)
    logger.info("  Défauts              : %d (%.1f%%)", n_defaults, default_rate)
    logger.info("  Censurés             : %d (%.1f%%)", n_censored, 100 - default_rate)
    logger.info("  Durée médiane (time) : %.0f mois", np.median(observed_time))
    logger.info("=" * 50)

    return df


# ------------------------------------------------------------------ #
#  Utilitaires internes                                               #
# ------------------------------------------------------------------ #


def _min_max_normalize(series: pd.Series | np.ndarray) -> pd.Series:
    """Normalisation min-max dans [0, 1], robuste aux constantes."""
    s = pd.Series(series)
    s_min, s_max = s.min(), s.max()
    if s_max - s_min == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)
