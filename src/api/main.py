"""
api.py — API REST FastAPI pour servir le modèle de risque de crédit.

Exécution locale :
    $ uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Initialisation de l'API
app = FastAPI(
    title="Credit Survival Risk API",
    description="API de prédiction du risque de défaut (Analyse de survie)",
    version="1.0.0"
)

# 1. Définition du profil client attendu (Validation des données)
class ClientProfile(BaseModel):
    Age: int = Field(..., example=35)
    Sex: str = Field(..., example="male")
    Job: int = Field(..., example=2)
    Housing: str = Field(..., example="own")
    Saving_accounts: str = Field(..., example="moderate")
    Checking_account: str = Field(..., example="little")
    Credit_amount: int = Field(..., example=5000)
    Purpose: str = Field(..., example="car")

# Variable globale pour stocker le modèle en mémoire
model_pipeline = None

@app.on_event("startup")
def load_model():
    """Charge le modèle en mémoire au démarrage de l'API pour des réponses ultra-rapides."""
    global model_pipeline
    model_path = Path("models/survival_model.joblib")
    if model_path.exists():
        model_pipeline = joblib.load(model_path)
        logger.info("Modèle chargé avec succès.")
    else:
        logger.warning("Modèle introuvable. Entraînez le modèle avec train.py avant d'utiliser /predict.")

@app.get("/")
def health_check():
    """Vérifie que l'API est en ligne."""
    status = "ready" if model_pipeline is not None else "model_missing"
    return {"status": "ok", "model_status": status}

@app.post("/predict")
def predict_risk(client: ClientProfile):
    """Reçoit un profil client et renvoie un score de risque."""
    if model_pipeline is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore entraîné/chargé.")
    
    # Transformation du JSON reçu en DataFrame Pandas (avec les bons noms de colonnes)
    client_dict = client.dict()
    # On remet les espaces dans les noms de colonnes comme attendu par le modèle
    client_dict["Saving accounts"] = client_dict.pop("Saving_accounts")
    client_dict["Checking account"] = client_dict.pop("Checking_account")
    client_dict["Credit amount"] = client_dict.pop("Credit_amount")
    
    df_client = pd.DataFrame([client_dict])
    
    try:
        # Prédiction du score de risque (plus il est élevé, plus le défaut interviendra tôt)
        risk_score = model_pipeline.predict(df_client)[0]
        
        return {
            "status": "success",
            "risk_score": float(risk_score),
            "message": "Calcul effectué avec succès."
        }
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))