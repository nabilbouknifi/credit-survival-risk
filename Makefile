# ============================================================
# Makefile — Credit Survival Risk
# ============================================================

.PHONY: install data train api app clean

# Installer les dépendances
install:
	pip install -r requirements.txt

# Exécuter le pipeline de données (ÉTAPE 1)
data:
	python -m src.data.prepare

# Entraîner les modèles (ÉTAPE 2)
train:
	python -m src.models.train

# Lancer l'API FastAPI (ÉTAPE 3)
api:
	uvicorn src.api.main:app --reload --port 8000

# Lancer l'interface Streamlit (ÉTAPE 4)
app:
	streamlit run src/app/streamlit_app.py

# Nettoyer les fichiers générés
clean:
	rm -rf data/processed/*.csv models/*.pkl
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
