import streamlit as st
import requests

# Configuration de la page
st.set_page_config(page_title="Credit Survival Risk", page_icon="🏦", layout="wide")

st.title("🏦 Simulateur de Risque de Crédit")
st.markdown("""
Interface métier connectée à notre moteur d'analyse de survie (FastAPI). 
Ajustez les paramètres du profil client pour simuler son score de risque de défaut.
""")

st.divider()

# --- FORMULAIRE METIER ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Profil Client")
    age = st.slider("Âge", min_value=18, max_value=100, value=35)
    sex = st.selectbox("Sexe", ["male", "female"])
    job = st.selectbox("Qualification (Job)", [0, 1, 2, 3], help="0: Non qualifié/Non résident, 3: Cadre/Hautement qualifié")
    housing = st.selectbox("Logement", ["own", "rent", "free"])

with col2:
    st.subheader("💰 Situation Bancaire")
    checking = st.selectbox("Solde Compte Courant", ["none", "little", "moderate", "rich"])
    saving = st.selectbox("Solde Compte Épargne", ["none", "little", "moderate", "quite rich", "rich"])

with col3:
    st.subheader("📝 Détails du Crédit")
    credit_amount = st.number_input("Montant (DM)", min_value=100, max_value=20000, value=5000, step=500)
    purpose = st.selectbox("Motif", ["car", "furniture/equipment", "radio/TV", "education", "business", "repairs", "vacation/others"])

st.divider()

# --- APPEL A L'API FastAPI ---
if st.button("Lancer la simulation de survie", type="primary", use_container_width=True):
    # On prépare le JSON exactement comme l'API (ClientProfile) l'attend
    payload = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "Saving_accounts": saving,
        "Checking_account": checking,
        "Credit_amount": credit_amount,
        "Purpose": purpose
    }
    
    with st.spinner("Analyse du profil par le modèle Machine Learning..."):
        try:
            # Envoi de la requête POST à notre API locale
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                score = data["risk_score"]
                
                st.success("Analyse terminée avec succès !")
                
                # Affichage des résultats
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric(label="Score de Risque (Hazard Score)", value=f"{score:.3f}")
                with res_col2:
                    if score > 50: # Seuil fictif à ajuster selon tes observations
                        st.error("⚠️ Risque élevé : Le défaut de paiement est susceptible d'arriver rapidement.")
                    else:
                        st.success("✅ Risque maîtrisé : Profil de survie favorable.")
            else:
                st.error(f"Erreur renvoyée par le modèle (Code {response.status_code}): {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Impossible de se connecter à l'API. Vérifie que Uvicorn tourne bien dans un autre terminal avec `uvicorn src.api.main:app --reload`.")