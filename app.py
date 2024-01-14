import streamlit as st 
import numpy as np
import pandas as pd 
import joblib 

# Chargement du modèle 
model = joblib.load("newData_kmeans.pkl")

# Fonction pour faire la détection
def make_detection(model, features):

    # Vérification de l'entrée des caractéristiques dans un tableau bidimensionnel
    features_array = features.values.reshape(1, -1)

    # Prédiction des indices de clusterss
    cluster_indices = model.predict(features_array)

    # Vérifier si le cluster numéro 3 est présent dans les prédictions
    if 3 in cluster_indices:
        st.subheader("Ce client est un potentiel fraudeur")
    else:
        st.subheader("Ce client n'est pas un potentiel fraudeur sauf erreur !")

    
# Interface utilisateurs
st.title("Application de détection de fraude par carte bancaire")
st.write("Cette application utilise un modèle de machine learning pour détecter les potentiel fraudeur par carte bancaire")

st.sidebar.header("Informations du client ")

# Saisie des caractéristiques du client 
gender = st.sidebar.selectbox("Sexe du client", ["Female", "Male"])
age = st.sidebar.number_input("Age du client", min_value=18,  value=18)
annual_income = st.sidebar.number_input("Revenu annuel (k $)", min_value=0, value=1)
spending_score = st.sidebar.number_input("Score de dépenses (100)", min_value=1, value=1, max_value=100 )

# Creation de dataframe à partir de caractéristique renseigné

input_data = pd.DataFrame({
    'gender': [gender],
    'age':[age],
    'annual_income':[annual_income],
    'spending_score':[spending_score]
})

# Fonction pour encoder les données
def encode_data(input_data):
    input_data_copy = input_data.copy()
    input_data_copy['gender'].replace({'Female':0, 'Male':1}, inplace=True)
    return input_data_copy

# Encoder les données
input_encoded = encode_data(input_data)    

# Détection 
if st.sidebar.button("Détecter la fraude"):
    result = make_detection(model, input_encoded)
    st.subheader(result)

    # Afficher le résultat uniquement s'il y a un cluster prédit
    if result:
        st.subheader(result)