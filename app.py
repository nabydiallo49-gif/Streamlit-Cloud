import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

st.title("Analyse & prédiction des inscriptions étudiantes")

# Chargement des données
df = pd.read_csv("inscriptions_etudiants.csv")

st.subheader("Aperçu des données")
st.dataframe(df)

st.subheader("Statistiques générales")
st.write(df.describe())

st.subheader("Répartition par filière")
st.bar_chart(df['filiere'].value_counts())

st.subheader("Préparation des données pour la prédiction")

data = df[['age', 'sexe', 'filiere', 'niveau', 'frais_scolarite', 'statut_paiement']]

encoder = LabelEncoder()
for col in ['sexe', 'filiere', 'niveau']:
    data[col] = encoder.fit_transform(data[col])

data['statut_paiement'] = data['statut_paiement'].map({'Payé': 1, 'En attente': 0})

X = data.drop('statut_paiement', axis=1)
y = data['statut_paiement']

model = LogisticRegression()
model.fit(X, y)

st.subheader("Test de prédiction")

age = st.slider("Âge", 18, 30, 22)
frais = st.number_input("Frais de scolarité", 300000, 700000, 450000)

input_data = pd.DataFrame([[age, 0, 0, 0, frais]],
                          columns=X.columns)

prediction = model.predict(input_data)

if prediction[0] == 1:
    st.success("✅ Paiement probable")
else:
    st.warning("⚠️ Risque de retard de paiement")
