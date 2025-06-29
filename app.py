import streamlit as st
import joblib
import zipfile
import os

# Descompresión del modelo 
if not os.path.exists("tfidf_logreg.joblib"):
    with zipfile.ZipFile("tfidf_logreg.zip", "r") as zip_ref:
        zip_ref.extractall()

model = joblib.load("tfidf_logreg.joblib")

st.title("🧠 Mental Health Distress Detector")
st.subheader("Escribe un tuit (en inglés) y detectaremos si expresa angustia emocional.")

user_input = st.text_area("Ingresa el tuit aquí:", height=150)

if st.button("Detectar Distress"):
    if user_input.strip() == "":
        st.warning("Por favor, escribe un tuit.")
    else:
        prediction = model.predict([user_input])[0]
        proba = model.predict_proba([user_input])[0]

        if prediction == 1:
            st.error("🔴 Distress detectado (Angustia emocional).")
            st.write(f"Probabilidad de distress: **{proba[1]:.2%}**")
        else:
            st.success("🟢 No distress detectado.")
            st.write(f"Probabilidad de distress: **{proba[1]:.2%}**")
