import streamlit as st
import pandas as pd
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
import joblib
import zipfile
import os
from datetime import datetime
from snscrape.base import ScraperException

# --- Cargar modelo entrenado ---
if not os.path.exists("tfidf_logreg.joblib"):
    with zipfile.ZipFile("tfidf_logreg.zip", "r") as zip_ref:
        zip_ref.extractall()
model = joblib.load("tfidf_logreg.joblib")

# --- Página principal ---
st.set_page_config(page_title="Analizador de Distress", layout="wide")
st.title("🧠 Analizador de Distress en Tuits")
st.markdown("Esta herramienta permite analizar la evolución emocional de un usuario de X (Twitter) o un archivo .csv de tuits.")

# --- Selección de modo ---
modo = st.radio("Selecciona cómo quieres analizar:", ["Ingresar usuario de X (Twitter)", "Subir archivo .csv"])

df = pd.DataFrame()

if modo == "Ingresar usuario de X (Twitter)":
    username = st.text_input("Ingresa el nombre de usuario (sin @):")
    num_tweets = st.slider("Número de tuits a analizar", 10, 300, 100, step=10)

    if st.button("🔍 Analizar Tuits") and username:
        with st.spinner("Extrayendo y analizando tuits..."):
            tweets = []
            try:
                for tweet in sntwitter.TwitterUserScraper(username).get_items():
                    if len(tweets) >= num_tweets:
                        break
                    tweets.append({"date": tweet.date, "text": tweet.content})
            except ScraperException:
                st.error("🚫 No se pudo acceder a ese usuario. Intenta con otro usuario o vuelve más tarde.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error inesperado: {e}")
                st.stop()

            if not tweets:
                st.warning("No se encontraron tuits públicos.")
                st.stop()

            df = pd.DataFrame(tweets)

elif modo == "Subir archivo .csv":
    archivo = st.file_uploader("📂 Sube tu archivo CSV con columnas 'date' y 'text'", type="csv")
    if archivo:
        try:
            df = pd.read_csv(archivo)
            if "date" not in df.columns or "text" not in df.columns:
                st.error("❌ El archivo debe tener las columnas 'date' y 'text'")
                st.stop()
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            st.error(f"❌ No se pudo leer el archivo: {e}")
            st.stop()

# --- Si hay datos, procesar ---
if not df.empty:
    with st.spinner("Analizando tuits..."):
        df["distress_proba"] = df["text"].apply(lambda x: model.predict_proba([x])[0][1])
        df["distress_label"] = (df["distress_proba"] >= 0.5).astype(int)
        df["month"] = df["date"].dt.to_period("M").astype(str)

        # --- Gráfico de evolución ---
        st.subheader("📈 Evolución del distress")
        avg_distress = df.groupby("month")["distress_proba"].mean()
        fig, ax = plt.subplots()
        avg_distress.plot(kind="line", marker="o", ax=ax)
        ax.set_title("Promedio de distress por mes")
        ax.set_ylabel("Probabilidad promedio de distress")
        ax.set_xlabel("Mes")
        ax.grid(True)
        st.pyplot(fig)

        # --- Métricas generales ---
        st.subheader("📊 Resumen general")
        col1, col2 = st.columns(2)
        col1.metric("Total de tuits", len(df))
        col1.metric("Distress promedio", f"{df['distress_proba'].mean():.2%}")
        distress_total = df["distress_label"].sum()
        col2.metric("Tuits con distress", f"{distress_total} ({distress_total/len(df):.2%})")

        # --- Tabla con filtro ---
        st.subheader("📝 Tuits analizados")
        if st.checkbox("🔴 Mostrar solo tuits con distress"):
            st.dataframe(df[df["distress_label"] == 1][["date", "text", "distress_proba"]])
        else:
            st.dataframe(df[["date", "text", "distress_proba"]])

        st.success("Análisis completo 🎉")

#cd prueba_2
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#venv\Scripts\activate
#streamlit run app2.py

