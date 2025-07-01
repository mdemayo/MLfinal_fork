import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import zipfile
import os
from datetime import datetime

# --- Cargar modelo entrenado desde archivo zip ---
if not os.path.exists("tfidf_logreg.joblib"):
    with zipfile.ZipFile("tfidf_logreg.zip", "r") as zip_ref:
        zip_ref.extractall()
model = joblib.load("tfidf_logreg.joblib")

# --- Configuración general ---
st.set_page_config(page_title="Análisis de Distress", layout="wide")
st.title("🧠 Analizador de Distress en Tuits")
st.markdown("Esta versión permite **subir un archivo .csv** con tuits para analizarlos emocionalmente.")

# --- Subida de archivo ---
archivo = st.file_uploader("📂 Sube tu archivo CSV con columnas `date` y `text`", type="csv")

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

    # --- Análisis del contenido ---
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

        # --- Resumen general ---
        st.subheader("📊 Resumen general")
        col1, col2 = st.columns(2)
        col1.metric("Total de tuits", len(df))
        col1.metric("Distress promedio", f"{df['distress_proba'].mean():.2%}")
        distress_total = df["distress_label"].sum()
        col2.metric("Tuits con distress", f"{distress_total} ({distress_total/len(df):.2%})")

        # --- Tabla filtrable ---
        st.subheader("📝 Tuits analizados")
        if st.checkbox("🔴 Mostrar solo tuits con distress"):
            st.dataframe(df[df["distress_label"] == 1][["date", "text", "distress_proba"]])
        else:
            st.dataframe(df[["date", "text", "distress_proba"]])

        st.success("Análisis completo 🎉")
else:
    st.info("Sube un archivo `.csv` con columnas 'date' y 'text' para comenzar.")

