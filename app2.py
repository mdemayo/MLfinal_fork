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
st.set_page_config(page_title="Distress Analyzer", layout="wide")

# --- Estilo personalizado ---
st.markdown("""
    <style>
        .main {
            background-color: #fffbe6;
        }
        h1 {
            color: #fad22f;
        }
        .stButton > button {
            background-color: #fad22f;
            color: black;
            font-weight: bold;
            border: none;
            padding: 0.5em 1.2em;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #f7c900;
            color: black;
        }
        .metric-label {
            color: #fad22f;
        }
    </style>
""", unsafe_allow_html=True)

# --- Título personalizado ---
st.title("🌟 ¡Bienvenido/a! Análisis de Distress Emocional")
st.markdown("Por favor, suba el archivo `.csv` con los tuits del paciente para analizar su evolución emocional a lo largo del tiempo.")

# --- Subida de archivo ---
archivo = st.file_uploader("📂 Selecciona el archivo CSV (debe contener las columnas `date` y `text`)", type="csv")

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
    with st.spinner("🔍 Analizando tuits..."):
        df["distress_proba"] = df["text"].apply(lambda x: model.predict_proba([x])[0][1])
        df["distress_label"] = (df["distress_proba"] >= 0.5).astype(int)
        df["month"] = df["date"].dt.to_period("M").astype(str)

        # --- Visualización de evolución ---
        st.markdown("### 📈 Evolución emocional del paciente")
        avg_distress = df.groupby("month")["distress_proba"].mean()
        fig, ax = plt.subplots()
        avg_distress.plot(kind="line", marker="o", color="#fad22f", ax=ax)
        ax.set_title("Promedio mensual de distress")
        ax.set_ylabel("Probabilidad promedio")
        ax.set_xlabel("Mes")
        ax.grid(True)
        st.pyplot(fig)

        # --- Métricas principales ---
        st.markdown("### 📊 Resumen del análisis")
        col1, col2 = st.columns(2)
        col1.metric("Total de tuits", len(df))
        col1.metric("Distress promedio", f"{df['distress_proba'].mean():.2%}")
        distress_total = df["distress_label"].sum()
        col2.metric("Tuits con distress", f"{distress_total} ({distress_total/len(df):.2%})")

        # --- Tabla de tuits ---
        st.markdown("### 📝 Detalle de tuits analizados")
        if st.checkbox("🔴 Mostrar solo tuits con distress"):
            st.dataframe(df[df["distress_label"] == 1][["date", "text", "distress_proba"]])
        else:
            st.dataframe(df[["date", "text", "distress_proba"]])

        # --- Botón para imprimir/guardar ---
        st.markdown("""
        <br>
        <center>
            <button onclick="window.print()" style="background-color:#fad22f; font-weight:bold; padding:10px 20px; border:none; border-radius:5px;">
                🖨️ Imprimir resultados
            </button>
        </center>
        """, unsafe_allow_html=True)

        st.success("Análisis completado con éxito 🎉")
else:
    st.info("📄 Esperando archivo. Sube un `.csv` con columnas 'date' y 'text' para comenzar.")

