import sys
import subprocess

# Instalación automática de matplotlib si no está presente
try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import os

# Configuración de la página para que ocupe todo el ancho
st.set_page_config(layout="wide")

# Función para cargar datos
@st.cache_data
def load_data(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error(f"No se encontró el archivo {csv_path}")
        return pd.DataFrame()

# Cargar datasets
df_cu = load_data("sulfuros.csv")
df_mixto = load_data("mixto.csv")

# ==== Sección Cobre ====
st.header("Reporte de Tonelaje - Cobre")

if not df_cu.empty:
    st.dataframe(df_cu)
    fig_cu, ax_cu = plt.subplots()
    df_cu.plot(ax=ax_cu)
    st.pyplot(fig_cu)
else:
    st.warning("No hay datos para Cobre.")

# ==== Sección Mineral Mixto ====
st.header("Reporte de Tonelaje - Mineral Mixto")

if not df_mixto.empty:
    st.dataframe(df_mixto)
    fig
