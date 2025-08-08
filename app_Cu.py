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

# Cargar datos
@st.cache_data
def load_data():
    csv_path = "sulfuros.csv"  # Ajusta el nombre de tu CSV si es distinto
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error(f"No se encontró el archivo {csv_path}")
        return pd.DataFrame()

df = load_data()

# Mostrar título
st.title("Reporte de Tonelaje - Cobre")

# Mostrar DataFrame
if not df.empty:
    st.dataframe(df)

    # Ejemplo de gráfico
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    st.pyplot(fig)
else:
    st.warning("No hay datos para mostrar.")
