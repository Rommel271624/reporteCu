import streamlit as st
import pandas as pd
import os

# Configuración de la página
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

# ==== Sección Cobre (Sulfuros) ====
st.header("Tabla - Sulfuros")
if not df_cu.empty:
    st.dataframe(df_cu)

    st.subheader("Gráfico de Pastel - Sulfuros")
    fig_cu, ax_cu = plt.subplots()
    df_cu.set_index(df_cu.columns[0]).plot.pie(
        y=df_cu.columns[1],  # Segunda columna como valores
        autopct='%1.1f%%',
        ax=ax_cu,
        legend=False
    )
    ax_cu.set_ylabel('')
    st.pyplot(fig_cu)
else:
    st.warning("No hay datos para Sulfuros.")

# ==== Sección Mineral Mixto ====
st.header("Tabla - Mineral Mixto")
if not df_mixto.empty:
    st.dataframe(df_mixto)

    st.subheader("Gráfico de Pastel - Mineral Mixto")
    fig_mixto, ax_mixto = plt.subplots()
    df_mixto.set_index(df_mixto.columns[0]).plot.pie(
        y=df_mixto.columns[1],  # Segunda columna como valores
        autopct='%1.1f%%',
        ax=ax_mixto,
        legend=False
    )
    ax_mixto.set_ylabel('')
    st.pyplot(fig_mixto)
else:
    st.warning("No hay datos para Mineral Mixto.")
