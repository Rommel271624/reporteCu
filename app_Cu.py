
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Configuración de página
    st.set_page_config(layout="wide")
    st.title("EDA de Planta - Reporte Tonelaje")

    # Nombre del archivo por defecto
    default_file_name = "sulfuros.csv"

    # Verificar existencia del archivo
    if os.path.exists(default_file_name):
        df = pd.read_csv(default_file_name, sep=';')
        df = df.dropna(axis=1, how='all')  # Eliminar columnas vacías
    else:
        st.error(f"El archivo '{default_file_name}' no se encuentra en la carpeta del proyecto.")
        return

    # Mostrar estadísticas generales
    st.subheader("Vista general del dataset")
    st.dataframe(df.head())
    st.write(f"**Número total de registros:** {df.shape[0]}")

    # Asegurar que exista la columna %Cu
    if "%Cu" not in df.columns:
        st.error("No se encontró la columna '%Cu' en el archivo.")
        return

    # Clasificación por ley de cobre
    df.set_index('%Cu', inplace=True)
    ley_alta = df[df.index > 1]
    ley_media = df[(df.index >= 0.8) & (df.index <= 1)]
    ley_baja = df[(df.index >= 0.1) & (df.index < 0.8)]

    # Mostrar tablas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Ley Alta (%Cu > 1)")
        st.write(f"Registros: {ley_alta.shape[0]}")
        st.dataframe(ley_alta.head())

    with col2:
        st.markdown("### Ley Media (0.8 ≤ %Cu ≤ 1)")
        st.write(f"Registros: {ley_media.shape[0]}")
        st.dataframe(ley_media.head())

    with col3:
        st.markdown("### Ley Baja (0.1 ≤ %Cu < 0.8)")
        st.write(f"Registros: {ley_baja.shape[0]}")
        st.dataframe(ley_baja.head())

    # Gráfico de distribución
    st.subheader("Distribución por Ley de Cobre")
    fig, ax = plt.subplots()
    categorias = ['Ley Alta', 'Ley Media', 'Ley Baja']
    valores = [ley_alta.shape[0], ley_media.shape[0], ley_baja.shape[0]]
    ax.bar(categorias, valores, color=['green', 'orange', 'red'])
    ax.set_ylabel("Número de Registros")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
