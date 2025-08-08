import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.set_page_config(layout="wide")
    st.title("Análisis de Leyes de Sulfuros")

    # Cargar datos
    try:
        df = pd.read_csv("sulfuros.csv", sep=';').dropna(axis=1, how='all')
    except FileNotFoundError:
        st.error("No se encontró el archivo 'sulfuros.csv'. Asegúrate de subirlo al mismo directorio de la app.")
        return

    # Configurar índice
    if '%Cu' not in df.columns:
        st.error("El archivo no contiene la columna '%Cu'. Verifica el formato del CSV.")
        return
    df.set_index('%Cu', inplace=True)

    # Filtrar por categorías
    ley_alta = df[df.index > 1]
    ley_media = df[(df.index >= 0.8) & (df.index <= 1)]
    ley_baja = df[(df.index >= 0.1) & (df.index < 0.8)]

    # Crear tabla resumen
    resumen = pd.DataFrame({
        'Categoría': ['Ley Alta (>1)', 'Ley Media (0.8-1)', 'Ley Baja (0.1-0.8)'],
        'Cantidad': [len(ley_alta), len(ley_media), len(ley_baja)]
    })

    # Mostrar tabla
    st.subheader("Resumen de Leyes de Sulfuros")
    st.dataframe(resumen)

    # Gráfica de pastel
    fig, ax = plt.subplots()
    ax.pie(
        resumen['Cantidad'],
        labels=resumen['Categoría'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
