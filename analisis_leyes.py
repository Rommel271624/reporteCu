import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main_sulfuros():
    st.set_page_config(layout="wide")
    st.title("Análisis de Leyes de Sulfuros")

    # Cargar datos
    try:
        df = pd.read_csv("sulfuros.csv", sep=';').dropna(axis=1, how='all')
    except FileNotFoundError:
        st.error("No se encontró el archivo 'sulfuros.csv'. Asegúrate de subirlo al mismo directorio de la app.")
        return

    # Verificar columnas necesarias
    required_cols = ['TMH', 'TMS', '%Cu', 'Au g/TM', 'Ag g/TM']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"El archivo no contiene la columna '{col}'. Verifica el formato del CSV.")
            return

    # Filtrar por categorías
    ley_alta = df[df['%Cu'] > 1]
    ley_media = df[(df['%Cu'] >= 0.8) & (df['%Cu'] <= 1)]
    ley_baja = df[(df['%Cu'] >= 0.1) & (df['%Cu'] < 0.8)]

    def resumen_categoria(nombre, subset):
        total_tmh = subset['TMH'].sum()
        total_tms = subset['TMS'].sum()
        cu_prom = (subset['%Cu'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        au_prom = (subset['Au g/TM'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        ag_prom = (subset['Ag g/TM'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        return [nombre, total_tmh, total_tms, cu_prom, au_prom, ag_prom]

    resumen_data = [
        resumen_categoria('Ley Alta', ley_alta),
        resumen_categoria('Ley Media', ley_media),
        resumen_categoria('Ley Baja', ley_baja)
    ]

    # Totales
    total_tmh = df['TMH'].sum()
    total_tms = df['TMS'].sum()
    cu_prom_total = (df['%Cu'] * df['TMS']).sum() / total_tms
    au_prom_total = (df['Au g/TM'] * df['TMS']).sum() / total_tms
    ag_prom_total = (df['Ag g/TM'] * df['TMS']).sum() / total_tms
    resumen_data.append(['Total', total_tmh, total_tms, cu_prom_total, au_prom_total, ag_prom_total])

    resumen_df = pd.DataFrame(resumen_data, columns=[
        'Categoría', 'Total TMH', 'Total TMS', 'Promedio Ponderado %Cu',
        'Promedio Ponderado Au g/TM', 'Promedio Ponderado Ag g/TM'
    ])

    # Mostrar tabla resumen
    st.subheader("Tabla Resumen:")
    st.dataframe(resumen_df.style.format({
        'Total TMH': "{:.4f}",
        'Total TMS': "{:.4f}",
        'Promedio Ponderado %Cu': "{:.4f}",
        'Promedio Ponderado Au g/TM': "{:.4f}",
        'Promedio Ponderado Ag g/TM': "{:.4f}"
    }))

    # Gráfica de pastel
    fig, ax = plt.subplots(figsize=(2,1))
    ax.pie(
        resumen_df.iloc[:-1]['Total TMS'],
        labels=resumen_df.iloc[:-1]['Categoría'],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize':2}
    )
    ax.axis('equal')
    st.pyplot(fig)


def main_mixto():
    st.set_page_config(layout="wide")
    st.title("Análisis de Leyes de Mixto")

    # Cargar datos
    try:
        df = pd.read_csv("mixto.csv", sep=';').dropna(axis=1, how='all')
    except FileNotFoundError:
        st.error("No se encontró el archivo 'mixto.csv'. Asegúrate de subirlo al mismo directorio de la app.")
        return

    # Verificar columnas necesarias
    required_cols = ['TMH', 'TMS', '%Cu', 'Au g/TM', 'Ag g/TM']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"El archivo no contiene la columna '{col}'. Verifica el formato del CSV.")
            return

    # Filtrar por categorías
    ley_alta = df[df['%Cu'] > 3]
    ley_media = df[(df['%Cu'] >= 2) & (df['%Cu'] <= 3)]
    ley_baja = df[(df['%Cu'] >= 0.1) & (df['%Cu'] < 2)]

    def resumen_categoria(nombre, subset):
        total_tmh = subset['TMH'].sum()
        total_tms = subset['TMS'].sum()
        cu_prom = (subset['%Cu'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        au_prom = (subset['Au g/TM'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        ag_prom = (subset['Ag g/TM'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        return [nombre, total_tmh, total_tms, cu_prom, au_prom, ag_prom]

    resumen_data = [
        resumen_categoria('Ley Alta', ley_alta),
        resumen_categoria('Ley Media', ley_media),
        resumen_categoria('Ley Baja', ley_baja)
    ]

    # Totales
    total_tmh = df['TMH'].sum()
    total_tms = df['TMS'].sum()
    cu_prom_total = (df['%Cu'] * df['TMS']).sum() / total_tms
    au_prom_total = (df['Au g/TM'] * df['TMS']).sum() / total_tms
    ag_prom_total = (df['Ag g/TM'] * df['TMS']).sum() / total_tms
    resumen_data.append(['Total', total_tmh, total_tms, cu_prom_total, au_prom_total, ag_prom_total])

    resumen_df = pd.DataFrame(resumen_data, columns=[
        'Categoría', 'Total TMH', 'Total TMS', 'Promedio Ponderado %Cu',
        'Promedio Ponderado Au g/TM', 'Promedio Ponderado Ag g/TM'
    ])

    # Mostrar tabla resumen
    st.subheader("Tabla Resumen:")
    st.dataframe(resumen_df.style.format({
        'Total TMH': "{:.4f}",
        'Total TMS': "{:.4f}",
        'Promedio Ponderado %Cu': "{:.4f}",
        'Promedio Ponderado Au g/TM': "{:.4f}",
        'Promedio Ponderado Ag g/TM': "{:.4f}"
    }))

    # Gráfica de pastel
    fig, ax = plt.subplots(figsize=(2,1))
    ax.pie(
        resumen_df.iloc[:-1]['Total TMS'],
        labels=resumen_df.iloc[:-1]['Categoría'],
        autopct='%1.1f%%',
        startangle=90,
          textprops={'fontsize':2}
    )
    ax.axis('equal')
    st.pyplot(fig)


if __name__ == "__main__":
    main_sulfuros()
    main_mixto()
