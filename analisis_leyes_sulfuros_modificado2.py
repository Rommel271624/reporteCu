import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.set_page_config(layout="wide")
    st.title("Análisis de Leyes de Sulfuros y Mineral Mixto")

    # Cargar datos
    try:
        df = pd.read_csv("sulfuros.csv", sep=';').dropna(axis=1, how='all')
    except FileNotFoundError:
        st.error("No se encontró el archivo 'sulfuros.csv'. Asegúrate de subirlo al mismo directorio de la app.")
        return

    # Verificar columnas necesarias
    required_cols = ['Tipo Mineral', 'TMH', 'TMS', '%Cu', 'Au g/TM', 'Ag g/TM']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"El archivo no contiene la columna '{col}'. Verifica el formato del CSV.")
            return

    def filtrar_y_resumir(tipo, nombre):
        subset = df[df['Tipo Mineral'] == tipo]
        ley_alta = subset[subset['%Cu'] > 1]
        ley_media = subset[(subset['%Cu'] >= 0.8) & (subset['%Cu'] <= 1)]
        ley_baja = subset[(subset['%Cu'] >= 0.1) & (subset['%Cu'] < 0.8)]

        def resumen_categoria(cat_nombre, subcat):
            total_tmh = subcat['TMH'].sum()
            total_tms = subcat['TMS'].sum()
            cu_prom = (subcat['%Cu'] * subcat['TMS']).sum() / total_tms if total_tms > 0 else 0
            au_prom = (subcat['Au g/TM'] * subcat['TMS']).sum() / total_tms if total_tms > 0 else 0
            ag_prom = (subcat['Ag g/TM'] * subcat['TMS']).sum() / total_tms if total_tms > 0 else 0
            return [f"{nombre} {cat_nombre}", total_tmh, total_tms, cu_prom, au_prom, ag_prom]

        data = [
            resumen_categoria('Ley Alta', ley_alta),
            resumen_categoria('Ley Media', ley_media),
            resumen_categoria('Ley Baja', ley_baja)
        ]
        return data

    resumen_data = []
    resumen_data.extend(filtrar_y_resumir('Sulfuros', 'Sulfuros'))
    resumen_data.extend(filtrar_y_resumir('Mixto', 'Mixto'))

    # Totales generales
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

    # Gráfica de pastel más pequeña
    fig, ax = plt.subplots(figsize=(4, 4))  # tamaño reducido
    ax.pie(
        resumen_df.iloc[:-1]['Total TMS'],
        labels=resumen_df.iloc[:-1]['Categoría'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
