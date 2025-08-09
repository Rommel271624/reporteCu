import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def cargar_datos(nombre_archivo):
    try:
        df = pd.read_csv(nombre_archivo, sep=';').dropna(axis=1, how='all')
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{nombre_archivo}'. Asegúrate de subirlo al mismo directorio de la app.")
        return None
    return df

def verificar_columnas(df):
    required_cols = ['TMH', 'TMS', '%Cu', 'Au g/TM', 'Ag g/TM']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"El archivo no contiene la columna '{col}'. Verifica el formato del CSV.")
            return False
    return True

def calcular_resumen(df, nombre):
    ley_alta = df[df['%Cu'] > 1]
    ley_media = df[(df['%Cu'] >= 0.8) & (df['%Cu'] <= 1)]
    ley_baja = df[(df['%Cu'] >= 0.1) & (df['%Cu'] < 0.8)]

    def resumen_categoria(nombre_cat, subset):
        total_tmh = subset['TMH'].sum()
        total_tms = subset['TMS'].sum()
        cu_prom = (subset['%Cu'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        au_prom = (subset['Au g/TM'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        ag_prom = (subset['Ag g/TM'] * subset['TMS']).sum() / total_tms if total_tms > 0 else 0
        return [nombre_cat, total_tmh, total_tms, cu_prom, au_prom, ag_prom]

    resumen_data = [
        resumen_categoria('Ley Alta', ley_alta),
        resumen_categoria('Ley Media', ley_media),
        resumen_categoria('Ley Baja', ley_baja)
    ]

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

    st.subheader(f"Tabla Resumen: {nombre}")
    st.dataframe(resumen_df.style.format({
        'Total TMH': "{:.4f}",
        'Total TMS': "{:.4f}",
        'Promedio Ponderado %Cu': "{:.4f}",
        'Promedio Ponderado Au g/TM': "{:.4f}",
        'Promedio Ponderado Ag g/TM': "{:.4f}"
    }))

    fig, ax = plt.subplots()
    ax.pie(
        resumen_df.iloc[:-1]['Total TMS'],
        labels=resumen_df.iloc[:-1]['Categoría'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')
    st.pyplot(fig)

    return total_tmh, total_tms, cu_prom_total, au_prom_total, ag_prom_total

def main():
    st.set_page_config(layout="wide")
    st.title("Análisis de Leyes de Sulfuros y Mixto")

    # Sulfuros
    df_s = cargar_datos("sulfuros.csv")
    if df_s is None or not verificar_columnas(df_s):
        return
    total_tmh_s, total_tms_s, cu_s, au_s, ag_s = calcular_resumen(df_s, "Sulfuros")

    # Mixto
    df_m = cargar_datos("mixto.csv")
    if df_m is None or not verificar_columnas(df_m):
        return
    total_tmh_m, total_tms_m, cu_m, au_m, ag_m = calcular_resumen(df_m, "Mixto")

    # Resumen general
    resumen_general = pd.DataFrame([
        ["Sulfuro", total_tmh_s, total_tms_s, cu_s, au_s, ag_s],
        ["Mixto", total_tmh_m, total_tms_m, cu_m, au_m, ag_m],
        ["Total General", total_tmh_s + total_tmh_m, total_tms_s + total_tms_m,
         (cu_s*total_tms_s + cu_m*total_tms_m)/(total_tms_s + total_tms_m),
         (au_s*total_tms_s + au_m*total_tms_m)/(total_tms_s + total_tms_m),
         (ag_s*total_tms_s + ag_m*total_tms_m)/(total_tms_s + total_tms_m)]
    ], columns=["Tipo de Material", "Total TMH", "Total TMS", "Promedio Ponderado %Cu",
                "Promedio Ponderado Au g/TM", "Promedio Ponderado Ag g/TM"])

    st.subheader("Tabla Resumen General (Sulfuros y Mixto):")
    st.dataframe(resumen_general.style.format({
        "Total TMH": "{:.4f}",
        "Total TMS": "{:.4f}",
        "Promedio Ponderado %Cu": "{:.4f}",
        "Promedio Ponderado Au g/TM": "{:.4f}",
        "Promedio Ponderado Ag g/TM": "{:.4f}"
    }))

    # Gráfico de pastel para resumen general
    fig, ax = plt.subplots()
    ax.pie(
        resumen_general.iloc[:-1]['Total TMS'],
        labels=resumen_general.iloc[:-1]['Tipo de Material'],
        autopct='%1.1f%%',
        startangle=90
    )
    ax.axis('equal')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
