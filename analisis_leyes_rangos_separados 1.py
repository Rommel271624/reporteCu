import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ==== Estilo de tabla ====
def estilo_tabla(df):
    return (
        df.style
        .format(precision=4)
        .set_table_styles([
            {'selector': 'th',
             'props': [
                 ('text-align', 'center'),
                 ('background-color', '#000000'),  # fondo negro
                 ('color', 'white'),               # letras blancas
                 ('font-weight', 'bold'),
                 ('white-space', 'nowrap')
             ]},
            {'selector': 'td',
             'props': [
                 ('text-align', 'center')
             ]}
        ])
        .set_properties(**{'text-align': 'center'})
    )

def cargar_datos(nombre_archivo):
    try:
        df = pd.read_csv(nombre_archivo, sep=';').dropna(axis=1, how='all')
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{nombre_archivo}'.")
        return None
    return df

def verificar_columnas(df):
    required_cols = ['TMH', 'TMS', '%Cu', 'Au g/TM', 'Ag g/TM']
    return all(col in df.columns for col in required_cols)

def calcular_resumen(df, ley_alta, ley_media, ley_baja, nombre):
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

    st.markdown(f"<h2 style='text-align:center;'>{nombre}</h2>", unsafe_allow_html=True)
    st.dataframe(estilo_tabla(resumen_df), use_container_width=True)

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

    # ==== Título general centrado ====
    st.markdown("<h1 style='text-align:center; color:#2ca02c;'>TM de Mineral Sulfuros y Mixto</h1>", unsafe_allow_html=True)

    # ==== Sulfuros ====
    df_s = cargar_datos("sulfuros.csv")
    if df_s is None or not verificar_columnas(df_s):
        return
    ley_alta_s = df_s[df_s['%Cu'] > 1.0]
    ley_media_s = df_s[(df_s['%Cu'] >= 0.8) & (df_s['%Cu'] <= 1.0)]
    ley_baja_s = df_s[(df_s['%Cu'] >= 0.1) & (df_s['%Cu'] < 0.8)]
    total_tmh_s, total_tms_s, cu_s, au_s, ag_s = calcular_resumen(df_s, ley_alta_s, ley_media_s, ley_baja_s, "Sulfuros")

    # ==== Mixto ====
    df_m = cargar_datos("mixto.csv")
    if df_m is None or not verificar_columnas(df_m):
        return
    ley_alta_m = df_m[df_m['%Cu'] > 3]
    ley_media_m = df_m[(df_m['%Cu'] >= 2) & (df_m['%Cu'] <= 3)]
    ley_baja_m = df_m[(df_m['%Cu'] >= 0.1) & (df_m['%Cu'] < 2)]
    total_tmh_m, total_tms_m, cu_m, au_m, ag_m = calcular_resumen(df_m, ley_alta_m, ley_media_m, ley_baja_m, "Mixto")

    # ==== Resumen General ====
    resumen_general = pd.DataFrame([
        ["Sulfuro", total_tmh_s, total_tms_s, cu_s, au_s, ag_s],
        ["Mixto", total_tmh_m, total_tms_m, cu_m, au_m, ag_m],
        ["Total General", total_tmh_s + total_tmh_m, total_tms_s + total_tms_m,
         (cu_s*total_tms_s + cu_m*total_tms_m)/(total_tms_s + total_tms_m),
         (au_s*total_tms_s + au_m*total_tms_m)/(total_tms_s + total_tms_m),
         (ag_s*total_tms_s + ag_m*total_tms_m)/(total_tms_s + total_tms_m)]
    ], columns=["Tipo de Material", "Total TMH", "Total TMS", "Promedio Ponderado %Cu",
                "Promedio Ponderado Au g/TM", "Promedio Ponderado Ag g/TM"])

    st.markdown("<h2 style='text-align:center;'>Resumen General</h2>", unsafe_allow_html=True)
    st.dataframe(estilo_tabla(resumen_general), use_container_width=True)

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
