def estilo_tabla(df, color_header="#4CAF50"):
    return (
        df.style
        .format(precision=4)
        .set_table_styles([
            {'selector': 'th',
             'props': [
                 ('text-align', 'center'),
                 ('background-color', color_header),
                 ('color', 'white'),
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

def main():
    st.set_page_config(layout="wide")
    # ==== Título general centrado ====
    st.markdown("<h1 style='text-align:center; color:#2ca02c;'>Análisis de Leyes de Sulfuros y Mixto (rangos separados)</h1>", unsafe_allow_html=True)

    # ==== Sulfuros ====
    df_s = cargar_datos("sulfuros.csv")
    if df_s is None or not verificar_columnas(df_s):
        return
    ley_alta_s = df_s[df_s['%Cu'] > 1.0]
    ley_media_s = df_s[(df_s['%Cu'] >= 0.8) & (df_s['%Cu'] <= 1.0)]
    ley_baja_s = df_s[(df_s['%Cu'] >= 0.1) & (df_s['%Cu'] < 0.8)]

    st.markdown("<h2 style='text-align:center; color:#1f77b4;'>Tabla Resumen: Sulfuros</h2>", unsafe_allow_html=True)
    total_tmh_s, total_tms_s, cu_s, au_s, ag_s = calcular_resumen(df_s, ley_alta_s, ley_media_s, ley_baja_s, "Sulfuros", "#1f77b4")

    # ==== Mixto ====
    df_m = cargar_datos("mixto.csv")
    if df_m is None or not verificar_columnas(df_m):
        return
    ley_alta_m = df_m[df_m['%Cu'] > 3]
    ley_media_m = df_m[(df_m['%Cu'] >= 2) & (df_m['%Cu'] <= 3)]
    ley_baja_m = df_m[(df_m['%Cu'] >= 0.1) & (df_m['%Cu'] < 2)]

    st.markdown("<h2 style='text-align:center; color:#ff7f0e;'>Tabla Resumen: Mixto</h2>", unsafe_allow_html=True)
    total_tmh_m, total_tms_m, cu_m, au_m, ag_m = calcular_resumen(df_m, ley_alta_m, ley_media_m, ley_baja_m, "Mixto", "#ff7f0e")

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

    st.markdown("<h2 style='text-align:center; color:#2ca02c;'>Tabla Resumen General</h2>", unsafe_allow_html=True)
    st.dataframe(estilo_tabla(resumen_general, "#2ca02c"), use_container_width=True)
