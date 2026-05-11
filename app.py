"""
Aplicación Streamlit — Correlación y Predicción MALLA 200 vs 325
Ejecutar localmente:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Malla 200 vs 325",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Correlación y Predicción: AGIT -200M vs AGIT -325M")
st.markdown("Sube tu archivo CSV para analizar la correlación y entrenar modelos de predicción.")

# ─────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────
archivo = st.file_uploader("Sube el archivo CSV", type=["csv"])

if archivo is None:
    st.info("⬆️  Sube un archivo CSV para comenzar.")
    st.stop()

df = pd.read_csv(archivo)

COLUMNA_X = "AGIT -200M"
COLUMNA_Y = "AGIT -325M"

if COLUMNA_X not in df.columns or COLUMNA_Y not in df.columns:
    st.error(f"El archivo no contiene las columnas '{COLUMNA_X}' y '{COLUMNA_Y}'.")
    st.stop()

datos = df[[COLUMNA_X, COLUMNA_Y]].dropna().copy()
datos.columns = ["malla_200", "malla_325"]
datos = datos[datos["malla_200"] < 200].copy()   # elimina typo 877.0

st.success(f"✅ Archivo cargado — {len(datos)} filas válidas encontradas.")

# ─────────────────────────────────────────────────────────────
# ESTADÍSTICAS
# ─────────────────────────────────────────────────────────────
with st.expander("📋 Estadísticas descriptivas", expanded=False):
    st.dataframe(datos.describe().round(2))

# ─────────────────────────────────────────────────────────────
# ELIMINACIÓN DE OUTLIERS
# ─────────────────────────────────────────────────────────────
def mascara_iqr(serie, factor=1.5):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return (serie >= q1 - factor * iqr) & (serie <= q3 + factor * iqr)

mask_iqr  = mascara_iqr(datos["malla_200"]) & mascara_iqr(datos["malla_325"])
datos_iqr = datos[mask_iqr].copy()

X_sm      = sm.add_constant(datos_iqr["malla_200"])
modelo_sm = sm.OLS(datos_iqr["malla_325"], X_sm).fit()
cooks_d   = modelo_sm.get_influence().cooks_distance[0]
mask_cook = cooks_d < 4 / len(datos_iqr)
datos_clean = datos_iqr[mask_cook].reset_index(drop=True)

col1, col2, col3 = st.columns(3)
col1.metric("Datos originales",  len(datos))
col2.metric("Outliers eliminados", len(datos) - len(datos_clean))
col3.metric("Datos limpios", len(datos_clean))

# ─────────────────────────────────────────────────────────────
# SECCIÓN 1 — GRÁFICAS DE CORRELACIÓN
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("1. Correlación")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor("#FAFAFA")

for ax, xd, yd, titulo, color in [
    (axes[0], datos["malla_200"],       datos["malla_325"],       "Con outliers", "#378ADD"),
    (axes[1], datos_clean["malla_200"], datos_clean["malla_325"], "Sin outliers", "#1D9E75"),
]:
    ax.set_facecolor("#FAFAFA")
    sl, ic, rv, _, _ = stats.linregress(xd, yd)
    xl = np.linspace(xd.min(), xd.max(), 300)
    ax.scatter(xd, yd, color=color, alpha=0.4, edgecolors="none", s=14, zorder=3)
    ax.plot(xl, sl * xl + ic, color="#D85A30", lw=2, ls="--",
            zorder=4, label="Regresión lineal")
    texto = (f"n  = {len(xd)}\nR  = {rv:.3f}\nR² = {rv**2:.3f}\n"
             f"y  = {sl:.3f}x + {ic:.2f}")
    props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CCC", alpha=0.9)
    ax.text(0.97, 0.05, texto, transform=ax.transAxes, fontsize=9,
            va="bottom", ha="right", bbox=props, family="monospace")
    ax.set_xlabel("AGIT -200M (%)", fontsize=10)
    ax.set_ylabel("AGIT -325M (%)", fontsize=10)
    ax.set_title(titulo, fontsize=12, fontweight="bold")
    ax.grid(True, ls="--", lw=0.4, alpha=0.5, color="#CCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# Métricas de correlación destacadas
sl_c, ic_c, rv_c, pv_c, _ = stats.linregress(
    datos_clean["malla_200"], datos_clean["malla_325"]
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("R (Pearson)",  f"{rv_c:.4f}")
c2.metric("R²",           f"{rv_c**2:.4f}")
c3.metric("p-valor",      "< 0.001" if pv_c < 0.001 else f"{pv_c:.4f}")
c4.metric("Ecuación",     f"y = {sl_c:.3f}x + {ic_c:.2f}")

# ─────────────────────────────────────────────────────────────
# SECCIÓN 2 — MODELOS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("2. Modelos de predicción")

X = datos_clean[["malla_200"]].values
y = datos_clean["malla_325"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with st.spinner("Entrenando modelos..."):
    modelo_lin = LinearRegression().fit(X_train, y_train)

    modelo_poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("reg",  LinearRegression()),
    ]).fit(X_train, y_train)

    modelo_rf = RandomForestRegressor(
        n_estimators=200, max_depth=10,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    ).fit(X_train, y_train)

    modelo_nn = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(64, 32, 16), activation="relu",
            max_iter=1000, random_state=42,
            early_stopping=True, validation_fraction=0.1,
        )),
    ]).fit(X_train, y_train)

nombres = ["Regresión Lineal", "Regresión Polinomial g=2",
           "Random Forest",    "Red Neuronal (MLP)"]
modelos = [modelo_lin, modelo_poly, modelo_rf, modelo_nn]

resultados = []
for nombre, modelo in zip(nombres, modelos):
    yp   = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, yp))
    mae  = mean_absolute_error(y_test, yp)
    r2   = r2_score(y_test, yp)
    resultados.append({"Modelo": nombre, "RMSE": rmse, "MAE": mae, "R²": r2})

df_res = pd.DataFrame(resultados).set_index("Modelo")

# Ranking combinado
df_res["rank_R2"]   = df_res["R²"].rank(ascending=False)
df_res["rank_RMSE"] = df_res["RMSE"].rank(ascending=True)
df_res["rank_MAE"]  = df_res["MAE"].rank(ascending=True)
df_res["rank_prom"] = (df_res["rank_R2"] + df_res["rank_RMSE"] + df_res["rank_MAE"]) / 3
mejor_global = df_res["rank_prom"].idxmin()

# Tabla de métricas con resaltado
st.subheader("Métricas (datos de prueba — 20%)")

def resaltar_mejor(s):
    """Verde en la fila del mejor modelo."""
    estilos = []
    for idx in s.index:
        if idx == mejor_global:
            estilos.append("background-color: #d4edda; font-weight: bold")
        else:
            estilos.append("")
    return estilos

st.dataframe(
    df_res[["RMSE", "MAE", "R²"]].round(4).style.apply(resaltar_mejor, axis=1),
    use_container_width=True,
)
st.success(f"🏆 **Mejor modelo global: {mejor_global}**  "
           f"(R² = {df_res.loc[mejor_global, 'R²']:.4f}  |  "
           f"RMSE = {df_res.loc[mejor_global, 'RMSE']:.4f})")

# ─────────────────────────────────────────────────────────────
# Gráfica comparativa de curvas
# ─────────────────────────────────────────────────────────────
st.subheader("Curvas de los modelos sobre los datos")

colores  = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]
x_rango  = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor("#FAFAFA")
ax2.set_facecolor("#FAFAFA")
ax2.scatter(datos_clean["malla_200"], datos_clean["malla_325"],
            color="#888787", alpha=0.2, s=10, zorder=2, label="Datos limpios")

for nombre, modelo, color in zip(nombres, modelos, colores):
    r2v = df_res.loc[nombre, "R²"]
    lw  = 3.0 if nombre == mejor_global else 1.8
    ls  = "-"  if nombre == mejor_global else "--"
    etiq = (f"★ {nombre} (R²={r2v:.3f})"
            if nombre == mejor_global else f"{nombre} (R²={r2v:.3f})")
    ax2.plot(x_rango, modelo.predict(x_rango), color=color,
             lw=lw, ls=ls, label=etiq, zorder=3)

ax2.set_xlabel("AGIT -200M (%)", fontsize=11)
ax2.set_ylabel("AGIT -325M (%)", fontsize=11)
ax2.set_title("Comparación de modelos\n(línea sólida = mejor modelo)",
              fontsize=12, fontweight="bold")
ax2.legend(fontsize=9.5, framealpha=0.9)
ax2.grid(True, ls="--", lw=0.4, alpha=0.5, color="#CCC")
ax2.set_axisbelow(True)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ─────────────────────────────────────────────────────────────
# Gráfica de barras de métricas
# ─────────────────────────────────────────────────────────────
st.subheader("Comparación de métricas por modelo")

fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4.5))
fig3.patch.set_facecolor("#FAFAFA")
fig3.suptitle("RMSE / MAE / R² por modelo", fontsize=13, fontweight="bold")
nombres_cortos = ["Lineal", "Polinomial", "Random\nForest", "Red\nNeuronal"]

for ax, metrica, titulo in zip(
    axes3,
    ["RMSE", "MAE", "R²"],
    ["RMSE  (↓ mejor)", "MAE  (↓ mejor)", "R²  (↑ mejor)"],
):
    ax.set_facecolor("#FAFAFA")
    valores  = df_res[metrica].values
    idx_best = np.argmax(valores) if metrica == "R²" else np.argmin(valores)
    barras   = ax.bar(nombres_cortos, valores, color=colores,
                      width=0.55, edgecolor="white")
    barras[idx_best].set_edgecolor("black")
    barras[idx_best].set_linewidth(2.5)
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + max(valores) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(titulo, fontsize=10, pad=6)
    ax.set_ylim(0, max(valores) * 1.2)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5, color="#CCC")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.5)

plt.tight_layout()
st.pyplot(fig3)
plt.close()

# ─────────────────────────────────────────────────────────────
# SECCIÓN 3 — PREDICCIÓN INTERACTIVA
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("3. Predicción interactiva")
st.markdown(f"Ingresa un valor de **AGIT -200M** y obtén la predicción de **AGIT -325M** "
            f"con los 4 modelos.")

val_input = st.number_input(
    "AGIT -200M (%)",
    min_value=float(X.min()),
    max_value=float(X.max()),
    value=float(np.median(X)),
    step=0.1,
    format="%.1f",
)

xv = np.array([[val_input]])
pred_data = []
for nombre, modelo in zip(nombres, modelos):
    pred = modelo.predict(xv)[0]
    pred_data.append({
        "Modelo": nombre,
        "AGIT -325M predicho (%)": round(pred, 2),
        "R²": round(df_res.loc[nombre, "R²"], 4),
    })

df_pred = pd.DataFrame(pred_data).set_index("Modelo")

def resaltar_pred(s):
    return ["background-color: #d4edda; font-weight: bold"
            if idx == mejor_global else "" for idx in s.index]

st.dataframe(
    df_pred.style.apply(resaltar_pred, axis=1),
    use_container_width=True,
)

mejor_pred = modelos[nombres.index(mejor_global)].predict(xv)[0]
st.metric(
    label=f"Predicción con el mejor modelo ({mejor_global})",
    value=f"{mejor_pred:.2f} %",
    delta=f"AGIT -200M ingresado: {val_input:.1f}%",
)

# ─────────────────────────────────────────────────────────────
# PIE DE PÁGINA
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Análisis de correlación AGIT -200M vs AGIT -325M  |  "
           "Modelos: Lineal · Polinomial · Random Forest · MLP")