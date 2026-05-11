"""
=============================================================
  ANÁLISIS DE CORRELACIÓN Y PREDICCIÓN: MALLA 200 vs 325
=============================================================
Pasos:
  1. Carga y limpieza de datos
  2. Eliminación de outliers (IQR + Distancia de Cook)
  3. Gráficas de correlación (antes y después)
  4. Modelos de predicción:
       a) Regresión Lineal Simple
       b) Regresión Polinomial (grado 2)
       c) Random Forest
       d) Red Neuronal (MLP)
  5. Comparación de modelos
  6. Predicción interactiva ingresando Malla 200
=============================================================

REQUISITOS (instalar una sola vez en terminal):
  pip install pandas numpy scipy statsmodels scikit-learn matplotlib
"""

import sys
import os

# Forzar UTF-8 en la consola (evita errores de encoding en Windows)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

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
# CONFIGURACIÓN  ← cambia aquí la ruta si mueves el archivo
# ─────────────────────────────────────────────────────────────
ARCHIVO_CSV = r"C:\Users\Rommel2025\Downloads\Data_Ordenada_Lineas_Agitador_Torta (1).csv"
COLUMNA_X   = "AGIT -200M"
COLUMNA_Y   = "AGIT -325M"


# ═══════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  CARGA DE DATOS")
print("=" * 60)

df    = pd.read_csv(ARCHIVO_CSV)
datos = df[[COLUMNA_X, COLUMNA_Y]].dropna().copy()
datos.columns = ["malla_200", "malla_325"]

# Eliminar valor erróneo evidente (877.0 es typo de 87.7)
datos = datos[datos["malla_200"] < 200].copy()

print(f"Filas cargadas (ambas columnas presentes): {len(datos)}")
print("\nEstadísticas iniciales:")
print(datos.describe().round(2))


# ═══════════════════════════════════════════════════════════════
# 2. ELIMINACIÓN DE OUTLIERS (IQR + Distancia de Cook)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  ELIMINACIÓN DE OUTLIERS")
print("=" * 60)

def mascara_iqr(serie, factor=1.5):
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    return (serie >= q1 - factor * iqr) & (serie <= q3 + factor * iqr)

mask_iqr  = mascara_iqr(datos["malla_200"]) & mascara_iqr(datos["malla_325"])
datos_iqr = datos[mask_iqr].copy()
print(f"Outliers eliminados por IQR:         {len(datos) - len(datos_iqr)}")

X_sm      = sm.add_constant(datos_iqr["malla_200"])
modelo_sm = sm.OLS(datos_iqr["malla_325"], X_sm).fit()
cooks_d   = modelo_sm.get_influence().cooks_distance[0]
mask_cook = cooks_d < 4 / len(datos_iqr)

datos_clean = datos_iqr[mask_cook].reset_index(drop=True)
print(f"Outliers eliminados por Cook's D:    {mask_cook.sum().__rsub__(len(datos_iqr))}")
print(f"Total outliers eliminados:           {len(datos) - len(datos_clean)}")
print(f"Puntos limpios restantes:            {len(datos_clean)}")


# ═══════════════════════════════════════════════════════════════
# 3. GRÁFICAS DE CORRELACIÓN (antes y después)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GENERANDO GRÁFICAS DE CORRELACIÓN")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
fig.patch.set_facecolor("#FAFAFA")
fig.suptitle("Correlación AGIT -200M vs AGIT -325M",
             fontsize=15, fontweight="bold", y=1.01)

for ax, xd, yd, titulo, color in [
    (axes[0], datos["malla_200"],       datos["malla_325"],       "Con outliers", "#378ADD"),
    (axes[1], datos_clean["malla_200"], datos_clean["malla_325"], "Sin outliers", "#1D9E75"),
]:
    ax.set_facecolor("#FAFAFA")
    sl, ic, rv, _, _ = stats.linregress(xd, yd)
    xl = np.linspace(xd.min(), xd.max(), 300)

    ax.scatter(xd, yd, color=color, alpha=0.4, edgecolors="none", s=18, zorder=3)
    ax.plot(xl, sl * xl + ic, color="#D85A30", lw=2, ls="--",
            zorder=4, label="Regresión lineal")

    texto = (f"n  = {len(xd)}\n"
             f"R  = {rv:.3f}\n"
             f"R² = {rv**2:.3f}\n"
             f"y  = {sl:.3f}x + {ic:.2f}")
    props = dict(boxstyle="round,pad=0.5", facecolor="white",
                 edgecolor="#CCCCCC", alpha=0.9)
    ax.text(0.97, 0.05, texto, transform=ax.transAxes, fontsize=9.5,
            va="bottom", ha="right", bbox=props, family="monospace")

    ax.set_xlabel("AGIT -200M (%)", fontsize=11)
    ax.set_ylabel("AGIT -325M (%)", fontsize=11)
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig("correlacion_outliers.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Guardada: correlacion_outliers.png")


# ═══════════════════════════════════════════════════════════════
# 4. PREPARACIÓN PARA MODELOS
# ═══════════════════════════════════════════════════════════════
X = datos_clean[["malla_200"]].values
y = datos_clean["malla_325"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def calcular_metricas(nombre, y_real, y_pred):
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mae  = mean_absolute_error(y_real, y_pred)
    r2   = r2_score(y_real, y_pred)
    return {"modelo": nombre, "RMSE": rmse, "MAE": mae, "R2": r2}

resultados = []

# ─────────────────────────────────────────────────────────────
# 4a. Regresión Lineal Simple
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MODELOS DE PREDICCIÓN")
print("=" * 60)
print("\n[a] Regresión Lineal Simple")

modelo_lin = LinearRegression().fit(X_train, y_train)
y_pred_lin = modelo_lin.predict(X_test)
resultados.append(calcular_metricas("Regresión Lineal", y_test, y_pred_lin))
print(f"    Coef: {modelo_lin.coef_[0]:.4f}  |  Intercepto: {modelo_lin.intercept_:.4f}")

# ─────────────────────────────────────────────────────────────
# 4b. Regresión Polinomial (grado 2)
# ─────────────────────────────────────────────────────────────
print("\n[b] Regresión Polinomial (grado 2)")

modelo_poly = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("reg",  LinearRegression()),
]).fit(X_train, y_train)
y_pred_poly = modelo_poly.predict(X_test)
resultados.append(calcular_metricas("Regresión Polinomial g=2", y_test, y_pred_poly))

# ─────────────────────────────────────────────────────────────
# 4c. Random Forest
# ─────────────────────────────────────────────────────────────
print("\n[c] Random Forest")

modelo_rf = RandomForestRegressor(
    n_estimators=200, max_depth=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1
).fit(X_train, y_train)
y_pred_rf = modelo_rf.predict(X_test)
resultados.append(calcular_metricas("Random Forest", y_test, y_pred_rf))

# ─────────────────────────────────────────────────────────────
# 4d. Red Neuronal (MLP)
# ─────────────────────────────────────────────────────────────
print("\n[d] Red Neuronal (MLP)")

modelo_nn = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )),
]).fit(X_train, y_train)
y_pred_nn = modelo_nn.predict(X_test)
resultados.append(calcular_metricas("Red Neuronal (MLP)", y_test, y_pred_nn))


# ═══════════════════════════════════════════════════════════════
# 5. COMPARACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════
df_res = pd.DataFrame(resultados).set_index("modelo")

# Ranking combinado: menor rango promedio = mejor en las 3 métricas
df_res["rank_R2"]   = df_res["R2"].rank(ascending=False)
df_res["rank_RMSE"] = df_res["RMSE"].rank(ascending=True)
df_res["rank_MAE"]  = df_res["MAE"].rank(ascending=True)
df_res["rank_prom"] = (df_res["rank_R2"] + df_res["rank_RMSE"] + df_res["rank_MAE"]) / 3
mejor_global = df_res["rank_prom"].idxmin()

print("\n" + "=" * 65)
print("  COMPARACIÓN DE MODELOS  (datos de prueba — 20%)")
print("=" * 65)
print(f"\n  {'Modelo':<28} {'RMSE':>7} {'MAE':>7} {'R²':>7}")
print(f"  {'-'*52}")
for idx, row in df_res[["RMSE", "MAE", "R2"]].iterrows():
    marca = "  ← MEJOR" if idx == mejor_global else ""
    print(f"  {idx:<28} {row['RMSE']:>7.3f} {row['MAE']:>7.3f} {row['R2']:>7.3f}{marca}")

print(f"\n  {'─'*52}")
print(f"  Mejor R²   → {df_res['R2'].idxmax()}  ({df_res['R2'].max():.4f})")
print(f"  Menor RMSE → {df_res['RMSE'].idxmin()}  ({df_res['RMSE'].min():.4f})")
print(f"  Menor MAE  → {df_res['MAE'].idxmin()}  ({df_res['MAE'].min():.4f})")
print(f"\n  ★ MEJOR MODELO GLOBAL → {mejor_global}")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════
# 5b. Gráfica: curvas de todos los modelos
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  GENERANDO GRÁFICA COMPARATIVA")
print("=" * 60)

x_rango  = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
colores  = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]
nombres  = ["Regresión Lineal", "Regresión Polinomial g=2",
            "Random Forest",    "Red Neuronal (MLP)"]
modelos  = [modelo_lin, modelo_poly, modelo_rf, modelo_nn]

fig2, ax2 = plt.subplots(figsize=(11, 7))
fig2.patch.set_facecolor("#FAFAFA")
ax2.set_facecolor("#FAFAFA")

ax2.scatter(datos_clean["malla_200"], datos_clean["malla_325"],
            color="#888787", alpha=0.25, s=12, zorder=2, label="Datos limpios")

for nombre, modelo, color in zip(nombres, modelos, colores):
    r2_val = df_res.loc[nombre, "R2"]
    lw = 3.0 if nombre == mejor_global else 1.8
    ls = "-"  if nombre == mejor_global else "--"
    etiq = (f"★ {nombre} (R²={r2_val:.3f})  ← MEJOR"
            if nombre == mejor_global else f"{nombre} (R²={r2_val:.3f})")
    ax2.plot(x_rango, modelo.predict(x_rango), color=color,
             lw=lw, ls=ls, label=etiq, zorder=3)

ax2.set_xlabel("AGIT -200M (%)", fontsize=12)
ax2.set_ylabel("AGIT -325M (%)", fontsize=12)
ax2.set_title("Comparación de modelos de predicción\n(línea sólida = mejor modelo)",
              fontsize=13, fontweight="bold", pad=12)
ax2.legend(fontsize=9.5, framealpha=0.9)
ax2.grid(True, ls="--", lw=0.4, alpha=0.5, color="#CCCCCC")
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig("modelos_comparacion.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Guardada: modelos_comparacion.png")


# ═══════════════════════════════════════════════════════════════
# 5c. Gráfica de barras de métricas
# ═══════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.patch.set_facecolor("#FAFAFA")
fig3.suptitle("Comparación de métricas por modelo",
              fontsize=14, fontweight="bold", y=1.01)

nombres_cortos = ["Lineal", "Polinomial", "Random\nForest", "Red\nNeuronal"]

for ax, metrica, titulo in zip(
    axes3,
    ["RMSE", "MAE", "R2"],
    ["RMSE  (menor = mejor)", "MAE  (menor = mejor)", "R²  (mayor = mejor)"],
):
    ax.set_facecolor("#FAFAFA")
    valores  = df_res[metrica].values
    idx_best = np.argmax(valores) if metrica == "R2" else np.argmin(valores)
    barras   = ax.bar(nombres_cortos, valores, color=colores, width=0.55,
                      edgecolor="white", linewidth=0.5)

    # Resaltar la barra ganadora con borde negro
    barras[idx_best].set_edgecolor("black")
    barras[idx_best].set_linewidth(2.5)

    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + max(valores) * 0.01,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="500")

    ax.set_title(titulo, fontsize=11, pad=8)
    ax.set_ylim(0, max(valores) * 1.2)
    ax.grid(axis="y", ls="--", lw=0.4, alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)

plt.tight_layout()
plt.savefig("metricas_modelos.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → Guardada: metricas_modelos.png")


# ═══════════════════════════════════════════════════════════════
# 6. PREDICCIÓN INTERACTIVA
# ═══════════════════════════════════════════════════════════════
modelo_elegido = dict(zip(nombres, modelos))[mejor_global]

print(f"\n{'='*60}")
print(f"  PREDICCIÓN INTERACTIVA")
print(f"  Mejor modelo: {mejor_global}")
print(f"  Ingresa un valor de AGIT -200M para predecir AGIT -325M.")
print(f"  Escribe 'salir' para terminar.")
print(f"{'='*60}\n")

while True:
    entrada = input("  Ingresa AGIT -200M (%): ").strip()
    if entrada.lower() in ("salir", "exit", "q"):
        print("  ¡Hasta luego!")
        break
    try:
        val = float(entrada)
        xv  = np.array([[val]])
        print(f"\n  Valor ingresado: {val:.2f}%")
        print(f"  {'-'*48}")
        print(f"  {'Modelo':<28} {'AGIT -325M pred.':>16}")
        print(f"  {'-'*48}")
        for nombre, modelo in zip(nombres, modelos):
            pred  = modelo.predict(xv)[0]
            marca = "  ← MEJOR" if nombre == mejor_global else ""
            print(f"  {nombre:<28} {pred:>14.2f}%{marca}")
        print(f"  {'-'*48}\n")
    except ValueError:
        print("  Valor inválido. Ingresa un número (ej: 87.5)\n")