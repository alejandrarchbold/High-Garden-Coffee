import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.data_loader import load_data
from src.forecasting import fig_forecast, forecast_all_countries, fig_top_forecast

st.markdown("""
<style>
h2 { color: #1E3A5F !important; border-bottom: 1px solid #6A8CBF; padding-bottom: 6px; }
h3 { color: #2D4E7A !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data()

df = get_data()

st.title("Pronóstico de Demanda — Regresión de Tendencia")
st.markdown(
    "High Garden Coffee | Proyecciones de consumo doméstico 2020–2025 | "
    "Método: Regresión lineal / polinómica sobre tendencia histórica 1990–2019"
)
st.markdown("---")

# ── 1. Fundamento Matemático ──────────────────────────────────────
st.header("1. Fundamento Matemático del Modelo")

st.subheader("1.1. Especificación del Modelo")
st.markdown("""
<div class="section">
Se ajusta la <strong>tendencia de largo plazo</strong> para cada país usando 30 años de datos históricos
(1990–2019). Para mercados con crecimiento lineal estable se aplica OLS; para mercados con
cambios de dirección se selecciona automáticamente un polinomio de grado 2. La proyección
extrapola la tendencia ajustada hacia 2020–2025.
</div>""", unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.latex(r"\hat{y}_t = \beta_0 + \beta_1 \cdot t + \varepsilon_t \quad \text{(Lineal OLS)}")
    st.caption("Modelo base. t = año (1990–2019), β₁ = pendiente de crecimiento anual.")
    st.latex(r"\hat{y}_t = \beta_0 + \beta_1 t + \beta_2 t^2 \quad \text{(Polinómico)}")
    st.caption("Se activa cuando R² del modelo lineal < 0.70 y mejora ≥ 5 pp.")

with col_m2:
    st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}}")
    st.caption("R² en muestra — qué tan bien la tendencia explica el comportamiento histórico.")
    st.latex(r"MAPE = \frac{100}{n}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right|")
    st.caption("Error porcentual absoluto medio sobre los 30 años históricos.")

st.subheader("1.2. Esquema de Ajuste y Proyección")
st.markdown("""
<div class="section">
<table style="width:100%;font-size:0.9em;color:#1a1a1a">
<tr style="background:#EEF2FF">
  <th style="padding:6px 10px">Etapa</th><th>Período</th><th>Años</th><th>Propósito</th>
</tr>
<tr>
  <td style="padding:5px 10px">Ajuste de tendencia</td>
  <td>1990/91 – 2019/20</td><td>30</td>
  <td>Estimación de β — máxima información histórica disponible</td>
</tr>
<tr style="background:#EEF2FF">
  <td>Proyección</td>
  <td>2020/21 – 2025/26</td><td>6</td>
  <td>Extrapolación de tendencia con banda de incertidumbre ±1.5σ</td>
</tr>
</table>
</div>""", unsafe_allow_html=True)

# ── 2. Pronóstico por País ────────────────────────────────────────
st.header("2. Pronóstico Individual por País")

countries = sorted(df["Country"].unique().tolist())
col_sel, col_type = st.columns([3, 1])
with col_sel:
    country = st.selectbox("País:", countries, index=countries.index("Colombia"))
with col_type:
    coffee_type = df[df["Country"] == country]["Coffee type"].iloc[0]
    st.markdown(f"Tipo de café: **{coffee_type}**")

fig, result = fig_forecast(country, df)
st.plotly_chart(fig, use_container_width=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("R² en muestra", f"{result['r2']:.3f}")
m2.metric("MAPE histórico", f"{result['mape']:.1f}%")
m3.metric("MAE histórico", f"{result['mae']/1e6:.2f}M tazas")
m4.metric("Modelo", result["model_type"])

forecast_rows = pd.DataFrame({
    "Año": result["future_years"],
    "Proyección (M tazas)": (result["future_pred"] / 1e6).round(2),
    "Límite inferior (M)": (np.maximum(result["future_pred"] - result["margin"], 0) / 1e6).round(2),
    "Límite superior (M)": ((result["future_pred"] + result["margin"]) / 1e6).round(2),
})
with st.expander("Ver tabla de proyecciones numéricas"):
    st.dataframe(forecast_rows, use_container_width=True)

st.markdown("""
<div class="obs">
Nota metodológica: El R² mide el ajuste sobre los 30 años de datos históricos (en muestra),
no sobre un conjunto de validación separado. La banda de incertidumbre se construye como ±1.5
desviaciones estándar de los residuales históricos y representa la variabilidad típica del mercado
en relación con la tendencia ajustada.
</div>""", unsafe_allow_html=True)

# ── 3. Ranking de Crecimiento Proyectado ─────────────────────────
st.header("3. Ranking de Crecimiento Proyectado (2019→2025)")

with st.spinner("Calculando proyecciones para todos los países..."):
    summary = forecast_all_countries(df)

st.plotly_chart(fig_top_forecast(df, 10), use_container_width=True)

# ── 4. Tabla Resumen Global ───────────────────────────────────────
st.header("4. Tabla Resumen de Proyecciones")

display_summary = summary.rename(columns={
    "Consumo_2019_M":    "Consumo 2019 (M)",
    "Proyeccion_2025_M": "Proyección 2025 (M)",
    "Crecimiento_%":     "Crecimiento %",
    "MAE_M":             "MAE (M)",
    "MAPE_%":            "MAPE %",
    "R2":                "R²",
    "Coffee type":       "Tipo",
}).round(2)

col_filter, col_sort = st.columns(2)
with col_filter:
    type_filter = st.multiselect(
        "Filtrar por tipo de café:",
        options=df["Coffee type"].unique().tolist(),
        default=df["Coffee type"].unique().tolist(),
    )
with col_sort:
    sort_col = st.selectbox("Ordenar por:", ["Crecimiento %", "Proyección 2025 (M)", "R²"])

filtered = display_summary[display_summary["Tipo"].isin(type_filter)].sort_values(
    sort_col, ascending=False
)
st.dataframe(
    filtered[["Country", "Tipo", "Consumo 2019 (M)", "Proyección 2025 (M)", "Crecimiento %", "MAPE %", "R²"]],
    use_container_width=True,
    height=400,
)

# ── 5. Evaluación Global del Modelo ──────────────────────────────
st.header("5. Evaluación Global del Modelo")

col_ev1, col_ev2, col_ev3 = st.columns(3)
with col_ev1:
    avg_r2 = summary["R2"].mean()
    good   = (summary["R2"] > 0.80).sum()
    st.metric("R² promedio (en muestra)", f"{avg_r2:.3f}")
    st.caption(f"{good} de {len(summary)} países con R² > 0.80")
with col_ev2:
    med_mape = summary["MAPE_%"].median()
    st.metric("MAPE mediana", f"{med_mape:.1f}%")
    st.caption("Mediana — robusta frente a países con consumo mínimo")
with col_ev3:
    growth_pos = (summary["Crecimiento_%"] > 0).sum()
    st.metric("Países con proyección positiva", f"{growth_pos}/{len(summary)}")

fig_r2 = px.histogram(
    summary, x="R2", nbins=20,
    title="Distribución del coeficiente R² — todos los países",
    labels={"R2": "R²", "count": "Frecuencia"},
    color_discrete_sequence=["#1E3A5F"],
)
fig_r2.add_vline(x=0.8, line_dash="dot", line_color="#A05020", annotation_text="R²=0.80")
st.plotly_chart(fig_r2, use_container_width=True)

st.subheader("5.1. Distribución por Tipo de Modelo Seleccionado")
model_counts = summary["Modelo"].value_counts().reset_index()
model_counts.columns = ["Modelo", "Países"]
fig_m = px.pie(
    model_counts, names="Modelo", values="Países",
    color_discrete_sequence=["#6F4E37", "#A0522D"],
    title="Proporción de países por tipo de modelo",
)
st.plotly_chart(fig_m, use_container_width=True)

st.markdown("""
<div class="obs">
Observación: El R² en muestra refleja qué tan bien la tendencia ajustada captura el comportamiento
histórico 1990–2019. Para mercados con alta volatilidad o cambios de ciclo (ex. Viet Nam, Colombia)
el modelo polinómico captura la curvatura. Mercados con crecimiento estable (ex. Brasil, Alemania)
ajustan mejor con OLS lineal. Para proyecciones de mayor horizonte o granularidad mensual se
recomendaría ARIMA o Prophet.
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.download_button(
    "Descargar proyecciones como CSV",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name="proyecciones_2025_highgarden.csv",
    mime="text/csv",
)
