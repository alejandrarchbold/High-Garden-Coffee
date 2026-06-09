import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from src.data_loader import load_data, top_countries, coffee_type_summary, growth_table, YEAR_COLS, YEAR_NUMS
from src.eda_charts import (
    fig_top_countries, fig_coffee_type_pie, fig_coffee_type_bar,
    fig_global_trend, fig_country_trends, fig_growth_rates,
    fig_cagr_scatter, fig_boxplot_by_type,
    fig_heatmap_countries, fig_cumulative_share,
)

st.markdown("""
<style>
h2 { color: #3E1F00 !important; border-bottom: 1px solid #D4956A; padding-bottom: 6px; }
h3 { color: #5A3010 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data()

df = get_data()

# ── Portada ──────────────────────────────────────────────────────
st.title("Análisis Exploratorio de Datos")
st.markdown(
    "High Garden Coffee — Consumo doméstico de café (1990/91 a 2019/20)"
)
st.markdown("---")

# ── 1. Descripción del Conjunto de Datos ────────────────────────
st.header("1. Descripción del Conjunto de Datos")

st.subheader("1.1. Estructura y Dimensiones")

n_rows, n_cols = df.shape
year_cols_only = [c for c in df.columns if "/" in c]
non_year = [c for c in df.columns if "/" not in c]

col_info, col_shape = st.columns(2)

with col_info:
    st.markdown("""
<div class="section">
<h4>1.1.1. Ficha Técnica del Dataset</h4>
<table style="width:100%;font-size:0.9em;color:#1a1a1a">
<tr style="background:#F5EDE4"><th style="padding:6px 10px">Atributo</th><th>Valor</th></tr>
<tr><td style="padding:5px 10px">Filas (registros)</td><td>{rows}</td></tr>
<tr style="background:#F5EDE4"><td style="padding:5px 10px">Columnas totales</td><td>{cols}</td></tr>
<tr><td style="padding:5px 10px">Columnas temporales</td><td>{yr} (una por temporada cafetalera)</td></tr>
<tr style="background:#F5EDE4"><td style="padding:5px 10px">Columnas categóricas</td><td>{non_yr}</td></tr>
<tr><td style="padding:5px 10px">Rango temporal</td><td>1990/91 a 2019/20 (30 temporadas)</td></tr>
<tr style="background:#F5EDE4"><td style="padding:5px 10px">Valores nulos</td><td>0 (dataset completo)</td></tr>
<tr><td style="padding:5px 10px">Tipo de dato temporal</td><td>int64 (tazas absolutas)</td></tr>
<tr style="background:#F5EDE4"><td style="padding:5px 10px">Países únicos</td><td>{countries}</td></tr>
<tr><td style="padding:5px 10px">Tipos de café únicos</td><td>{types}</td></tr>
</table>
</div>""".format(
        rows=n_rows, cols=n_cols, yr=len(year_cols_only),
        non_yr=len(non_year), countries=df["Country"].nunique(),
        types=df["Coffee type"].nunique(),
    ), unsafe_allow_html=True)

with col_shape:
    st.markdown("""
<div class="section">
<h4>1.1.2. Variables del Dataset</h4>
<table style="width:100%;font-size:0.9em;color:#1a1a1a">
<tr style="background:#F5EDE4"><th style="padding:6px 10px">Variable</th><th>Tipo</th><th>Descripción</th></tr>
<tr><td style="padding:5px 10px">Country</td><td>object</td><td>Nombre del país productor</td></tr>
<tr style="background:#F5EDE4"><td>Coffee type</td><td>object</td><td>Tipo de café dominante</td></tr>
<tr><td>1990/91 ... 2019/20</td><td>int64</td><td>Consumo doméstico anual en tazas</td></tr>
<tr style="background:#F5EDE4"><td>Total_domestic_consumption</td><td>int64</td><td>Suma acumulada de 30 temporadas</td></tr>
</table>
</div>""", unsafe_allow_html=True)

st.subheader("1.2. Muestra de Datos")
show_cols = ["Country", "Coffee type", "1990/91", "2000/01", "2010/11", "2019/20", "Total_domestic_consumption"]
st.dataframe(df[show_cols], use_container_width=True, height=250)
st.caption(f"Mostrando las {n_rows} filas del dataset. Valores en tazas absolutas.")

# ── 2. Estadísticas Descriptivas ────────────────────────────────
st.header("2. Estadísticas Descriptivas")

st.subheader("2.1. Medidas de Tendencia Central y Dispersión")

st.markdown("""
<div class="section">
Las medidas de tendencia central describen el valor típico de la distribución,
mientras las de dispersión cuantifican la variabilidad. Las fórmulas empleadas son:
</div>""", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.latex(r"\mu = \frac{1}{N}\sum_{i=1}^{N} x_i")
    st.caption("Media aritmética")
with col_f2:
    st.latex(r"\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}")
    st.caption("Desviación estándar")
with col_f3:
    st.latex(r"CV = \frac{\sigma}{\mu} \times 100")
    st.caption("Coeficiente de variación (%)")

desc = df[YEAR_COLS].describe().T
desc.index = [y.split("/")[0] for y in desc.index]
desc_M = (desc[["mean", "std", "min", "50%", "max"]] / 1e6).round(2)
desc_M.columns = ["Media (M)", "Desv.Std (M)", "Mínimo (M)", "Mediana (M)", "Máximo (M)"]
st.dataframe(desc_M, use_container_width=True, height=280)
st.caption("Valores en millones de tazas. Cada fila corresponde a una temporada cafetalera.")

st.markdown("""
<div class="obs">
Observación: La media y la mediana divergen significativamente en todos los años,
indicando una distribución fuertemente asimétrica a la derecha (sesgo positivo).
Unos pocos países con gran volumen elevan la media por encima de la mediana.
El coeficiente de variación supera el 200% en la mayoría de temporadas, lo que
refleja la alta heterogeneidad entre mercados.
</div>""", unsafe_allow_html=True)

total = df["Total_domestic_consumption"].sum()
g1990 = df["1990/91"].sum()
g2020 = df["2019/20"].sum()
growth = (g2020 - g1990) / g1990 * 100

st.subheader("2.2. Indicadores Globales")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Consumo acumulado (30 años)", f"{total/1e9:.2f}B tazas")
k2.metric("Consumo global 1990", f"{g1990/1e6:,.0f}M")
k3.metric("Consumo global 2020", f"{g2020/1e6:,.0f}M")
k4.metric("Crecimiento 1990–2020", f"{growth:.0f}%")
k5.metric("Países sin consumo inicial", int((df["1990/91"] == 0).sum()))

# ── 3. Análisis de Distribución por País ────────────────────────
st.header("3. Análisis de Distribución por País")

st.subheader("3.1. Clasificación por Consumo Total Acumulado")
n_countries = st.slider("Número de países a mostrar", 5, 20, 10, key="top_n")
st.plotly_chart(fig_top_countries(df, n_countries), use_container_width=True)

top5 = top_countries(df, 5)
top5_share = top5["Total_domestic_consumption"].sum() / total * 100
st.markdown(f"""
<div class="obs">
Observación: Brasil lidera el consumo acumulado con una diferencia de orden de magnitud
respecto al segundo lugar. Los 5 principales mercados concentran el {top5_share:.1f}% del
consumo global total. Esta concentración implica un riesgo de dependencia estratégica.
</div>""", unsafe_allow_html=True)

# ── 4. Análisis por Tipo de Café ─────────────────────────────────
st.header("4. Análisis por Tipo de Café")

st.subheader("4.1. Participación de Mercado por Variedad")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(fig_coffee_type_pie(df), use_container_width=True)
with c2:
    st.plotly_chart(fig_coffee_type_bar(df), use_container_width=True)

st.markdown("""
<div class="obs">
Observación: El tipo Arábica/Robusta domina con el 53.5% del consumo total,
impulsado por Brasil (mayor mercado global). El tipo Robusta/Arábica representa el 22.7%,
concentrado en mercados asiáticos de alto crecimiento: Vietnam, India, Indonesia y Filipinas.
El Arábica puro (21.5%) corresponde principalmente a América Latina productora de calidad
(Colombia, Costa Rica, Etiopía).
</div>""", unsafe_allow_html=True)

# ── 5. Análisis de Tendencia Temporal ───────────────────────────
st.header("5. Análisis de Tendencia Temporal")

st.subheader("5.1. Serie de Tiempo Global")
st.plotly_chart(fig_global_trend(df), use_container_width=True)
st.markdown(f"""
<div class="obs">
Observación: El consumo global pasó de {g1990/1e6:,.0f} millones de tazas en 1990
a {g2020/1e6:,.0f} millones en 2020, representando un incremento del {growth:.1f}%.
La tendencia se acelera a partir del año 2000, correlacionando con el crecimiento
económico de mercados emergentes en Asia y África.
</div>""", unsafe_allow_html=True)

st.subheader("5.2. Evolución por País (Top 10)")
st.plotly_chart(fig_country_trends(df, 10), use_container_width=True)
st.markdown("""
<div class="obs">
Observación: Se identifican tres patrones de comportamiento distintos:
(a) crecimiento fuerte sostenido (Filipinas CAGR 5.3%, Indonesia 4.8%, Etiopía 4.0%, Brasil 3.5%),
(b) crecimiento moderado (México 2.0%, Colombia 1.7%, India 1.7%), y
(c) estancamiento o declive (Costa Rica CAGR −0.2%, Camerún).
</div>""", unsafe_allow_html=True)

# ── 6. Análisis de Tasas de Crecimiento ─────────────────────────
st.header("6. Análisis de Tasas de Crecimiento")

st.subheader("6.1. Fórmula de Crecimiento Acumulado y CAGR")
col_cg1, col_cg2 = st.columns(2)
with col_cg1:
    st.latex(r"g_{total} = \frac{V_{2020} - V_{1990}}{V_{1990}} \times 100")
    st.caption("Tasa de crecimiento total 1990–2020 (%)")
with col_cg2:
    st.latex(r"CAGR = \left(\frac{V_{2020}}{V_{1990}}\right)^{\frac{1}{29}} - 1")
    st.caption("Tasa de crecimiento anual compuesto (29 períodos)")

tab_growth, tab_decline = st.tabs(["Mayor Crecimiento", "Mercados en Declive"])
with tab_growth:
    st.plotly_chart(fig_growth_rates(df, 15), use_container_width=True)
with tab_decline:
    gt = growth_table(df).dropna(subset=["growth_pct"]).sort_values("growth_pct").head(10)
    st.dataframe(
        gt[["Country", "1990/91", "2019/20", "growth_pct", "cagr"]].assign(**{
            "1990/91": lambda x: (x["1990/91"] / 1e6).round(1),
            "2019/20": lambda x: (x["2019/20"] / 1e6).round(1),
            "growth_pct": lambda x: x["growth_pct"].round(1),
            "cagr": lambda x: (x["cagr"] * 100).round(2),
        }).rename(columns={"growth_pct": "Crecimiento %", "cagr": "CAGR %"}),
        use_container_width=True
    )
    st.caption("Valores de consumo en millones de tazas")

# ── 7. Diagrama de Dispersión Tamaño vs CAGR ────────────────────
st.header("7. Relación Tamaño de Mercado y CAGR")
st.subheader("7.1. Diagrama de Burbujas (escala logarítmica)")
st.plotly_chart(fig_cagr_scatter(df), use_container_width=True)
st.markdown("""
<div class="obs">
Observación: El cuadrante superior-derecho (alto volumen + alto CAGR) representa los
mercados de mayor prioridad estratégica. El cuadrante superior-izquierdo agrupa
mercados emergentes con alto potencial y bajo volumen actual: Vietnam, Laos, Yemen.
</div>""", unsafe_allow_html=True)

# ── 8. Distribución por Tipo de Café ────────────────────────────
st.header("8. Distribución del Consumo por Tipo de Café")
st.plotly_chart(fig_boxplot_by_type(df), use_container_width=True)

# ── 9. Mapa de Calor Histórico ───────────────────────────────────
st.header("9. Mapa de Calor Histórico")
st.subheader("9.1. Consumo Anual — Top 15 Países")
st.plotly_chart(fig_heatmap_countries(df, 15), use_container_width=True)

# ── 10. Participación de Mercado ─────────────────────────────────
st.header("10. Evolución de la Participación de Mercado")
st.subheader("10.1. Share Relativo Anual — Top 5 + Resto")
st.plotly_chart(fig_cumulative_share(df), use_container_width=True)
st.markdown("""
<div class="obs">
Observación: La participación relativa de Brasil se reduce gradualmente desde 1990,
mientras la categoría "Otros" (50 países) gana peso, indicando una progresiva
diversificación del mercado global. Esto reduce el riesgo de concentración a largo plazo.
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.download_button(
    "Descargar estadísticas descriptivas como CSV",
    data=desc_M.to_csv().encode("utf-8"),
    file_name="estadisticas_eda_highgarden.csv",
    mime="text/csv",
)
