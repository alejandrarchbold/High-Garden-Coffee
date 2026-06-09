import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_data
from src.clustering import (
    build_features, fig_elbow, run_clustering,
    fig_cluster_scatter, fig_cluster_profiles, cluster_summary_table,
    CLUSTER_NAMES, CLUSTER_COLORS,
)

st.markdown("""
<style>
h2 { color: #1B4332 !important; border-bottom: 1px solid #52B788; padding-bottom: 6px; }
h3 { color: #2D6A4F !important; }
.seg-card {
    background: #F9FBF7;
    border-left: 4px solid #2D6A4F;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #1a1a1a !important;
}
.seg-card * { color: #1a1a1a !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def get_clustering(k):
    df = load_data()
    return run_clustering(df, k=k), build_features(df)

df = get_data()

st.title("Segmentación de Mercados — K-Means Clustering")
st.markdown(
    "High Garden Coffee | Agrupamiento de 55 países en segmentos estratégicos | "
    "Algoritmo: K-Means con reducción PCA para visualización bidimensional"
)
st.markdown("---")

# ── 1. Fundamento Matemático ─────────────────────────────────────
st.header("1. Fundamento Matemático")

st.subheader("1.1. Función Objetivo de K-Means")
st.markdown("""
<div class="section">
El algoritmo K-Means minimiza la suma de distancias cuadradas intra-cluster,
buscando la asignación de cada punto al centroide más cercano.
</div>""", unsafe_allow_html=True)

col_f1, col_f2 = st.columns(2)
with col_f1:
    st.latex(
        r"J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} "
        r"\|\mathbf{x} - \boldsymbol{\mu}_k\|^2"
    )
    st.caption(
        "Inercia total. C_k es el conjunto de puntos asignados al cluster k, "
        "μ_k es el centroide del cluster k."
    )
with col_f2:
    st.latex(
        r"\boldsymbol{\mu}_k = \frac{1}{|C_k|}\sum_{\mathbf{x} \in C_k} \mathbf{x}"
    )
    st.caption("Actualización del centroide como media de los puntos asignados.")

st.subheader("1.2. Análisis de Componentes Principales (PCA)")
st.markdown("""
<div class="section">
Para visualizar el espacio de 5 dimensiones en 2D se aplica PCA, que proyecta
los datos sobre las direcciones de máxima varianza (autovectores de la matriz de covarianza).
</div>""", unsafe_allow_html=True)
st.latex(
    r"\mathbf{Z} = \mathbf{X} \mathbf{W}, \quad "
    r"\mathbf{W} = \text{eigenvectors}\left(\text{Cov}(\mathbf{X})\right)"
)
st.caption("Proyección PCA. W contiene los dos primeros componentes principales.")

st.subheader("1.3. Variables de Entrada (Features)")
st.markdown("""
<div class="section">
<table style="width:100%;font-size:0.9em;color:#1a1a1a">
<tr style="background:#EBF7EF"><th style="padding:6px 10px">Variable</th><th>Fórmula</th><th>Justificación</th></tr>
<tr><td style="padding:5px 10px">log_total</td><td>log(1 + consumo total)</td><td>Tamaño del mercado (escala logarítmica para reducir sesgo)</td></tr>
<tr style="background:#EBF7EF"><td>log_last</td><td>log(1 + consumo 2019/20)</td><td>Posición actual del mercado</td></tr>
<tr><td>cagr</td><td>(V_2020/V_1990)^(1/29) - 1</td><td>Tasa de crecimiento anual compuesto</td></tr>
<tr style="background:#EBF7EF"><td>cv</td><td>σ / μ</td><td>Coeficiente de variación (volatilidad)</td></tr>
<tr><td>momentum</td><td>(media_últ5 - media_prim5) / media_prim5</td><td>Impulso reciente vs histórico</td></tr>
</table>
Todas las variables se normalizan con StandardScaler antes del ajuste del modelo.
</div>""", unsafe_allow_html=True)

# ── 2. Selección de k ─────────────────────────────────────────────
st.header("2. Selección del Número de Clusters")

st.subheader("2.1. Método del Codo (Elbow Method)")
st.markdown("""
<div class="section">
Se evalúa la inercia J para valores de k de 2 a 10. El "codo" visual corresponde
al punto donde el incremento marginal en reducción de inercia se vuelve despreciable,
indicando el k óptimo.
</div>""", unsafe_allow_html=True)

feats = build_features(df)
col_elbow, col_k = st.columns([3, 1])
with col_elbow:
    st.plotly_chart(fig_elbow(feats), use_container_width=True)
with col_k:
    k = st.slider("Número de clusters (k):", 2, 8, 4)
    st.markdown(
        "La reducción de inercia es sustancial hasta k=4 y marginal para k>4. "
        "Se selecciona k=4 como punto de equilibrio entre parsimonia e "
        "interpretabilidad estratégica."
    )

# ── 3. Resultados ─────────────────────────────────────────────────
with st.spinner("Ejecutando K-Means..."):
    result, feats = get_clustering(k)

st.header("3. Resultados de la Segmentación")

st.subheader("3.1. Visualización en Espacio PCA Bidimensional")
st.plotly_chart(fig_cluster_scatter(result), use_container_width=True)
st.markdown("""
<div class="obs">
Observación: Los dos primeros componentes principales capturan la mayor parte de la
varianza del espacio original. El tamaño de cada burbuja representa el consumo total
acumulado. La separación visual entre segmentos confirma la coherencia del agrupamiento.
</div>""", unsafe_allow_html=True)

# ── 4. Descripción de Segmentos ───────────────────────────────────
st.header("4. Descripción de Segmentos Estratégicos")

for seg_id, seg_name in CLUSTER_NAMES.items():
    seg_data = result[result["cluster"] == seg_id]
    if seg_data.empty:
        continue
    countries_list = seg_data["Country"].tolist()
    avg_cagr = seg_data["cagr"].mean() * 100
    total_m  = seg_data["Total_M"].sum()
    color    = CLUSTER_COLORS[seg_id]
    st.markdown(f"""
<div class="seg-card" style="border-left-color:{color}">
<span style="color:{color};font-size:1.05em">{seg_name}</span>
({len(countries_list)} países) | Consumo total: {total_m:,.1f}M tazas | CAGR promedio: {avg_cagr:.2f}%<br>
<span style="font-size:0.85em;color:#555">{', '.join(countries_list)}</span>
</div>""", unsafe_allow_html=True)

# ── 5. Perfiles y Tabla Resumen ────────────────────────────────────
st.header("5. Perfiles de Variables por Segmento")

st.subheader("5.1. Comparación de Variables Promedio")
st.plotly_chart(fig_cluster_profiles(result), use_container_width=True)

st.subheader("5.2. Tabla Resumen Estadística")
summary = cluster_summary_table(result)
st.dataframe(summary, use_container_width=True)

# ── 6. Detalle por País ───────────────────────────────────────────
st.header("6. Detalle por País")

col_f1, col_f2 = st.columns(2)
with col_f1:
    seg_filter = st.multiselect(
        "Filtrar por segmento:",
        options=result["cluster_name"].unique().tolist(),
        default=result["cluster_name"].unique().tolist(),
    )
with col_f2:
    sort_field = st.selectbox("Ordenar por:", ["Total_M", "cagr", "momentum"])

detail = result[result["cluster_name"].isin(seg_filter)].sort_values(sort_field, ascending=False)
display_detail = detail[["Country", "Coffee_type", "cluster_name", "Total_M", "cagr", "momentum", "cv"]].rename(
    columns={
        "Coffee_type": "Tipo", "cluster_name": "Segmento",
        "Total_M": "Consumo Total (M)", "cagr": "CAGR", "momentum": "Momentum", "cv": "Volatilidad"
    }
).round(4)

st.dataframe(display_detail, use_container_width=True, height=380)

st.markdown("---")
st.download_button(
    "Descargar segmentación como CSV",
    data=display_detail.to_csv(index=False).encode("utf-8"),
    file_name="segmentacion_kmeans_highgarden.csv",
    mime="text/csv",
)

# ── 7. Implicaciones Estratégicas ────────────────────────────────
st.header("7. Implicaciones Estratégicas por Segmento")

recs_by_rank = [
    ("#6F4E37",
     "Estrategia de retención y diferenciación de valor. Son el núcleo del negocio. "
     "Invertir en relaciones de largo plazo, contratos de suministro garantizado y "
     "certificaciones de calidad (Fair Trade, Rainforest Alliance)."),
    ("#2E8B57",
     "Estrategia de penetración agresiva. CAGR alto con volumen creciente. "
     "Priorizar presencia de marca, alianzas con distribuidores locales y adaptación "
     "del producto al perfil de consumo regional."),
    ("#C49A6C",
     "Monitoreo con bajo costo de servicio. CAGR negativo o estancado — evaluar causas "
     "estructurales antes de nuevas inversiones. Identificar si el declive es temporal."),
    ("#1E3A5F",
     "Mercados con datos limitados o consumo mínimo. No se recomienda inversión a corto "
     "plazo. Reevaluar si se obtienen datos de consumo relevantes."),
]

for seg_id in range(min(k, len(recs_by_rank))):
    seg_data = result[result["cluster"] == seg_id]
    if seg_data.empty:
        continue
    seg_name = CLUSTER_NAMES.get(seg_id, f"Segmento {seg_id + 1}")
    color, text = recs_by_rank[seg_id]
    st.markdown(f"""
<div class="seg-card" style="border-left-color:{color}">
<span style="color:{color}">{seg_name}</span><br>
<span style="font-size:0.9em">{text}</span>
</div>""", unsafe_allow_html=True)
