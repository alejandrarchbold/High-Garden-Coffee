import streamlit as st

st.title("High Garden Coffee — Plataforma de Análisis de Mercado")
st.markdown(
    "Plataforma de inteligencia artificial para el análisis del consumo doméstico de café (1990–2020)."
)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="card"><div class="kpi-val">55</div><div class="kpi-label">Países analizados</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="card"><div class="kpi-val">30</div><div class="kpi-label">Temporadas cafetaleras (1990–2020)</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="card"><div class="kpi-val">4</div><div class="kpi-label">Tipos de café</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="card"><div class="kpi-val">+137%</div><div class="kpi-label">Crecimiento global 1990–2020</div></div>', unsafe_allow_html=True)

st.markdown("---")
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
<div class="card">
<h3 style="color:#3E1F00;margin-top:0">Módulos de la plataforma</h3>
<table style="width:100%;font-size:0.9em;color:#1a1a1a">
<tr style="background:#F5EDE4">
  <th style="padding:6px 10px;color:#1a1a1a">Sección</th>
  <th style="color:#1a1a1a">Descripción</th>
</tr>
<tr>
  <td style="padding:5px 10px;color:#1a1a1a">Análisis Exploratorio</td>
  <td style="color:#1a1a1a">Tendencias, distribuciones y rankings por país y tipo de café</td>
</tr>
<tr style="background:#F5EDE4">
  <td style="color:#1a1a1a">Pronóstico de Demanda</td>
  <td style="color:#1a1a1a">Proyecciones 2020–2025 por país con selección automática de modelo</td>
</tr>
<tr>
  <td style="color:#1a1a1a">Segmentación de Mercados</td>
  <td style="color:#1a1a1a">Agrupamiento estratégico con K-Means y reducción PCA</td>
</tr>
<tr style="background:#F5EDE4">
  <td style="color:#1a1a1a">CoffeeBot — Asistente IA</td>
  <td style="color:#1a1a1a">LLM de código abierto con contexto RAG (Ollama local)</td>
</tr>
<tr>
  <td style="color:#1a1a1a">Caso de Negocio</td>
  <td style="color:#1a1a1a">Roadmap, arquitectura cloud y calculadora de costos</td>
</tr>
</table>
</div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="card">
<h3 style="color:#3E1F00;margin-top:0">Contexto empresarial</h3>
<p style="color:#1a1a1a">High Garden Coffee — exportadora de café de nivel internacional.</p>
<p style="color:#1a1a1a">Objetivo: aprovechar datos históricos de consumo para obtener ventaja competitiva mediante:</p>
<ul>
  <li style="color:#1a1a1a">Identificación de tendencias y mercados emergentes</li>
  <li style="color:#1a1a1a">Proyección de demanda futura por mercado</li>
  <li style="color:#1a1a1a">Segmentación estratégica de mercados</li>
  <li style="color:#1a1a1a">Asistente inteligente para el equipo de innovación</li>
</ul>
<p style="color:#1a1a1a"><strong>Stack:</strong> Python · Pandas · Plotly · Scikit-learn · Ollama · Streamlit</p>
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div class="card">
<h3 style="color:#3E1F00;margin-top:0">Cómo navegar</h3>
<p style="color:#1a1a1a">
Usar la barra lateral izquierda para moverse entre módulos. Ruta recomendada:
<strong>Análisis Exploratorio</strong> → <strong>Pronóstico de Demanda</strong> →
<strong>Segmentación de Mercados</strong> → <strong>CoffeeBot</strong> → <strong>Caso de Negocio</strong>.
</p>
</div>""", unsafe_allow_html=True)
