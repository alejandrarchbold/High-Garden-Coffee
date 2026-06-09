import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from src.data_loader import load_data
from src.chatbot import build_context, chat_ollama, check_ollama_status, AVAILABLE_MODELS, SUGGESTED_QUESTIONS
from src.clustering import run_clustering
from src.forecasting import forecast_all_countries

st.markdown("""
<style>
.chat-header {
    background: linear-gradient(135deg, #3E1F00, #6F4E37);
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 20px;
}
.chat-header h2, .chat-header p { color: white !important; }
.rag-card {
    background: #F9F5F0;
    border: 1px solid #C49A6C;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 10px 0;
    color: #1a1a1a !important;
}
.rag-card * { color: #1a1a1a !important; }
.rag-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin: 10px 0;
    padding: 10px 14px;
    background: white;
    border-radius: 8px;
    border-left: 4px solid #6F4E37;
}
.rag-num {
    background: #6F4E37;
    color: white !important;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    flex-shrink: 0;
    font-size: 0.9em;
}
/* Skeleton loading: shimmer café en vez de iconos de deporte */
/* Replace Streamlit loading skeleton animation with calm shimmer */
[data-testid="stSkeleton"] {
    background: linear-gradient(90deg, #F5EDE4 25%, #E8D5C0 50%, #F5EDE4 75%) !important;
    background-size: 200% 100% !important;
    animation: coffeeShimmer 1.8s ease-in-out infinite !important;
    border-radius: 4px;
}
@keyframes coffeeShimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="chat-header">
<h2 style="margin:0">CoffeeBot — Asistente de Análisis de Mercado</h2>
<p style="margin:6px 0 0 0; opacity:0.88; font-size:0.95em">
Consultas en lenguaje natural sobre datos de consumo de café (1990-2020),
resultados de forecasting y segmentación de mercados. LLM local vía Ollama.
</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Configuración del modelo**")
    ollama_ok, available = check_ollama_status()

    if ollama_ok:
        st.success("Ollama activo")
        if available:
            st.caption(f"Modelos instalados: {', '.join(available)}")
    else:
        st.error("Ollama no detectado en localhost:11434")

    model_label   = st.selectbox("Modelo LLM:", options=list(AVAILABLE_MODELS.keys()))
    selected_model = AVAILABLE_MODELS[model_label]

    st.markdown("---")
    st.markdown("**Instalación de Ollama**")
    st.markdown("[ollama.com](https://ollama.com) → instalar → ejecutar:")
    st.code("ollama serve", language="bash")
    st.markdown("Descargar un modelo:")
    st.code("ollama pull phi3\nollama pull mistral\nollama pull llama3\nollama pull llama3.1", language="bash")

    st.markdown("---")
    st.markdown("**Comparación de modelos**")
    st.dataframe(
        {
            "Modelo":    ["Phi-3 Mini",  "Mistral 7B", "Llama 3 8B", "Llama 3.1 8B"],
            "Tamaño":    ["2.3 GB",      "4.1 GB",     "4.7 GB",     "4.7 GB"],
            "Velocidad": ["Muy alta",    "Alta",       "Media",      "Media"],
        },
        hide_index=True, use_container_width=True,
    )
    st.caption("Mínimo recomendado: 8 GB RAM")

    st.markdown("---")
    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.rerun()

# ── Construcción del contexto RAG ────────────────────────────────
@st.cache_data(show_spinner="Cargando datos y ejecutando modelos ML...")
def get_context():
    df = load_data()
    try:
        cluster_df = run_clustering(df, k=4)
    except Exception as e:
        st.warning(f"Clustering no disponible: {e}")
        cluster_df = None
    try:
        forecast_df = forecast_all_countries(df)
    except Exception as e:
        st.warning(f"Forecasting no disponible: {e}")
        forecast_df = None
    return build_context(df, cluster_df, forecast_df)

if st.sidebar.button("Recargar contexto ML", key="reload_ctx"):
    st.cache_data.clear()
    st.rerun()

context = get_context()

# ── Explicación del mecanismo RAG ────────────────────────────────
with st.expander("¿Cómo funciona el CoffeeBot? — Arquitectura RAG", expanded=False):
    st.markdown("""
<div class="rag-card">
<p style="margin-top:0">El CoffeeBot utiliza <strong>RAG (Retrieval-Augmented Generation)</strong>:
en lugar de responder desde su conocimiento general, el modelo recibe los datos reales
del dataset y los resultados de los modelos ML antes de responder.</p>

<div class="rag-step">
  <div class="rag-num">1</div>
  <div><strong>Carga del dataset</strong><br>
  Se lee <code>coffee_db.parquet</code> con los 55 países y 30 temporadas.
  Se calculan CAGR, rankings y estadísticas por país y tipo de café.</div>
</div>

<div class="rag-step">
  <div class="rag-num">2</div>
  <div><strong>Ejecución de los modelos ML</strong><br>
  <em>Clustering K-Means:</em> cada país recibe un segmento (Gigante Consolidado,
  Mediano en Crecimiento, Emergente Dinámico, Pequeño Estable).<br>
  <em>Forecasting (selección automática de modelo):</em> se generan proyecciones de consumo 2025
  con R², MAPE y crecimiento proyectado para cada país.</div>
</div>

<div class="rag-step">
  <div class="rag-num">3</div>
  <div><strong>Construcción del contexto estructurado</strong><br>
  Se compila un texto plano con todos los datos:
  datos históricos · segmento ML · proyección 2025 por cada uno de los 55 países.
  Este texto se inyecta como <em>system prompt</em> antes de cada consulta.</div>
</div>

<div class="rag-step">
  <div class="rag-num">4</div>
  <div><strong>Respuesta fundamentada</strong><br>
  El modelo LLM local (Ollama) recibe: contexto + historial de conversación + pregunta.
  Solo puede responder basándose en los datos provistos — no inventa cifras.</div>
</div>

<p style="margin-bottom:0;font-size:0.88em;color:#666">
Con 1,650 registros (55 países × 30 temporadas), la inyección de contexto completo
es la estrategia correcta — el dataset cabe en un solo prompt. Para datasets de escala
mayor (millones de filas), se usarían embeddings vectoriales (ChromaDB, FAISS, pgvector)
para recuperación semántica selectiva.
</p>
</div>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Preguntas sugeridas ──────────────────────────────────────────
st.markdown("#### Preguntas sugeridas")
cols = st.columns(2)
for i, q in enumerate(SUGGESTED_QUESTIONS):
    with cols[i % 2]:
        if st.button(q, key=f"sq_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner(f"Pensando con {selected_model}..."):
                response = chat_ollama(st.session_state.messages, context, selected_model)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

st.markdown("---")

# ── Historial del chat ───────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Entrada del usuario ──────────────────────────────────────────
if prompt := st.chat_input("Escribe una pregunta sobre el mercado del café..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(f"Pensando con {selected_model}..."):
            response = chat_ollama(st.session_state.messages, context, selected_model)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption(
    f"Modelo activo: {selected_model} · Ollama localhost:11434 · "
    "Datos: High Garden Coffee 1990-2020 · 55 países · 30 temporadas"
)
