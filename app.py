import streamlit as st

st.set_page_config(
    page_title="High Garden Coffee — Plataforma de Analisis",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Plataforma de analisis de mercado de cafe — High Garden Coffee",
        "Get Help": None,
        "Report a bug": None,
    },
)

# CSS global aplicado a todas las paginas
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6F4E37 0%, #9B7355 100%) !important;
    color: white;
}
/* Texto general del sidebar — blanco sobre fondo café */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] p { color: white !important; }
/* Navegación — nombres de módulos en blanco */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] span,
[data-testid="stSidebarNav"] *,
[data-testid="stSidebarNavItems"] *,
[data-testid="stSidebar"] [data-testid="stPageLink"] *,
[data-testid="stSidebar"] [data-testid="stPageLink"] span,
[data-testid="stSidebar"] nav a,
[data-testid="stSidebar"] nav span { color: white !important; }
/* Selectbox: fondo claro, texto oscuro */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #fff8f0 !important;
    color: #1a1a1a !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #1a1a1a !important;
}
/* Botón: fondo semitransparente visible */
[data-testid="stSidebar"] button {
    background-color: rgba(255,255,255,0.18) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
}
[data-testid="stSidebar"] button p,
[data-testid="stSidebar"] button span {
    color: white !important;
}
/* Bloques de código */
[data-testid="stSidebar"] pre {
    background-color: #1e1008 !important;
    border: 1px solid #6F4E37 !important;
}
[data-testid="stSidebar"] pre code,
[data-testid="stSidebar"] code {
    color: #f0e6d6 !important;
    background: transparent !important;
}
/* Dataframe */
[data-testid="stSidebar"] [data-testid="stDataFrame"] {
    background: #fff8f0 !important;
}
[data-testid="stSidebar"] [data-testid="stDataFrame"] * {
    color: #1a1a1a !important;
}

/* Reemplazar icono de persona corriendo por reloj de arena */
[data-testid="stStatusWidget"] svg {
    display: none !important;
}
[data-testid="stStatusWidget"] > div > div:first-child::before {
    content: "⏳";
    font-size: 1.15em;
    animation: sandClock 1.2s ease-in-out infinite;
    display: inline-block;
}
@keyframes sandClock {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}

/* Forzar fondo blanco y texto oscuro en contenido principal */
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background-color: #FFFFFF !important;
}
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] li,
[data-testid="stMainBlockContainer"] td,
[data-testid="stMainBlockContainer"] th,
[data-testid="stMainBlockContainer"] div,
[data-testid="stMainBlockContainer"] label {
    color: #1a1a1a !important;
}
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3,
[data-testid="stMainBlockContainer"] h4 {
    color: #3E1F00 !important;
}

/* Clases compartidas */
.card {
    background: white;
    border-radius: 10px;
    padding: 22px;
    border: 1px solid #e0d5cb;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 14px;
    color: #1a1a1a !important;
}
.card * { color: #1a1a1a !important; }
.card h3, .card h4 { color: #3E1F00 !important; }
.kpi-val { font-size: 2.2em; font-weight: 700; color: #6F4E37 !important; }
.kpi-label { font-size: 0.85em; color: #666 !important; margin-top: 4px; }
.section {
    background: white;
    border-radius: 8px;
    padding: 22px;
    margin: 14px 0;
    border: 1px solid #ddd;
    color: #1a1a1a !important;
}
.section * { color: #1a1a1a !important; }
.section h4 { color: #3E1F00 !important; }
.obs {
    background: #F9F5F0;
    border-left: 3px solid #A05020;
    padding: 12px 16px;
    margin: 10px 0;
    font-size: 0.92em;
    color: #333 !important;
}
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("home_page.py",             title="Inicio",                       default=True),
    st.Page("pages/1_EDA.py",           title="Análisis Exploratorio"),
    st.Page("pages/2_Forecasting.py",   title="Pronóstico de Demanda"),
    st.Page("pages/3_Clustering.py",    title="Segmentación de Mercados"),
    st.Page("pages/4_Chatbot.py",       title="CoffeeBot — Asistente IA"),
    st.Page("pages/5_Business_Case.py", title="Caso de Negocio"),
])
pg.run()
