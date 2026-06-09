# Módulo 5 — Arquitectura General de la Plataforma

---

## Visión general

High Garden Coffee es una plataforma de análisis de mercado construida íntegramente
con herramientas de código abierto. Corre localmente sin dependencias de APIs externas
de pago ni infraestructura en la nube.

```
Usuario
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│                  app.py  (punto de entrada)                 │
│  st.navigation() → routing entre 5 páginas + home          │
│  CSS global inyectado (sidebar, tema café)                  │
└──────┬──────────────────────────────────────────────────────┘
       │
       ├─── pages/1_EDA.py          ──► src/data_loader.py, src/eda_charts.py
       ├─── pages/2_Forecasting.py  ──► src/forecasting.py
       ├─── pages/3_Clustering.py   ──► src/clustering.py
       ├─── pages/4_Chatbot.py      ──► src/chatbot.py ──► Ollama (localhost:11434)
       └─── pages/5_Business_Case.py  (sin src dedicado)
```

---

## Flujo de datos

```
coffee_db.parquet
       │
       ▼
src/data_loader.py
  load_data()           → df:  55 × 34  (Country, Coffee type, 30 cols de temporadas)
  growth_table(df)      → CAGR, crecimiento total por país
  season_to_year(col)   → "1990/91" → 1990
       │
       ├─► EDA:          df → gráficos Plotly (eda_charts.py)
       │
       ├─► Forecasting:  por cada país → fit_forecast(years, values) → dict de métricas
       │                 forecast_all_countries(df) → DataFrame resumen (55 filas)
       │
       ├─► Clustering:   build_features(df) → 5 features → StandardScaler → KMeans(k)
       │                 PCA(2) → scatter 2D
       │
       └─► Chatbot:      build_context(df) → string ~6K tokens → system prompt
                         + historial → POST Ollama → respuesta LLM
```

---

## Módulos Python reutilizables (`src/`)

| Archivo | Función principal | Retorna |
|---|---|---|
| `data_loader.py` | `load_data()` | DataFrame limpio con YEAR_COLS |
| `eda_charts.py` | Múltiples `fig_*()` | Figuras Plotly |
| `forecasting.py` | `fit_forecast()` | dict con modelo, métricas, predicciones |
| `clustering.py` | `run_clustering()` | DataFrame con cluster + features |
| `chatbot.py` | `build_context()` | String de contexto para system prompt |

---

## Stack tecnológico

| Capa | Tecnología | Versión |
|---|---|---|
| Frontend / App | Streamlit | 1.36+ |
| Datos | Pandas, NumPy, PyArrow | 2.x |
| Visualización | Plotly | 5.x |
| Machine Learning | Scikit-learn | 1.x |
| LLM (local) | Ollama + phi3/mistral/llama3 | Última estable |
| Comunicación LLM | requests (REST) | 2.x |
| Formato de datos | Parquet | — |

---

## Multipage con `st.navigation()`

```python
# app.py
pages = {
    "INICIO": [st.Page(home_page.show, title="Inicio", icon="☕")],
    "ANÁLISIS": [
        st.Page("pages/1_EDA.py",         title="Análisis Exploratorio"),
        st.Page("pages/2_Forecasting.py", title="Pronóstico de Demanda"),
        st.Page("pages/3_Clustering.py",  title="Segmentación de Mercados"),
        st.Page("pages/4_Chatbot.py",     title="CoffeeBot — Asistente IA"),
        st.Page("pages/5_Business_Case.py", title="Caso de Negocio"),
    ],
}
pg = st.navigation(pages)
pg.run()
```

`set_page_config()` solo se llama en `app.py` — nunca en las páginas individuales
(restricción de Streamlit 1.36+).

---

## Caché de Streamlit

| Función | Decorador | Cuándo se invalida |
|---|---|---|
| `load_data()` | `@st.cache_data` | Al cambiar el código fuente o `ttl` |
| `build_context(df)` | `@st.cache_data` | Al cambiar el código fuente o botón manual |
| `run_clustering(df, k)` | `@st.cache_data` | Al cambiar `k` o el código fuente |

---

## Caso de negocio — arquitectura cloud

La página 5 incluye un diagrama de arquitectura cloud de referencia con:

| Componente | Función |
|---|---|
| Ingesta de datos | Carga y validación del dataset parquet |
| Almacenamiento | Object storage (S3 / GCS / Azure Blob) |
| Procesamiento ML | Contenedor con scikit-learn y pandas |
| Selección automática | Cascada de 4 modelos por país |
| Visualización | Streamlit desplegado en contenedor |
| LLM local / cloud | Ollama en local o API externa en producción |

La calculadora de costos estima gastos mensuales para AWS, GCP y Azure
según parámetros configurables (usuarios concurrentes, frecuencia de actualización).

---

## Ejecución local — inicio rápido

```bash
# 1. Clonar e instalar
git clone https://github.com/alejandrarchbold/High-Garden-Coffee.git
cd High-Garden-Coffee
pip install -r requirements.txt

# 2. Iniciar LLM (terminal separada, solo para chatbot)
ollama serve
ollama pull phi3   # o mistral, llama3, llama3.1

# 3. Lanzar la app
streamlit run app.py
# → http://localhost:8501
```
