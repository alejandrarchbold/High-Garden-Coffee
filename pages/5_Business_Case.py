import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import datetime
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


st.markdown("""
<style>
.slide { background: white; border-radius: 10px; padding: 24px;
         border: 1px solid #e0d5cb; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin: 14px 0; }
.section { background: white; border-radius: 8px; padding: 22px;
           margin: 14px 0; border: 1px solid #ddd; }
.phase { border-radius: 8px; padding: 16px; margin: 8px 0; border-left: 4px solid; background: #FAFAFA; }
.cloud-card { background: white; border-radius: 10px; padding: 18px;
              border: 2px solid; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.kpi-box { background: #FAF6F1; border-radius: 8px; padding: 14px;
           text-align: center; border: 1px solid #e0d5cb; }
h1, h2 { color: #3E1F00; }
h3 { color: #5A3010; }
</style>
""", unsafe_allow_html=True)

st.title("Caso de Negocio — High Garden Coffee AI Platform")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Propuesta", "Roadmap", "Arquitectura", "Calculadora Cloud", "Entregables"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1: PROPUESTA
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.header("1. Definición del Problema de Negocio")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="slide">
<h4 style="color:#3E1F00;margin-top:0">1.1. Contexto Empresarial</h4>
High Garden Coffee es una exportadora de café de nivel internacional.
El área de innovación dispone de una base de datos de consumo doméstico (en tazas)
que abarca 55 países y 30 temporadas cafetaleras (1990/91 a 2019/20).

<br><br>Preguntas clave del negocio:
<ol>
  <li>¿Qué mercados presentan mayor crecimiento proyectado?</li>
  <li>¿Qué tipo de café priorizar en cada región geográfica?</li>
  <li>¿Cómo segmentar los mercados para estrategias diferenciadas?</li>
  <li>¿Cómo democratizar el acceso a datos en el equipo de innovación?</li>
</ol>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="slide">
<h4 style="color:#3E1F00;margin-top:0">1.2. Solución Propuesta</h4>
Plataforma de analítica de IA con 5 módulos integrados:

<br><br>
<table style="width:100%;font-size:0.88em">
<tr style="background:#F5EDE4"><th style="padding:6px 10px">Módulo</th><th>Tecnología</th><th>Valor generado</th></tr>
<tr><td style="padding:5px 10px">EDA</td><td>Pandas + Plotly</td><td>Visibilidad de 30 años de datos</td></tr>
<tr style="background:#F5EDE4"><td>Forecasting</td><td>Scikit-learn</td><td>Proyecciones 2020–2025 con selección de modelo</td></tr>
<tr><td>Clustering</td><td>K-Means + PCA</td><td>Segmentación estratégica</td></tr>
<tr style="background:#F5EDE4"><td>LLM Chatbot</td><td>Ollama (código abierto)</td><td>IA conversacional sin costo por token</td></tr>
<tr><td>Business Case</td><td>Streamlit</td><td>ROI y arquitectura cloud</td></tr>
</table>
</div>""", unsafe_allow_html=True)

    st.header("1.3. Indicadores del Dataset")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [("55","Países"), ("30","Temporadas"), ("4","Tipos de café"),
            ("0","Valores nulos"), ("+137%","Crecimiento global"), ("1,650","Registros totales")]
    for col, (val, label) in zip([k1,k2,k3,k4,k5,k6], kpis):
        col.markdown(
            f'<div class="kpi-box"><div style="font-size:1.9em;font-weight:700;color:#6F4E37">{val}</div>'
            f'<div style="font-size:0.82em;color:#888;margin-top:4px">{label}</div></div>',
            unsafe_allow_html=True
        )

    st.header("1.4. Hallazgos Principales")
    findings = [
        ("#2E8B57", "Mercados de Alto Crecimiento",
         "Vietnam (+1,667%), Filipinas (+352%), Laos, Yemen — CAGR superior al 5% anual. "
         "Representan oportunidades de penetración inmediata con bajo volumen actual."),
        ("#6F4E37", "Concentración Latinoamericana",
         "Brasil concentra aproximadamente el 44% del consumo global acumulado. "
         "Colombia y México son mercados maduros con crecimiento lineal moderado."),
        ("#1E3A5F", "Tendencia Estructural Positiva",
         "Crecimiento del 137% en 30 años. La demanda global de café es estructuralmente "
         "creciente, lo que respalda inversiones de largo plazo en el sector."),
        ("#CC4400", "Mercados en Contracción",
         "Ecuador, Ghana, Zimbabwe y Sri Lanka muestran reducción del consumo. "
         "Requieren análisis de causas y estrategias de recuperación o desinversión."),
    ]
    cols_f = st.columns(2)
    for i, (color, title, desc) in enumerate(findings):
        with cols_f[i % 2]:
            st.markdown(f"""
<div class="slide" style="border-left:4px solid {color}">
<span style="color:{color};font-size:1.05em">{title}</span><br>
<span style="font-size:0.9em">{desc}</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2: ROADMAP
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.header("2. Roadmap de Implementación — 6 Meses")

    phases = [
        ("Fase 1", "Meses 1-2", "Fundamentos de Datos", "#6F4E37",
         ["Pipeline de ingesta y actualización del dataset",
          "EDA automatizado con reportes HTML",
          "Dashboard ejecutivo en Streamlit",
          "Documentación técnica y de negocio"],
         "Resultado: visibilidad completa de los datos históricos. El equipo puede explorar 30 años en minutos."),
        ("Fase 2", "Meses 2-4", "Machine Learning", "#1E3A5F",
         ["Modelos de pronóstico por país (selección automática de modelo)",
          "Clustering de mercados (K-Means + PCA)",
          "Evaluación y validación de modelos",
          "API de predicción (FastAPI)"],
         "Resultado: proyecciones de demanda 2020–2025 y mapa estratégico de segmentación de mercados."),
        ("Fase 3", "Meses 4-5", "IA Generativa y LLM", "#2D6A4F",
         ["Integración con Ollama (modelos de código abierto)",
          "CoffeeBot — chatbot de negocio con contexto RAG",
          "Ajuste del contexto con conocimiento de dominio",
          "Evaluación de respuestas (LLM-as-judge)"],
         "Resultado: cualquier miembro del equipo puede consultar datos en lenguaje natural, sin SQL ni código."),
        ("Fase 4", "Meses 5-6", "Producción y Monitoreo", "#CC4400",
         ["Despliegue en AWS/GCP/Azure",
          "Pipeline CI/CD (GitHub Actions)",
          "Monitoreo de modelos (detección de drift)",
          "Actualización automática con nuevos datos"],
         "Resultado: plataforma en producción, escalable, con actualizaciones automáticas y alertas de negocio."),
    ]

    for phase, months, title, color, tasks, outcome in phases:
        st.markdown(f"""
<div class="phase" style="border-left-color:{color}">
<div style="display:flex;justify-content:space-between;align-items:center">
  <span style="font-size:1.05em;color:{color}">{phase} — {title}</span>
  <span style="background:{color};color:white;padding:3px 12px;border-radius:16px;font-size:0.82em">{months}</span>
</div>
<ul style="margin:10px 0 6px 0;font-size:0.9em">
{"".join(f"<li>{t}</li>" for t in tasks)}
</ul>
<div style="background:{color}20;border-radius:5px;padding:8px 12px;font-size:0.87em">{outcome}</div>
</div>""", unsafe_allow_html=True)

    st.header("2.1. Diagrama de Gantt")
    gantt_data = pd.DataFrame([
        dict(Task="Ingesta y Pipeline",   Start="2024-01-01", Finish="2024-02-15", Phase="Fase 1"),
        dict(Task="EDA y Dashboard",      Start="2024-01-15", Finish="2024-02-28", Phase="Fase 1"),
        dict(Task="ML Forecasting",       Start="2024-02-15", Finish="2024-04-01", Phase="Fase 2"),
        dict(Task="ML Clustering",        Start="2024-03-01", Finish="2024-04-15", Phase="Fase 2"),
        dict(Task="Ollama y RAG",         Start="2024-04-01", Finish="2024-05-15", Phase="Fase 3"),
        dict(Task="CoffeeBot Chatbot",    Start="2024-04-15", Finish="2024-05-31", Phase="Fase 3"),
        dict(Task="Despliegue y CI/CD",   Start="2024-05-15", Finish="2024-06-15", Phase="Fase 4"),
        dict(Task="Monitoreo y MLOps",    Start="2024-05-25", Finish="2024-06-30", Phase="Fase 4"),
    ])
    color_map = {"Fase 1":"#6F4E37","Fase 2":"#1E3A5F","Fase 3":"#2D6A4F","Fase 4":"#CC4400"}
    fig_gantt = px.timeline(
        gantt_data, x_start="Start", x_end="Finish", y="Task", color="Phase",
        color_discrete_map=color_map,
        title="Roadmap de Implementación — High Garden Coffee AI Platform",
    )
    fig_gantt.update_yaxes(categoryorder="total ascending")
    fig_gantt.update_layout(height=380)
    st.plotly_chart(fig_gantt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3: ARQUITECTURA
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.header("3. Arquitectura Técnica de la Solución")

    st.subheader("3.1. Diagrama de Arquitectura")

    # Diagrama con Plotly (sin dependencias externas)
    def make_arch_fig():
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[0, 11], showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white", paper_bgcolor="white",
            height=520, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )

        def box(x0, y0, x1, y1, color, label, sublabel=""):
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          fillcolor=color, line=dict(color="#333", width=1.2))
            text = f"<b>{label}</b>"
            if sublabel:
                text += f"<br><span style='font-size:10px'>{sublabel}</span>"
            fig.add_annotation(x=(x0+x1)/2, y=(y0+y1)/2, text=text,
                               showarrow=False, font=dict(color="white", size=11),
                               align="center")

        def arrow(x, y0, y1):
            fig.add_annotation(x=x, y=y0, ax=x, ay=y1, xref="x", yref="y",
                               axref="x", ayref="y",
                               arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                               arrowcolor="#555", showarrow=True, text="")

        def harrow(x0, x1, y):
            fig.add_annotation(x=x0, y=y, ax=x1, ay=y, xref="x", yref="y",
                               axref="x", ayref="y",
                               arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                               arrowcolor="#555", showarrow=True, text="")

        # Fila 1: Dataset
        box(3.2, 9.4, 6.8, 10.6, "#C49A6C", "coffee_db.parquet", "55 países · 30 temporadas")
        arrow(5, 9.4, 8.6)

        # Fila 2: ETL
        box(3.2, 7.8, 6.8, 9.0, "#8B6347", "ETL y Preprocesamiento", "Pandas · Feature Engineering")
        arrow(5, 7.8, 7.2)

        # Fila 3: Feature Store
        box(3.2, 6.2, 6.8, 7.4, "#6F8B6F", "Feature Store", "CAGR · Volumen · Momentum · CV")
        # Bifurcaciones
        arrow(5,   6.2, 5.6)   # centro → abajo
        harrow(5,  2.5, 5.9)   # centro → izquierda (clustering)
        harrow(5,  7.5, 5.9)   # centro → derecha (RAG)

        # Fila 4: tres módulos paralelos
        box(0.2, 4.4, 2.8, 5.6, "#1E3A5F", "Forecasting ML", "Selección automática")
        box(3.7, 4.4, 6.3, 5.6, "#2D6A4F", "Clustering ML", "K-Means + PCA")
        box(7.2, 4.4, 9.8, 5.6, "#7A3810", "Contexto RAG", "55 países · ML outputs")

        arrow(1.5, 4.4, 3.8)
        arrow(5.0, 4.4, 3.8)
        arrow(8.5, 4.4, 3.8)   # RAG → Ollama

        # Fila 5: Ollama
        box(6.8, 2.6, 9.8, 3.8, "#6F4E37", "LLM Local — Ollama", "Llama3 · Mistral · Phi-3")
        arrow(8.3, 2.6, 2.0)   # Ollama → dashboard

        # Fila 6: Dashboard
        box(2.0, 0.4, 8.0, 1.6, "#3E1F00", "Dashboard Streamlit",
            "Análisis Exploratorio · Pronóstico · Segmentación · CoffeeBot · Caso de Negocio")

        # Flechas forecasting y clustering → dashboard
        harrow(2.8, 2.0, 1.0)
        harrow(6.3, 8.0, 1.0)

        return fig

    st.plotly_chart(make_arch_fig(), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="slide">
<h4 style="color:#3E1F00;margin-top:0">3.2. Stack Tecnológico</h4>
<table style="width:100%;font-size:0.88em">
<tr style="background:#F5EDE4"><th style="padding:6px 10px">Capa</th><th>Tecnología</th></tr>
<tr><td style="padding:5px 10px">Datos</td><td>Pandas, Parquet, DuckDB</td></tr>
<tr style="background:#F5EDE4"><td>Visualización</td><td>Plotly, Streamlit</td></tr>
<tr><td>ML</td><td>Scikit-learn, Statsmodels</td></tr>
<tr style="background:#F5EDE4"><td>LLM</td><td>Ollama (Llama 3, Mistral, Phi-3)</td></tr>
<tr><td>Backend API</td><td>FastAPI, Pydantic</td></tr>
<tr style="background:#F5EDE4"><td>Almacenamiento</td><td>S3 / GCS / Blob Storage</td></tr>
<tr><td>Orquestación ML</td><td>MLflow / SageMaker</td></tr>
<tr style="background:#F5EDE4"><td>CI/CD</td><td>GitHub Actions</td></tr>
<tr><td>Monitoreo</td><td>CloudWatch / Grafana</td></tr>
</table>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="slide">
<h4 style="color:#3E1F00;margin-top:0">3.3. Patrones de Diseño</h4>
<p><strong>RAG</strong> (Retrieval-Augmented Generation): los datos del dataset se resumen en un
contexto estructurado que se inyecta en el system prompt del LLM. Con 1,650 registros,
la inyección completa es viable. Para escala mayor, se usarían embeddings vectoriales
con ChromaDB, FAISS o pgvector.</p>
<p><strong>Feature Store:</strong> se centralizan las variables computadas (CAGR, crecimiento,
segmento) para reutilización consistente entre modelos y servicios.</p>
<p><strong>Model Registry:</strong> MLflow registra cada versión de modelo con sus métricas,
permitiendo rollback y comparación A/B.</p>
<p><strong>Diseño API-First:</strong> FastAPI expone predicciones y clusters como endpoints REST,
desacoplando el frontend del motor de inferencia.</p>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4: CALCULADORA CLOUD
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.header("4. Arquitectura Cloud — AWS vs GCP vs Azure")
    st.markdown(
        "Mapeo de cada componente de la plataforma a los servicios concretos de cada proveedor, "
        "con estimación de costos mensuales segun parametros de uso."
    )

    # ── 4.1 Tabla de componentes por proveedor ──────────────────
    st.subheader("4.1. Servicios requeridos por componente")

    COMPONENTS = [
        {
            "Componente":   "Almacenamiento de datos",
            "Descripción":  "Dataset Parquet, reportes HTML, artefactos de modelos",
            "AWS":          "Amazon S3 Standard\n($0.023 / GB-mes)",
            "GCP":          "Cloud Storage Standard\n($0.020 / GB-mes)",
            "Azure":        "Azure Blob Storage (LRS)\n($0.018 / GB-mes)",
        },
        {
            "Componente":   "Aplicacion web / Dashboard",
            "Descripción":  "Streamlit multimodulo en contenedor",
            "AWS":          "AWS Fargate (ECS)\nt3.medium equiv. — $0.04/vCPU-hr",
            "GCP":          "Cloud Run\n$0.00002400 / vCPU-s, escala a cero",
            "Azure":        "Azure Container Instances\nB2s — $0.052/hr",
        },
        {
            "Componente":   "API de prediccion ML",
            "Descripción":  "FastAPI que expone forecasting y clustering",
            "AWS":          "AWS Elastic Beanstalk (EC2 t3.small)\no Lambda + API Gateway",
            "GCP":          "Cloud Run (mismo contenedor)\nAutoescalado hasta 0 instancias",
            "Azure":        "Azure App Service (B1)\no Azure Functions Premium",
        },
        {
            "Componente":   "Entrenamiento y reentrenamiento ML",
            "Descripción":  "Pipeline sklearn: LinearRegression, KMeans, PCA",
            "AWS":          "Amazon SageMaker Training\nml.m5.large — $0.192/hr",
            "GCP":          "Vertex AI Training\nn1-standard-4 — $0.180/hr",
            "Azure":        "Azure Machine Learning Compute\nDS3 v2 — $0.200/hr",
        },
        {
            "Componente":   "Registro de modelos",
            "Descripción":  "Versionado, metricas y rollback de modelos ML",
            "AWS":          "SageMaker Model Registry\n(incluido en SageMaker Studio)",
            "GCP":          "Vertex AI Model Registry\n(incluido en Vertex AI)",
            "Azure":        "Azure ML Model Registry\n(incluido en Azure ML workspace)",
        },
        {
            "Componente":   "LLM / Chatbot",
            "Descripción":  "Ollama con Llama 3, Mistral o Phi-3 (codigo abierto)",
            "AWS":          "EC2 g4dn.xlarge (GPU)\n$0.526/hr — o Amazon Bedrock si se migra a LLM gestionado",
            "GCP":          "Cloud Run con GPU T4\no Vertex AI Generative AI (Gemini)",
            "Azure":        "ACI con GPU (NC6 v3)\no Azure OpenAI Service si se migra a LLM gestionado",
        },
        {
            "Componente":   "Base de datos / Feature store",
            "Descripción":  "Almacén de features computados (CAGR, segmentos, tendencias)",
            "AWS":          "Amazon RDS PostgreSQL (db.t3.micro)\no DynamoDB on-demand",
            "GCP":          "Cloud SQL PostgreSQL\no BigQuery (consultas analiticas)",
            "Azure":        "Azure Database for PostgreSQL\no Azure Cosmos DB",
        },
        {
            "Componente":   "Monitoreo y alertas",
            "Descripción":  "Logs de aplicacion, drift de modelos, latencia",
            "AWS":          "Amazon CloudWatch\nLogs + Metrics + Alarms",
            "GCP":          "Cloud Monitoring + Cloud Logging\nError Reporting incluido",
            "Azure":        "Azure Monitor + Application Insights\nLog Analytics Workspace",
        },
        {
            "Componente":   "Red y entrega de contenido",
            "Descripción":  "Egreso de datos, CDN para reportes HTML",
            "AWS":          "Amazon CloudFront (CDN)\n+ VPC Data Transfer $0.09/GB",
            "GCP":          "Cloud CDN\n+ Egress $0.08/GB",
            "Azure":        "Azure CDN (Standard Verizon)\n+ Bandwidth $0.087/GB",
        },
        {
            "Componente":   "CI/CD y control de versiones",
            "Descripción":  "Automatización de despliegue y reentrenamiento",
            "AWS":          "AWS CodePipeline + CodeBuild\no GitHub Actions + ECR",
            "GCP":          "Cloud Build\n+ Artifact Registry",
            "Azure":        "Azure DevOps Pipelines\no GitHub Actions + ACR",
        },
    ]

    df_comp = pd.DataFrame(COMPONENTS)

    st.markdown("""
<style>
.comp-table { width:100%; border-collapse:collapse; font-size:0.82em; margin-top:10px; }
.comp-table th { background:#3E1F00; color:white !important; padding:9px 12px; text-align:left; }
.comp-table td { padding:8px 12px; border-bottom:1px solid #f0e8e0; vertical-align:top; }
.comp-table tr:nth-child(even) td { background:#FAF6F1; }
.comp-table tr:hover td { background:#F5EDE4; }
.badge-aws   { background:#FF9900; color:white; border-radius:4px; padding:1px 7px; font-size:0.78em; }
.badge-gcp   { background:#4285F4; color:white; border-radius:4px; padding:1px 7px; font-size:0.78em; }
.badge-azure { background:#0078D4; color:white; border-radius:4px; padding:1px 7px; font-size:0.78em; }
</style>
""", unsafe_allow_html=True)

    rows_html = ""
    for r in COMPONENTS:
        rows_html += f"""
<tr>
  <td><strong>{r['Componente']}</strong><br><span style="color:#777;font-size:0.9em">{r['Descripción']}</span></td>
  <td>{r['AWS'].replace(chr(10),'<br>')}</td>
  <td>{r['GCP'].replace(chr(10),'<br>')}</td>
  <td>{r['Azure'].replace(chr(10),'<br>')}</td>
</tr>"""

    st.markdown(f"""
<table class="comp-table">
<thead>
<tr>
  <th>Componente / Descripción</th>
  <th><span class="badge-aws">AWS</span> &nbsp;Servicio</th>
  <th><span class="badge-gcp">GCP</span> &nbsp;Servicio</th>
  <th><span class="badge-azure">Azure</span> &nbsp;Servicio</th>
</tr>
</thead>
<tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── 4.2 Calculadora de costos ───────────────────────────────
    st.subheader("4.2. Calculadora de costos mensual")
    st.markdown("Ajustar los parámetros de uso para estimar el costo mensual de operación.")

    p1, p2, p3 = st.columns(3)
    with p1:
        queries_day    = st.slider("Consultas al chatbot por día", 10, 5000, 200, step=10)
        users          = st.slider("Usuarios concurrentes máximos", 1, 100, 10)
    with p2:
        storage_gb     = st.slider("Almacenamiento de datos (GB)", 1, 1000, 5)
        training_freq  = st.selectbox("Frecuencia de reentrenamiento:", ["Mensual", "Semanal", "Diario"])
    with p3:
        avg_tokens     = st.slider("Tokens promedio por consulta", 500, 8000, 2000, step=100)
        dashboard_hrs  = st.slider("Horas de uso del dashboard por mes", 10, 720, 160)

    training_hours   = {"Mensual": 2, "Semanal": 8, "Diario": 60}[training_freq]
    monthly_tokens_M = queries_day * 30 * avg_tokens / 1_000_000

    COST_CATS = [
        "S3 / GCS / Blob (almacenamiento)",
        "Fargate / Cloud Run / ACI (computo)",
        "SageMaker / Vertex AI / Azure ML (entrenamiento)",
        "EC2 GPU / Cloud Run GPU / ACI GPU (LLM)",
        "CloudFront / Cloud CDN / Azure CDN (red)",
        "CloudWatch / Cloud Monitoring / Azure Monitor",
    ]

    def compute_costs(provider):
        if provider == "AWS":
            vals = [
                storage_gb * 0.023,
                dashboard_hrs * 0.0416,
                training_hours * 0.192,
                monthly_tokens_M * 1.5 + (queries_day * 30 / 1000) * 0.526,
                0.09 * max(storage_gb * 0.01, 0.1),
                10.0,
            ]
        elif provider == "GCP":
            vals = [
                storage_gb * 0.020,
                dashboard_hrs * 0.0475,
                training_hours * 0.180,
                monthly_tokens_M * 1.4 + (queries_day * 30 / 1000) * 0.45,
                0.08 * max(storage_gb * 0.01, 0.1),
                8.0,
            ]
        else:
            vals = [
                storage_gb * 0.018,
                dashboard_hrs * 0.0520,
                training_hours * 0.200,
                monthly_tokens_M * 1.6 + (queries_day * 30 / 1000) * 0.60,
                0.087 * max(storage_gb * 0.01, 0.1),
                9.0,
            ]
        return dict(zip(COST_CATS, vals))

    aws   = compute_costs("AWS")
    gcp   = compute_costs("GCP")
    azure = compute_costs("Azure")
    t_aws, t_gcp, t_azure = sum(aws.values()), sum(gcp.values()), sum(azure.values())

    cc1, cc2, cc3 = st.columns(3)
    for col, provider, items, total, color, subtitle in [
        (cc1, "Amazon Web Services",   aws,   t_aws,   "#FF9900", "S3 · Fargate · SageMaker · EC2 · CloudFront · CloudWatch"),
        (cc2, "Google Cloud Platform", gcp,   t_gcp,   "#4285F4", "GCS · Cloud Run · Vertex AI · Cloud CDN · Cloud Monitoring"),
        (cc3, "Microsoft Azure",       azure, t_azure, "#0078D4", "Blob · ACI · Azure ML · Azure CDN · Azure Monitor"),
    ]:
        with col:
            st.markdown(f"""
<div class="cloud-card" style="border-color:{color}">
<div style="font-size:1.8em;font-weight:700;color:{color}">${total:,.2f}/mes</div>
<div style="color:#555;font-size:0.82em;margin:4px 0 2px 0">{provider}</div>
<div style="color:#999;font-size:0.75em">{subtitle}</div>
</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for item, cost in items.items():
                short = item.split("(")[0].strip()
                st.markdown(f"<small style='color:#555'>{short}: <strong>${cost:.2f}</strong></small><br>", unsafe_allow_html=True)

    st.subheader("4.3. Comparación de costos por componente")
    SHORT_CATS = ["Almacenamiento", "Computo app", "Entrenamiento ML", "LLM / GPU", "Red / CDN", "Monitoreo"]
    fig_cost = go.Figure()
    for p, items, color in [("AWS", aws, "#FF9900"), ("GCP", gcp, "#4285F4"), ("Azure", azure, "#0078D4")]:
        vals = list(items.values())
        fig_cost.add_trace(go.Bar(
            name=p, x=SHORT_CATS, y=vals,
            marker_color=color,
            text=[f"${v:.2f}" for v in vals],
            textposition="outside",
        ))
    fig_cost.update_layout(
        barmode="group",
        title="Desglose de costos mensuales por servicio y proveedor",
        yaxis_title="Costo mensual (USD)", height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    st.subheader("4.4. Proyección anual con crecimiento de uso")
    annual = pd.DataFrame({
        "Mes": list(range(1, 13)),
        "AWS":   [t_aws   * (1 + 0.020 * i) for i in range(12)],
        "GCP":   [t_gcp   * (1 + 0.015 * i) for i in range(12)],
        "Azure": [t_azure * (1 + 0.018 * i) for i in range(12)],
    })
    fig_annual = px.line(
        annual.melt(id_vars="Mes", var_name="Proveedor", value_name="Costo USD"),
        x="Mes", y="Costo USD", color="Proveedor",
        color_discrete_map={"AWS": "#FF9900", "GCP": "#4285F4", "Azure": "#0078D4"},
        title="Proyeccion de costo mensual — crecimiento lineal de uso durante el primer año",
        markers=True,
    )
    st.plotly_chart(fig_annual, use_container_width=True)

    st.subheader("4.5. Análisis de retorno sobre la inversión (ROI)")
    with st.expander("Parámetros de ROI"):
        c_roi1, c_roi2 = st.columns(2)
        with c_roi1:
            saved_hours  = st.slider("Horas de analista ahorradas por mes", 10, 200, 60)
            analyst_rate = st.slider("Costo hora analista (USD)", 20, 150, 50)
            deals_better = st.slider("Contratos ganados gracias al análisis (mes)", 0, 20, 3)
            deal_value   = st.slider("Valor promedio por contrato (USD)", 500, 50000, 5000)
        with c_roi2:
            benefit  = saved_hours * analyst_rate + deals_better * deal_value * 0.05
            cost_min = min(t_aws, t_gcp, t_azure)
            net      = benefit - cost_min
            roi_pct  = (net / cost_min) * 100 if cost_min > 0 else 0
            payback  = cost_min / benefit * 30 if benefit > 0 else 0
            st.metric("Beneficio mensual estimado (USD)", f"${benefit:,.0f}")
            st.metric("Costo mínimo mensual (USD)",       f"${cost_min:,.2f}")
            st.metric("Beneficio neto mensual (USD)",     f"${net:,.0f}")
            st.metric("ROI mensual",                      f"{roi_pct:.0f}%")
            st.metric("Período de recuperación",          f"{payback:.0f} dias")

    st.subheader("4.6. Comparación cualitativa de proveedores")
    comparison = pd.DataFrame({
        "Criterio":            ["LLM gestionado", "ML gestionado", "Despliegue Python",
                                "Precio relativo", "Cobertura LatAm", "Cumplimiento normativo",
                                "Ecosistema de datos", "LLM de codigo abierto (Ollama)"],
        "AWS":                 ["Amazon Bedrock (Titan, Claude)", "SageMaker Studio",
                                "Fargate / Elastic Beanstalk", "Medio",
                                "Buena — region Sao Paulo (sa-east-1)", "SOC2, GDPR, HIPAA",
                                "Amplio — Athena, Glue, Redshift", "EC2 g4dn.xlarge ($0.526/hr)"],
        "GCP":                 ["Vertex AI — Gemini Pro/Flash", "Vertex AI Pipelines",
                                "Cloud Run (serverless)", "Bajo",
                                "Regular — region Chile (southamerica-west1)", "ISO 27001, SOC2",
                                "BigQuery destaca en analitica", "Cloud Run GPU T4 ($0.35/hr)"],
        "Azure":               ["Azure OpenAI Service (GPT-4o)", "Azure Machine Learning",
                                "App Service / Container Instances", "Medio-alto",
                                "Buena — region Brasil Sur (brazilsouth)", "SOC2, GDPR, HIPAA",
                                "Integracion Microsoft 365 / Fabric", "ACI NC6 v3 ($0.60/hr)"],
    })
    st.dataframe(comparison, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 5: ENTREGABLES
# ══════════════════════════════════════════════════════════════════
with tab5:
    st.header("5. Entregables del Proyecto")
    st.markdown("Documentos formales para el cliente High Garden Coffee.")

    c_a, c_b, c_c = st.columns(3)

    with c_a:
        st.subheader("5.1. Detalle de la Solución")
        st.markdown("""
- Plataforma de analítica de IA sobre 55 países y 30 temporadas
- **Análisis Exploratorio** — Pandas + Plotly, 10 vistas interactivas
- **Pronóstico de Demanda** — OLS con selección automática de modelo
- **Segmentación** — K-Means k=4, PCA 2D, 4 perfiles estratégicos
- **CoffeeBot** — LLM local (Ollama) con contexto RAG de 55 países
- **Caso de Negocio** — roadmap, arquitectura cloud, calculadora
- Stack: Python · Scikit-learn · Streamlit · Plotly · Ollama
""")
        st.download_button(
            "Descargar (.txt)",
            data="""DETALLE DE LA SOLUCIÓN — HIGH GARDEN COFFEE

MÓDULOS
- Análisis Exploratorio: 10 visualizaciones, rankings por país y tipo de café.
- Pronóstico de Demanda: OLS/polinómico con selección automática, proyecciones 2020-2025.
- Segmentación: K-Means k=4, PCA, 4 segmentos estratégicos.
- CoffeeBot: LLM local vía Ollama con contexto RAG de 55 países.
- Caso de Negocio: roadmap 8 meses, arquitectura cloud AWS/GCP/Azure, calculadora.

STACK
Python 3.9+ · Pandas · Plotly · Scikit-learn · Streamlit · Ollama

REPOSITORIO
https://github.com/alejandrarchbold/High-Garden-Coffee
""".encode("utf-8"),
            file_name="detalle_solucion_highgarden.txt",
            mime="text/plain",
        )

    with c_b:
        st.subheader("5.2. Informe de Datos")
        st.markdown("""
- **Cobertura:** 55 países · 30 temporadas · 0 nulos
- **Crecimiento global:** +137% en 30 años
- **Concentración:** top 5 países = 64% del consumo
- **Mayor crecimiento:** Vietnam (+1,667%), Filipinas (+352%)
- **Mercados en plateau:** Ecuador, Costa Rica, Panamá
- **Tipo dominante:** Arábica/Robusta (Brasil, Colombia)
- **Segmento gigante:** Brasil, Etiopía, México, Colombia
- **Modelos ML:** R² mediana 0.80 · MAPE mediana 4%
""")
        st.download_button(
            "Descargar (.txt)",
            data="""INFORME DE DATOS — HIGH GARDEN COFFEE

COBERTURA
55 países · 30 temporadas (1990/91–2019/20) · 1,650 registros · 0 nulos

HALLAZGOS
1. Consumo global +137% en 30 años.
2. Top 5 mercados concentran el 64% del consumo total.
3. Vietnam: crecimiento histórico de +1,667%.
4. K-Means identifica 4 perfiles estratégicos de mercado.
5. R² mediana > 0.80 · MAPE mediana < 5%.

RECOMENDACIONES
- Priorizar mercados de alto crecimiento y alto volumen.
- Mercados emergentes de Asia ofrecen la mayor oportunidad.
- Mercados en plateau requieren estrategia de mantenimiento.
""".encode("utf-8"),
            file_name="informe_datos_highgarden.txt",
            mime="text/plain",
        )

    with c_c:
        st.subheader("5.3. Acta de Cierre")
        today = datetime.date.today().strftime("%d/%m/%Y")
        st.markdown(f"""
- Fecha: **{today}**
- Responsable: Alejandra Campo Archbold
- Cliente: High Garden Coffee

**Entregables verificados:**
- ✓ Análisis Exploratorio de Datos
- ✓ Modelo de Pronóstico ML (OLS)
- ✓ Segmentación K-Means + PCA
- ✓ Asistente IA CoffeeBot (Ollama)
- ✓ Caso de Negocio y Calculadora Cloud
- ✓ Repositorio de código (GitHub)
- ✓ README con instalación paso a paso
""")
        acta_text = f"""ACTA DE CIERRE — HIGH GARDEN COFFEE
Fecha: {today}
Responsable: Alejandra Campo Archbold

ENTREGABLES VERIFICADOS
[OK] Análisis Exploratorio de Datos
[OK] Modelo de Pronóstico ML — OLS, proyecciones 2020-2025
[OK] Segmentación K-Means k=4, PCA, 4 segmentos
[OK] CoffeeBot — LLM local vía Ollama con contexto RAG
[OK] Caso de Negocio — roadmap, arquitectura cloud, calculadora
[OK] Repositorio: github.com/alejandrarchbold/High-Garden-Coffee
[OK] README con instalación paso a paso

TECNOLOGÍAS
Python 3.9 · Pandas · NumPy · Plotly · Scikit-learn · Streamlit · Ollama

La plataforma fue entregada en su totalidad y se encuentra disponible
para evaluación en el repositorio indicado.
"""
        st.download_button(
            "Descargar (.txt)",
            data=acta_text.encode("utf-8"),
            file_name="acta_cierre_highgarden.txt",
            mime="text/plain",
        )

    st.markdown("---")
    st.markdown("**Repositorio:** https://github.com/alejandrarchbold/High-Garden-Coffee")
