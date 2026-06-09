# High Garden Coffee — Plataforma de Análisis de Mercado

Plataforma de inteligencia artificial para el análisis del consumo doméstico de café (1990–2020).
Desarrollada en Python con Streamlit, incluye análisis exploratorio de datos, proyecciones
con machine learning, segmentación de mercados y un asistente de lenguaje natural con modelos
de código abierto ejecutados localmente.

---

## Estructura del proyecto

```
High-Garden-Coffee/
│
├── about_this_challenge/
│   └── coffee_db.parquet             # Dataset: 55 países × 30 temporadas
│
├── src/                              # Módulos Python reutilizables
│   ├── data_loader.py                # Carga y preprocesamiento del dataset
│   ├── eda_charts.py                 # Visualizaciones Plotly (EDA)
│   ├── forecasting.py                # Selección automática de modelo por país
│   ├── clustering.py                 # K-Means + PCA + perfiles de segmento
│   └── chatbot.py                    # Contexto RAG + comunicación con Ollama
│
├── pages/                            # Páginas de la app Streamlit
│   ├── 1_EDA.py                      # Análisis exploratorio interactivo
│   ├── 2_Forecasting.py              # Proyecciones 2020–2025 por país
│   ├── 3_Clustering.py               # Segmentación K-Means con PCA
│   ├── 4_Chatbot.py                  # CoffeeBot — asistente LLM local
│   └── 5_Business_Case.py            # Presentación ejecutiva + calculadora cloud
│
├── docs/                             # Documentación técnica de cada módulo
│   ├── 01_eda.md
│   ├── 02_forecasting.md
│   ├── 03_clustering.md
│   ├── 04_chatbot.md
│   └── 05_arquitectura.md
│
├── .streamlit/
│   └── config.toml                   # Configuración del tema visual
│
├── app.py                            # Punto de entrada — navegación principal
├── home_page.py                      # Página de inicio
├── requirements.txt                  # Dependencias Python
└── README.md
```

---

## Requisitos previos

| Requisito | Versión mínima | Notas |
|---|---|---|
| Python | 3.9+ | [python.org](https://www.python.org/downloads/) |
| pip | Incluido con Python | Gestor de paquetes |
| Git | Cualquiera | Para clonar el repositorio |
| Ollama | Última estable | Solo para el módulo CoffeeBot |
| RAM | 8 GB mínimo | 16 GB recomendado con Llama 3 |

Compatible con **macOS**, **Linux** y **Windows**.

---

## Instalación paso a paso

### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/alejandrarchbold/High-Garden-Coffee.git
cd High-Garden-Coffee
```

### Paso 2 — Crear un entorno virtual (recomendado)

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

El prompt mostrará `(venv)` cuando el entorno esté activo.

### Paso 3 — Instalar dependencias Python

```bash
pip install -r requirements.txt
```

Instala: `pandas`, `numpy`, `plotly`, `scikit-learn`, `streamlit`, `requests`.
Tiempo estimado: 1–3 minutos según la conexión.

Verificar instalación:
```bash
python3 -c "import streamlit, pandas, sklearn, plotly; print('Dependencias OK')"
```

### Paso 4 — Instalar Ollama (para el chatbot)

El CoffeeBot usa modelos de lenguaje open-source ejecutados **completamente en local**.
No requiere cuenta ni API key de ningún proveedor externo.

**4.1. Instalar Ollama**

- **macOS / Windows:** descargar desde [ollama.com](https://ollama.com) e instalar.
- **Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**4.2. Iniciar el servidor** (mantener esta terminal abierta)

```bash
ollama serve
```

Debe mostrar: `Listening on 127.0.0.1:11434`

**4.3. Descargar un modelo**

```bash
ollama pull phi3        # Phi-3 Mini   (Microsoft) — 2.3 GB — recomendado para empezar
ollama pull mistral     # Mistral 7B   (Mistral AI) — 4.1 GB — buena calidad
ollama pull llama3      # Llama 3 8B   (Meta AI)    — 4.7 GB — mayor capacidad
ollama pull llama3.1    # Llama 3.1 8B (Meta AI)    — 4.7 GB — ventana de 128K tokens
```

Verificar descarga:
```bash
ollama list
```

### Paso 5 — Ejecutar la aplicación

```bash
streamlit run app.py
```

Streamlit intentará abrir el navegador automáticamente. Si no lo hace, navegar a:
```
http://localhost:8501
```

---

## Módulos de la plataforma

| Módulo | Descripción |
|---|---|
| **Inicio** | Resumen de la plataforma, KPIs y guía de navegación |
| **Análisis Exploratorio** | Tendencias, distribuciones y rankings por país y tipo de café |
| **Pronóstico de Demanda** | Proyecciones 2020–2025 con selección automática de modelo |
| **Segmentación de Mercados** | K-Means k=4, reducción PCA y 4 perfiles estratégicos |
| **CoffeeBot — Asistente IA** | LLM local vía Ollama con contexto RAG del dataset completo |
| **Caso de Negocio** | Roadmap 6 meses, arquitectura cloud AWS/GCP/Azure y calculadora de costos |

Orden de exploración recomendado:
**Inicio → Análisis Exploratorio → Pronóstico → Segmentación → CoffeeBot → Caso de Negocio**

---

## Hallazgos principales

- **55 países** analizados, **30 temporadas** (1990/91–2019/20), **0 valores nulos**
- El consumo global creció **+137%** entre 1990 y 2020
- Los 5 principales mercados concentran el **64%** del consumo global
- Vietnam: crecimiento histórico de **+1,667%** en 30 años
- Pronóstico: R² mediana **0.80**, MAPE mediana **3.4%**
- K-Means identifica **4 segmentos** estratégicos de mercado

---

## Modelos de machine learning

### Pronóstico de demanda (`src/forecasting.py`)

Selección automática entre 4 modelos por país:

| Modelo | Cuándo se activa |
|---|---|
| Lineal OLS (30 años) | Tendencia lineal estable en toda la serie histórica |
| Polinómico grado 2 | R² in-sample < 0.70 y mejora ≥ 5 pp en período reciente |
| Lineal OLS reciente | Mercados en plateau: OLS global sobreestima los últimos años |
| Media estable | Ruptura estructural seguida de estabilización en nuevo nivel |

Métricas evaluadas sobre los **últimos 10 años** (o últimos 5 años para el modelo plano).

### Segmentación K-Means (`src/clustering.py`)

Features: `log_total`, `log_last`, `CAGR`, `CV (volatilidad)`, `momentum`.
Preprocesamiento: `StandardScaler`. Visualización: `PCA 2D`.

| Segmento | Perfil |
|---|---|
| Mercado Gigante Consolidado | Brasil, Etiopía, México — grandes mercados establecidos |
| Mercado Dinámico de Alto Crecimiento | Indonesia, Filipinas, Vietnam — CAGR > 4.5% |
| Mercado Pequeño en Declive | Mercados pequeños con CAGR negativo |
| Sin Consumo Registrado | Países con datos históricos mínimos |

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Datos | Pandas 2.x, NumPy, Parquet (PyArrow) |
| Visualización | Plotly 5.x, Streamlit |
| Machine Learning | Scikit-learn (LinearRegression, KMeans, PCA, PolynomialFeatures) |
| LLM / Chatbot | Ollama — Phi-3, Mistral 7B, Llama 3, Llama 3.1 |
| Arquitectura LLM | RAG con contexto completo inyectado en system prompt |
| Frontend | Streamlit 1.36+ con `st.navigation()` (multipage) |

---

## Preguntas frecuentes

**El chatbot dice que Ollama no está activo**
Ejecutar `ollama serve` en una terminal y mantenerla abierta. Recargar la página del chatbot.

**Error `ModuleNotFoundError` al iniciar**
Verificar que el entorno virtual está activado y ejecutar `pip install -r requirements.txt`.

**La app abre pero no carga datos**
Verificar que `about_this_challenge/coffee_db.parquet` existe en el repositorio clonado.


**El entorno virtual no se activa en Windows (error de permisos)**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**¿Puedo usar la app sin Ollama?**
Sí. Los módulos EDA, Pronóstico, Segmentación y Caso de Negocio funcionan sin Ollama.
Solo el módulo CoffeeBot requiere que Ollama esté activo.

---

## Documentación técnica

Ver carpeta [`docs/`](docs/) para documentación detallada de cada módulo:

- [`docs/01_eda.md`](docs/01_eda.md) — Dataset, visualizaciones y hallazgos del EDA
- [`docs/02_forecasting.md`](docs/02_forecasting.md) — Modelos de pronóstico y métricas
- [`docs/03_clustering.md`](docs/03_clustering.md) — K-Means, features y segmentos
- [`docs/04_chatbot.md`](docs/04_chatbot.md) — Arquitectura RAG y configuración de Ollama
- [`docs/05_arquitectura.md`](docs/05_arquitectura.md) — Flujo de datos y arquitectura general

---

## Autor

Alejandra Campo Archbold
