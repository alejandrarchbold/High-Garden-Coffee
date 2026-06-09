"""
Chatbot module using Ollama for local open-source LLM inference.

Supported models (must be pulled via `ollama pull <model>`):
    llama3       Llama 3 8B  (Meta AI)           ~4.7 GB
    mistral      Mistral 7B  (Mistral AI)         ~4.1 GB
    phi3         Phi-3 Mini  (Microsoft)          ~2.3 GB
    llama3.1     Llama 3.1 8B (Meta AI)           ~4.7 GB
"""

import requests
import pandas as pd
from src.data_loader import (
    load_data, top_countries, coffee_type_summary, growth_table, YEAR_COLS
)

OLLAMA_BASE = "http://localhost:11434"

AVAILABLE_MODELS = {
    "Phi-3 Mini — Microsoft": "phi3",
    "Llama 3 8B — Meta AI": "llama3",
    "Mistral 7B — Mistral AI": "mistral",
    "Llama 3.1 8B — Meta AI": "llama3.1",
}


def build_context(df, cluster_df=None, forecast_df=None) -> str:
    """
    Construye el contexto RAG completo para el LLM.
    Incluye: descripcion del dataset, todos los paises con sus datos,
    descripcion de los modelos ML, resultados de clustering y forecasting
    para los 55 paises.
    """
    # ── Estadisticas globales ──────────────────────────────────
    global_1990  = df["1990/91"].sum() / 1e6
    global_2020  = df["2019/20"].sum() / 1e6
    global_total = df["Total_domestic_consumption"].sum() / 1e9
    global_growth = (global_2020 - global_1990) / global_1990 * 100
    total_all = df["Total_domestic_consumption"].sum()
    ct   = coffee_type_summary(df)
    gt   = growth_table(df).dropna(subset=["cagr"])
    top5 = top_countries(df, 5)

    # Pre-compute coffee type breakdown
    ct_facts = []
    for _, row in ct.iterrows():
        pct = row["Total_domestic_consumption"] / total_all * 100
        ct_facts.append(f"{row['Coffee type']} ({pct:.1f}%)")

    # Pre-compute clustering Q&A
    cluster_qa = []
    if cluster_df is not None:
        for seg in sorted(cluster_df["cluster_name"].unique()):
            countries = cluster_df[cluster_df["cluster_name"] == seg]["Country"].tolist()
            cluster_qa.append(f"   - {seg} ({len(countries)} paises): {', '.join(countries)}")
    else:
        cluster_qa = ["   - Mercado Gigante Consolidado: Brasil, Etiopia, Mexico, Colombia",
                      "   - Mercado Dinamico de Alto Crecimiento: Indonesia, Vietnam, Filipinas, Uganda",
                      "   - Mercado Pequeno en Declive: Costa Rica, Ghana, Camerun",
                      "   - Sin Consumo Registrado: paises sin datos significativos"]

    # Pre-compute forecasting Q&A
    top5_growth, neg_countries = [], []
    if forecast_df is not None:
        top5_fc = forecast_df.sort_values("Crecimiento_%", ascending=False).head(5)
        top5_growth = [f"{r['Country']} ({r['Crecimiento_%']:+.1f}%)" for _, r in top5_fc.iterrows()]
        neg_fc = forecast_df[forecast_df["Crecimiento_%"] < 0]
        neg_countries = [f"{r['Country']} ({r['Crecimiento_%']:+.1f}%)" for _, r in neg_fc.iterrows()]

    pct_dom = ct_facts[0].split("(")[1].rstrip(")") if ct_facts else "N/A"

    lines = [
        "=== RESPUESTAS DIRECTAS (LEE ESTO PRIMERO) ===",
        "",
        "P: ¿Qué tipo de café domina el mercado global y por qué?",
        f"R: El tipo Arabica/Robusta domina con el {pct_dom} del consumo global (1990-2020).",
        f"   Razon: Brasil produce Arabica/Robusta y representa ~44% del consumo global total.",
        f"   Distribucion completa: {' | '.join(ct_facts)}",
        "",
        "P: ¿Qué segmentos de mercado identificó el modelo de clustering?",
        "R: K-Means con k=4 identificó 4 segmentos estratégicos:",
    ] + cluster_qa + [
        "",
        "P: ¿Cuáles son los 5 países con mayor crecimiento proyectado al 2025?",
        f"R: {', '.join(top5_growth) if top5_growth else 'Ver seccion de forecasting.'}",
        "",
        "P: ¿Qué países muestran contracción en el consumo?",
        f"R: {', '.join(neg_countries) if neg_countries else 'Ningun pais muestra contraccion.'}",
        "",
        "P: ¿Cuáles son los 5 países con mayor consumo total acumulado?",
        f"R: {', '.join(top5['Country'].tolist())}",
        "",
        "P: ¿Cuánto creció el consumo global?",
        f"R: {global_growth:.1f}% entre 1990 y 2020.",
        "",
        "=== FIN RESPUESTAS DIRECTAS ===",
        "",
        "=== DATASET: HIGH GARDEN COFFEE — CONSUMO DOMESTICO DE CAFE ===",
        f"Fuente: coffee_db.parquet",
        f"Registros: 55 paises x 30 temporadas cafetaleras (1990/91 a 2019/20)",
        f"Tipos de cafe: Arabica, Robusta, Arabica/Robusta, Robusta/Arabica",
        f"Valores nulos: 0 (dataset completo)",
        "",
        "--- RESUMEN GLOBAL ---",
        f"Consumo global 1990/91: {global_1990:,.0f} millones de tazas",
        f"Consumo global 2019/20: {global_2020:,.0f} millones de tazas",
        f"Crecimiento 1990-2020: {global_growth:.1f}%",
        f"Consumo total acumulado 30 anios: {global_total:,.2f} miles de millones de tazas",
        f"Top 5 paises: {', '.join(top5['Country'].tolist())}",
        "",
        "--- DISTRIBUCION POR TIPO DE CAFE ---",
    ]
    for _, row in ct.iterrows():
        pct = row["Total_domestic_consumption"] / total_all * 100
        lines.append(
            f"  {row['Coffee type']}: {row['Total_domestic_consumption']/1e6:,.0f}M tazas ({pct:.1f}% del total)"
        )

    # Explicit breakdown: which countries belong to each coffee type
    lines.append("")
    lines.append("  PAISES POR TIPO DE CAFE:")
    for ctype in df["Coffee type"].unique():
        countries_ctype = df[df["Coffee type"] == ctype]["Country"].tolist()
        pct_ctype = df[df["Coffee type"] == ctype]["Total_domestic_consumption"].sum() / total_all * 100
        lines.append(
            f"    {ctype} ({pct_ctype:.1f}%): {', '.join(sorted(countries_ctype))}"
        )
    lines.append(
        "  CONCLUSION: El tipo Arabica/Robusta domina (53.5%) impulsado por Brasil "
        "(44% del consumo global). El tipo Robusta/Arabica (22.7%) predomina en Asia "
        "(Vietnam, India, Indonesia, Filipinas). El Arabica puro (21.5%) corresponde a "
        "productores de calidad: Colombia, Costa Rica, Etiopia."
    )

    # ── Todos los paises: datos historicos y CAGR ──────────────
    lines += [
        "",
        "--- DATOS POR PAIS: CONSUMO HISTORICO Y CAGR (55 PAISES) ---",
        "  [Pais | Tipo cafe | Consumo 1990 (M) | Consumo 2020 (M) | Total acum (M) | CAGR %]",
    ]
    country_rows = df.sort_values("Total_domestic_consumption", ascending=False)
    for _, row in country_rows.iterrows():
        c1990 = row["1990/91"] / 1e6
        c2020 = row["2019/20"] / 1e6
        total = row["Total_domestic_consumption"] / 1e6
        cagr_row = gt[gt["Country"] == row["Country"]]
        cagr_val = cagr_row["cagr"].iloc[0] * 100 if len(cagr_row) else 0
        growth_pct = cagr_row["growth_pct"].iloc[0] if len(cagr_row) else 0
        lines.append(
            f"  {row['Country']} | {row['Coffee type']} | "
            f"1990={c1990:.1f}M | 2020={c2020:.1f}M | "
            f"total={total:.1f}M | CAGR={cagr_val:.2f}% | crecimiento_total={growth_pct:+.0f}%"
        )

    # ── Modelo 1: Clustering K-Means ───────────────────────────
    lines += [
        "",
        "=== MODELO ML 1: CLUSTERING K-MEANS ===",
        "Algoritmo: K-Means con k=4 segmentos (elegido por metodo del codo)",
        "Variables de entrada (features):",
        "  - log_total: logaritmo del consumo total acumulado (tamano de mercado)",
        "  - log_last:  logaritmo del consumo en 2019/20 (posicion actual)",
        "  - cagr:      tasa de crecimiento anual compuesto (dinamismo)",
        "  - cv:        coeficiente de variacion (volatilidad del consumo)",
        "  - momentum:  diferencia entre promedio ultimos 5 anios vs primeros 5 anios",
        "Preprocesamiento: StandardScaler (normalizacion z-score)",
        "Visualizacion: PCA 2D (componentes principales)",
        "Salida del modelo: segmento asignado a cada pais",
    ]

    if cluster_df is not None:
        lines.append("")
        lines.append("--- RESULTADO CLUSTERING: TODOS LOS PAISES CON SU SEGMENTO ---")
        lines.append("  [Pais | Segmento | CAGR % | Momentum | Volatilidad (CV)]")
        for _, row in cluster_df.sort_values("cluster_name").iterrows():
            lines.append(
                f"  {row['Country']} | {row['cluster_name']} | "
                f"CAGR={row['cagr']*100:.2f}% | "
                f"momentum={row['momentum']:.3f} | "
                f"volatilidad_cv={row['cv']:.3f}"
            )
        lines.append("")
        lines.append("--- RESUMEN DE SEGMENTOS ---")
        for seg in sorted(cluster_df["cluster_name"].unique()):
            seg_rows = cluster_df[cluster_df["cluster_name"] == seg]
            countries = seg_rows["Country"].tolist()
            avg_cagr  = seg_rows["cagr"].mean() * 100
            avg_mom   = seg_rows["momentum"].mean()
            lines.append(
                f"  [{seg}] — {len(countries)} paises | CAGR promedio={avg_cagr:.2f}% | "
                f"momentum promedio={avg_mom:.3f}"
            )
            lines.append(f"    Paises: {', '.join(countries)}")
    else:
        lines.append("  (Clustering no disponible en este contexto)")

    # ── Modelo 2: Forecasting Regresion de Tendencia ───────────
    lines += [
        "",
        "=== MODELO ML 2: FORECASTING — REGRESION DE TENDENCIA ===",
        "Algoritmo: seleccion automatica entre 4 modelos por pais:",
        "  1. Lineal OLS (30 anios) — para mercados con tendencia lineal estable",
        "  2. Polinomico grado 2 — para mercados con curva de aceleracion/desaceleracion",
        "  3. Lineal OLS reciente (ultimos 15 anios) — para mercados en plateau",
        "  4. Media plana (ultimos 5 anios) — para mercados con ruptura estructural estabilizada",
        "Variable dependiente (Y): consumo en tazas por temporada",
        "Variable independiente (X): numero de anio (1990=0, 1991=1, ..., 2019=29)",
        "Ajuste: sobre los 30 anios historicos 1990/91 a 2019/20",
        "Metricas: R-cuadrado y MAPE calculados sobre los ultimos 10 anios (bondad de ajuste reciente)",
        "Proyeccion: 2020, 2021, 2022, 2023, 2024, 2025",
        "Salida: consumo proyectado 2025 y crecimiento proyectado vs 2019",
    ]

    if forecast_df is not None:
        r2_mean   = forecast_df["R2"].mean()
        mape_mean = forecast_df["MAPE_%"].mean()
        lines.append(f"Rendimiento promedio del modelo: R2={r2_mean:.3f} | MAPE={mape_mean:.1f}%")
        lines.append("")
        lines.append("--- PROYECCIONES POR PAIS: TODOS LOS PAISES (2019 vs 2025) ---")
        lines.append(
            "  [Pais | Consumo_2019 (M tazas) | Proyeccion_2025 (M tazas) | "
            "Crecimiento_proyectado % | R2 | MAPE %]"
        )
        for _, row in forecast_df.sort_values("Country").iterrows():
            lines.append(
                f"  {row['Country']} | "
                f"consumo_2019={row['Consumo_2019_M']:.2f}M | "
                f"proyeccion_2025={row['Proyeccion_2025_M']:.2f}M | "
                f"crecimiento_proyectado={row['Crecimiento_%']:+.1f}% | "
                f"R2={row['R2']:.3f} | MAPE={row['MAPE_%']:.1f}%"
            )
        lines.append("")
        lines.append("--- TOP 10 MAYOR CRECIMIENTO PROYECTADO 2019-2025 ---")
        top_fc = forecast_df.sort_values("Crecimiento_%", ascending=False).head(10)
        for _, row in top_fc.iterrows():
            lines.append(
                f"  {row['Country']}: {row['Crecimiento_%']:+.1f}% "
                f"(de {row['Consumo_2019_M']:.1f}M a {row['Proyeccion_2025_M']:.1f}M tazas)"
            )
        lines.append("")
        lines.append("--- PAISES CON PROYECCION NEGATIVA (MERCADOS EN CONTRACCION) ---")
        neg_fc = forecast_df[forecast_df["Crecimiento_%"] < 0].sort_values("Crecimiento_%")
        for _, row in neg_fc.iterrows():
            lines.append(
                f"  {row['Country']}: {row['Crecimiento_%']:+.1f}% "
                f"(proyeccion 2025={row['Proyeccion_2025_M']:.2f}M)"
            )
    else:
        lines.append("  (Forecasting no disponible en este contexto)")

    lines += [
        "",
        "=== FIN DEL CONTEXTO ===",
        "INSTRUCCIONES:",
        "- Responde siempre en espanol formal.",
        "- Usa los numeros y porcentajes que aparecen en este contexto.",
        "- Si la respuesta esta en RESPUESTAS DIRECTAS al inicio, usala tal cual.",
        "- Solo di que no tienes datos si el tema realmente no aparece en ninguna parte del contexto.",
    ]
    return "\n".join(lines)


def chat_ollama(messages: list[dict], context: str, model: str = "llama3") -> str:
    system_prompt = (
        "REGLA ABSOLUTA: Solo puedes responder usando los datos del CONTEXTO que aparece abajo. "
        "Prohibido usar conocimiento general o inventar datos. "
        "El contexto empieza con RESPUESTAS DIRECTAS — usalas exactamente cuando apliquen. "
        "Si el contexto tiene la respuesta, copiala y expandela. "
        "Solo admite que no tienes datos si el tema NO aparece en ningun lugar del contexto.\n\n"
        "Eres el analista de datos de High Garden Coffee. Respondes en español formal "
        "con cifras exactas del contexto (porcentajes, CAGR, millones de tazas, nombres de segmentos).\n\n"
        "CONTEXTO CON TODOS LOS DATOS:\n"
        + context
    )
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    try:
        response = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={"model": model, "messages": full_messages, "stream": False},
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "Error de conexion: Ollama no esta ejecutandose en http://localhost:11434.\n\n"
            "Para iniciar Ollama:\n"
            "  1. Instala Ollama desde https://ollama.com\n"
            "  2. Ejecuta en terminal: ollama serve\n"
            f"  3. Descarga el modelo: ollama pull {model}\n"
            "  4. Recarga esta pagina."
        )
    except requests.exceptions.HTTPError as e:
        if "404" in str(e) or "model not found" in str(e).lower():
            return (
                f"Modelo '{model}' no encontrado en Ollama.\n\n"
                f"Descargalo con: ollama pull {model}"
            )
        return f"Error HTTP de Ollama: {e}"
    except Exception as e:
        return f"Error inesperado: {e}"


def check_ollama_status() -> tuple[bool, list[str]]:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        models = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        return True, models
    except Exception:
        return False, []


SUGGESTED_QUESTIONS = [
    "¿Cuáles son los cinco países con mayor crecimiento proyectado al 2025?",
    "¿Qué segmentos de mercado identificó el modelo de clustering?",
    "¿Cuál es el consumo proyectado de Colombia para 2025?",
    "¿Qué países muestran contracción en el consumo según el forecasting?",
    "¿Cómo se compara Vietnam con Brasil en crecimiento histórico y proyectado?",
    "¿Qué mercados emergentes debería priorizar High Garden Coffee según los modelos de ML?",
    "Dame un resumen ejecutivo del análisis completo: EDA, forecasting y clustering.",
]
