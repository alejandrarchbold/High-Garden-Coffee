# Módulo 4 — CoffeeBot (Asistente IA)

**Archivo:** `pages/4_Chatbot.py`  
**Lógica de contexto y API:** `src/chatbot.py`

---

## Objetivo

Permitir al usuario hacer preguntas en lenguaje natural sobre el dataset de consumo de
café y recibir respuestas contextualizadas, sin necesidad de saber Python o SQL.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit (frontend)                 │
│  Entrada de texto → historial de conversación → salida  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              src/chatbot.py  (contexto RAG)             │
│  build_context(df) → ~6,000 tokens de resumen del       │
│  dataset inyectados como system prompt en cada llamada  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Ollama REST API (local)                    │
│  POST http://localhost:11434/api/chat                   │
│  Modelo seleccionable: phi3 / mistral / llama3 / etc.   │
└─────────────────────────────────────────────────────────┘
```

---

## RAG con contexto completo

Con solo 1,650 registros (55 países × 30 temporadas), el dataset completo cabe en un
prompt único (≈6,000 tokens). No se usa recuperación vectorial ni embeddings.

### Contenido del system prompt (`build_context`)

```
CONTEXT: Coffee consumption dataset 1990-2020
- 55 países, 30 temporadas
- Países por tipo de café (Arabica/Robusta, etc.)
- Top 10 países por consumo total
- CAGR y crecimiento total por país
- Estadísticas globales (media, mediana, CV)
- Pronósticos 2025 por país (modelo + métricas)
- Segmentación K-Means: segmento de cada país
- PAÍSES POR TIPO DE CAFÉ: lista explícita por tipo
- CONCLUSIÓN: tipo dominante y razón
```

### Por qué contexto completo y no embeddings

Los embeddings son útiles cuando el corpus es demasiado grande para caber en un
prompt (millones de tokens). Con 1,650 registros, inyectar el contexto completo
garantiza que el LLM tiene acceso a toda la información sin riesgo de
recuperación parcial.

---

## Comunicación con Ollama

```python
payload = {
    "model": selected_model,
    "messages": [
        {"role": "system", "content": context},
        *conversation_history,
        {"role": "user", "content": user_input},
    ],
    "stream": False,
}
response = requests.post(
    "http://localhost:11434/api/chat",
    json=payload,
    timeout=90,
)
```

- `stream: False` — respuesta completa en un solo JSON (no streaming)
- `timeout: 90` — modelos pequeños (phi3) responden en 5–30 segundos

---

## Modelos soportados

| Modelo | Tamaño | Calidad recomendada | Descargar |
|---|---|---|---|
| `phi3` | 2.3 GB | Inicio rápido | `ollama pull phi3` |
| `mistral` | 4.1 GB | Buena calidad | `ollama pull mistral` |
| `llama3` | 4.7 GB | Alta capacidad | `ollama pull llama3` |
| `llama3.1` | 4.7 GB | Mejor precisión factual | `ollama pull llama3.1` |

El modelo activo se selecciona desde el sidebar. Solo aparecen los modelos
que ya están descargados en el sistema (consultados vía `ollama list`).

---

## Historial de conversación

El historial se mantiene en `st.session_state.messages`. Cada turno agrega:

```python
{"role": "user",      "content": user_input}
{"role": "assistant", "content": response_text}
```

El botón "Limpiar conversación" reinicia `st.session_state.messages = []`.

---

## Caché del contexto

`build_context(df)` está decorado con `@st.cache_data`. La caché se invalida
automáticamente si el código fuente de la función cambia. También hay un botón
"Recargar contexto ML" que ejecuta `st.cache_data.clear()` para forzar la
regeneración del contexto sin reiniciar la app.

---

## Limitaciones conocidas

| Limitación | Causa | Mitigación |
|---|---|---|
| Bloqueo durante respuesta | Streamlit es síncrono (no async) | Usar modelos más rápidos (phi3) |
| Alucinaciones de clusters | LLM puede inventar k=5 si no fue instruidoÂ | Contexto incluye k=4 explícitamente |
| Respuestas lentas con Llama 3 | Modelo grande en CPU | Aceptable en hardware con GPU |
| No recuerda sesiones anteriores | Sin persistencia de historial | El historial existe solo en la sesión actual |

---

## Preguntas de ejemplo que CoffeeBot puede responder

- ¿Qué tipo de café domina el mercado global y por qué?
- ¿Cuáles son los países con mayor crecimiento proyectado al 2025?
- ¿En qué segmento K-Means está Indonesia?
- ¿Qué modelo de pronóstico se usó para Togo y por qué?
- ¿Cuántos países tienen consumo decreciente?
