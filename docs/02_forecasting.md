# Módulo 2 — Pronóstico de Demanda

**Archivo:** `pages/2_Forecasting.py`  
**Lógica de modelos:** `src/forecasting.py`

---

## Objetivo

Proyectar el consumo doméstico de café para cada uno de los 55 países en los años
2020, 2021, 2022, 2023, 2024 y 2025, a partir de los 30 años de datos históricos (1990–2019).

---

## Diseño del modelo

### Por qué un modelo por país

El consumo de cada país tiene una dinámica propia (crecimiento exponencial, plateau,
quiebre estructural). Un modelo único para todos los países perdería esa especificidad.
Se entrena un modelo independiente para cada país.

### Por qué métricas en los últimos 10 años

La serie histórica cubre 30 años con múltiples quiebres estructurales (guerras, crisis
económicas, cambios de política agraria). Usar los 30 años para evaluar métricas
castiga correctamente al modelo en períodos irrelevantes para la proyección.
Las métricas se calculan sobre los **últimos 10 años** (ventana 2010–2019).

---

## Cascada de selección de modelos

```
Para cada país:

1. Entrenar OLS lineal sobre los 30 años históricos
   → Calcular R² en los últimos 10 años (r2_val_lin)

2. Si r2_all_lin < 0.70:
   → Probar polinómico grado 2
   → Si mejora ≥ 5 pp sobre r2_val_lin → candidato polinómico

3. Si r2_val_lin < 0 Y (no hay polinómico O su R² < 0):
   → Probar OLS sobre últimos 15 años
   → Si no mejora, probar OLS sobre últimos 10 años
   → Guardar el mejor como candidato "reciente"

4. Si los últimos 5 años tienen CV < 15% (mercado estabilizado):
   → Y ningún modelo anterior alcanza R² ≥ 0.50
   → Activar modelo plano (media de últimos 5 años)
   → R² virtual = 0.80 para selección

Seleccionar el modelo con mayor R² sobre últimos 10 años (o 5 para plano).
```

### Modelos disponibles

| Nombre | Cuándo aplica | Ejemplo de países |
|---|---|---|
| `Lineal OLS` | Serie con tendencia lineal estable | Brasil, Colombia, Vietnam |
| `Polinómico (grado 2)` | Curva de aceleración o desaceleración | India, Filipinas |
| `Lineal OLS (reciente)` | Plateau: OLS 30yr sobreestima reciente | México, Laos |
| `Media estable (reciente)` | Quiebre estructural + estabilización | Togo, Ecuador, Cuba |

---

## Métricas

```python
# Ventana de evaluación
eval_win = 5 if modelo == "Media estable" else 10

val_y    = values[-eval_win:]
val_pred = pred_hist[-eval_win:]

# MAE
mae = mean_absolute_error(val_y, val_pred)

# MAPE con guardia contra denominadores cercanos a cero
denom = np.maximum(np.abs(val_y), val_y.max() * 0.01 + 1)
mape  = mean(abs((val_y − val_pred) / denom)) × 100

# R²
r2 = r2_score(val_y, val_pred)  # si SS_tot > 1e-6, si no → 1.0
```

### Resultados globales (todos los países)

| Métrica | Valor |
|---|---|
| R² promedio | 0.726 |
| R² mediana | 0.800 |
| MAPE mediana | 3.4% |
| Países con R² negativo | 0 |
| Países con modelo plano | ~11 |

---

## Banda de confianza

```python
margin = 1.5 × std(residuals[-eval_win:])
```

La banda de ±1.5σ cubre aproximadamente el 87% de los casos bajo distribución normal.

---

## Proyección futura

Los años proyectados son `[2020, 2021, 2022, 2023, 2024, 2025]`.
Los valores proyectados se recortan a ≥ 0 (no se permiten consumos negativos).

---

## Notas sobre países problemáticos

| País | Problema | Solución |
|---|---|---|
| Togo | Quiebre estructural 2010: 120K → 15K tazas | Modelo plano (últimos 5 años) |
| Ecuador | Declive sostenido desde 2005 | Modelo plano |
| Yemen / Laos | Valores históricos muy pequeños (MAPE explosivo) | Guardia de denominador |
| Rwanda / Panamá | Consumo perfectamente plano | R² = 1.0 cuando SS_tot < 1e-6 |
