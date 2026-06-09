# Módulo 3 — Segmentación de Mercados (K-Means)

**Archivo:** `pages/3_Clustering.py`  
**Lógica del modelo:** `src/clustering.py`

---

## Objetivo

Agrupar los 55 países en segmentos estratégicos homogéneos basados en su perfil de
consumo histórico, dinámica de crecimiento y comportamiento reciente.

---

## Features de entrada

| Variable | Fórmula | Justificación |
|---|---|---|
| `log_total` | log(1 + consumo total acumulado) | Tamaño de mercado en escala logarítmica |
| `log_last` | log(1 + consumo 2019/20) | Posición actual del mercado |
| `cagr` | (V_2020/V_1990)^(1/29) − 1 | Dinamismo histórico |
| `cv` | σ / μ del consumo anual | Volatilidad / estabilidad |
| `momentum` | (media_últ5 − media_prim5) / media_prim5 | Impulso reciente vs histórico |

Todas las variables se normalizan con `StandardScaler` antes del entrenamiento.

---

## Selección de k (método del codo)

La inercia (within-cluster SSE) se evalúa para k de 2 a 10.
El codo visual aparece en **k = 4**: la reducción de inercia es sustancial hasta k=4
y marginal para k>4. Se selecciona k=4 por equilibrio entre parsimonia e interpretabilidad.

---

## Asignación de nombres a los segmentos

Los segmentos se reordenan por **máximo `log_total`** (no mediana) para garantizar
que el cluster que contiene a Brasil (el mayor mercado individual) reciba el rango 0
y el nombre "Mercado Gigante Consolidado".

```python
order = (
    result.groupby("cluster")["log_total"]
    .max()
    .sort_values(ascending=False)
    .index
    .tolist()
)
remap = {old: new for new, old in enumerate(order)}
```

---

## Segmentos resultantes (k=4)

| Rango | Nombre | Perfil | Países representativos |
|---|---|---|---|
| 0 | Mercado Gigante Consolidado | Alto volumen, CAGR moderado, larga trayectoria | Brasil, Etiopía, México, Colombia |
| 1 | Mercado Dinámico de Alto Crecimiento | CAGR > 4.5%, momentum muy positivo | Indonesia, Filipinas, Vietnam, Uganda |
| 2 | Mercado Pequeño en Declive | Bajo volumen, CAGR negativo o cercano a 0 | Costa Rica, Ghana, Zimbabwe |
| 3 | Sin Consumo Registrado | Datos históricos mínimos o nulos | Guinea Ecuatorial, Nepal |

---

## Visualizaciones

1. **Método del codo** — inercia vs k para seleccionar k óptimo
2. **Scatter PCA 2D** — cada país como burbuja (tamaño = consumo total), coloreada por segmento
3. **Perfiles por variable** — comparación de `log_total`, `cagr`, `cv`, `momentum` promedio por segmento
4. **Tabla resumen** — países por segmento, consumo total, CAGR y momentum promedio
5. **Detalle por país** — tabla filtrable con todos los indicadores

---

## Implicaciones estratégicas

| Segmento | Estrategia recomendada |
|---|---|
| Gigante Consolidado | Retención y diferenciación de valor. Contratos de largo plazo, certificaciones de calidad. |
| Dinámico de Alto Crecimiento | Penetración agresiva. Alianzas con distribuidores locales, adaptación al perfil regional. |
| Pequeño en Declive | Monitoreo con bajo costo. Evaluar causas estructurales antes de invertir. |
| Sin Consumo Registrado | No invertir en el corto plazo. Reevaluar si se disponen de datos. |

---

## Reducción PCA

```
Z = X × W
W = eigenvectors(Cov(X))   [primeros 2 componentes]
```

Los dos primeros componentes explican la mayor parte de la varianza.
Se usan exclusivamente para visualización — el clustering se realiza en el espacio original de 5 dimensiones.
