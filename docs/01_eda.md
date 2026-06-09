# Módulo 1 — Análisis Exploratorio de Datos (EDA)

**Archivo:** `pages/1_EDA.py`  
**Dependencias:** `src/data_loader.py`, `src/eda_charts.py`

---

## Dataset

| Atributo | Valor |
|---|---|
| Archivo | `about_this_challenge/coffee_db.parquet` |
| Filas | 55 (una por país) |
| Columnas | 34 (Country, Coffee type, 30 temporadas, Total) |
| Rango temporal | 1990/91 – 2019/20 (30 temporadas cafetaleras) |
| Valores nulos | 0 |
| Tipos de café | Arabica, Robusta, Arabica/Robusta, Robusta/Arabica |
| Unidad | Tazas absolutas (entero) |

### Tipos de café por país

| Tipo | Participación | Países representativos |
|---|---|---|
| Arabica/Robusta | 53.5% | Brasil, México, Guatemala |
| Robusta/Arabica | 22.7% | Vietnam, India, Indonesia, Filipinas |
| Arabica | 21.5% | Colombia, Costa Rica, Etiopía, Kenia |
| Robusta | 2.3% | Congo, Costa de Marfil, Nigeria |

---

## Visualizaciones incluidas

| Sección | Gráfico | Descripción |
|---|---|---|
| 3 | Barras horizontales | Top N países por consumo total acumulado |
| 4 | Pie + Barras | Participación de mercado por tipo de café |
| 5 | Serie de tiempo | Tendencia global 1990–2020 |
| 5 | Líneas multi-país | Evolución del top 10 por país |
| 6 | Barras horizontales | Rankings por tasa de crecimiento total |
| 6 | Tabla | Mercados en declive (menor crecimiento) |
| 7 | Burbujas | Tamaño de mercado (log) vs CAGR |
| 8 | Boxplot | Distribución del consumo por tipo de café |
| 9 | Mapa de calor | Consumo anual top 15 países (1990–2020) |
| 10 | Área apilada | Participación relativa de mercado top 5 + resto |

---

## Cálculos clave (`src/data_loader.py`)

### CAGR (Tasa de Crecimiento Anual Compuesto)

```
CAGR = (V_2020 / V_1990)^(1/29) − 1
```

Implementación: `growth_table(df)` — excluye países con consumo inicial cero.

### Crecimiento total

```
g_total = (V_2020 − V_1990) / V_1990 × 100
```

---

## Hallazgos principales

1. **Distribución asimétrica**: la media supera ampliamente la mediana en todos los años (coeficiente de variación > 200%). Unos pocos mercados grandes elevan el promedio global.
2. **Concentración**: top 5 países concentran el ~64% del consumo total acumulado.
3. **Crecimiento global**: +137% entre 1990 y 2020; aceleración notable a partir de 2000 (mercados emergentes en Asia y África).
4. **Líderes por CAGR**: Tanzania (11.5%), Vietnam (10.4%), Tailandia (7.2%).
5. **Mercados en plateau/declive**: Ecuador (CAGR −0.1%), Costa Rica (−0.2%), Camerún, Ghana.
6. **Indonesia** (CAGR 4.8%, +287% total): mercado de alto crecimiento frecuentemente subestimado.

---

## Descarga de datos

Al final de la página hay un botón para descargar las estadísticas descriptivas como CSV.
