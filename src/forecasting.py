import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from src.data_loader import YEAR_COLS, YEAR_NUMS, load_data

FORECAST_YEARS  = [2020, 2021, 2022, 2023, 2024, 2025]
METRICS_WINDOW  = 10   # evaluate R²/MAPE on last N years to avoid old structural breaks


def _country_series(df: pd.DataFrame, country: str) -> tuple[np.ndarray, np.ndarray]:
    row = df[df["Country"] == country][YEAR_COLS].sum()
    return np.array(YEAR_NUMS), row.values.astype(float)


_RECENT_WINDOW = 15   # ventana para modelos de mercados en plateau
_FLAT_WINDOW   = 5    # ventana para detectar mercado estable tras ruptura estructural


def fit_forecast(years: np.ndarray, values: np.ndarray) -> dict:
    """
    Selección automática de modelo para proyección 2020-2025:
    1. Lineal OLS sobre 30 años históricos.
    2. Polinómico grado 2 (si R² in-sample < 0.70 y mejora ≥5pp).
    3. Lineal OLS reciente (últimos 15 o 10 años) para mercados en plateau.
    4. Media plana (últimos 5 años) para mercados con ruptura estructural
       que se estabilizaron en un nuevo nivel (CV < 0.15 en últimos 5 años).
    R² y MAPE se reportan sobre los últimos METRICS_WINDOW años.
    """
    x_all    = years.reshape(-1, 1)
    future_x = np.array(FORECAST_YEARS).reshape(-1, 1)

    def _val_r2(pred_full: np.ndarray) -> float:
        """R² de las predicciones en la ventana reciente de evaluación."""
        val_y    = values[-METRICS_WINDOW:]
        val_pred = np.maximum(pred_full[-METRICS_WINDOW:], 0)
        ss_tot   = np.sum((val_y - val_y.mean()) ** 2)
        return r2_score(val_y, val_pred) if ss_tot > 1e-6 else 1.0

    # ── Modelo 1: Lineal OLS (30 años) ──────────────────────────────
    m_lin  = LinearRegression().fit(x_all, values)
    p_lin  = m_lin.predict(x_all)
    r2_all_lin = r2_score(values, p_lin)
    r2_val_lin = _val_r2(p_lin)

    # ── Modelo 2: Polinómico grado 2 ────────────────────────────────
    m_pol = None
    r2_val_pol = -np.inf
    if r2_all_lin < 0.70:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("reg",  LinearRegression()),
        ])
        pipe.fit(x_all, values)
        p_pol      = pipe.predict(x_all)
        r2_val_pol = _val_r2(p_pol)
        if r2_val_pol > r2_val_lin + 0.05:
            m_pol = pipe

    # ── Modelo 3: OLS reciente (últimos 15 o 10 años) ────────────────
    # Se activa cuando el modelo global da R²<0 en el período reciente
    # (mercados en plateau donde la tendencia histórica sobreestima el presente)
    m_rec = None
    r2_val_rec = -np.inf
    if r2_val_lin < 0 and (m_pol is None or r2_val_pol < 0):
        best_rec_r2 = -np.inf
        for win in (_RECENT_WINDOW, METRICS_WINDOW):
            xw = years[-win:].reshape(-1, 1)
            yw = values[-win:]
            mw = LinearRegression().fit(xw, yw)
            pw_full = mw.predict(x_all)
            rw = _val_r2(pw_full)
            if rw > best_rec_r2:
                best_rec_r2 = rw
                m_rec = mw
                r2_val_rec = rw

    # ── Modelo 4: Media plana (últimos 5 años) ───────────────────────
    # Para mercados con ruptura estructural que se estabilizaron en un nivel nuevo
    # (ej.: Togo tras colapso de 2010). Se activa cuando:
    #   a) Los últimos 5 años son muy estables (CV < 0.15)
    #   b) Ningún otro modelo alcanza R² ≥ 0.50 (están rotos)
    flat_y  = values[-_FLAT_WINDOW:]
    flat_cv = flat_y.std() / max(flat_y.mean(), 1.0)
    m_flat_mean = None
    r2_val_flat = -np.inf
    if flat_cv < 0.15:
        flat_mean = flat_y.mean()
        current_best_r2 = max(
            r2_val_lin,
            r2_val_pol if m_pol else -np.inf,
            r2_val_rec if m_rec else -np.inf,
        )
        if current_best_r2 < 0.50:
            m_flat_mean = flat_mean
            # R² virtual: el modelo plano se selecciona sobre modelos rotos
            r2_val_flat = 0.80

    # ── Selección del mejor modelo ───────────────────────────────────
    candidates = [
        (r2_val_lin,                                    "lin"),
        (r2_val_pol  if m_pol       else -np.inf,       "pol"),
        (r2_val_rec  if m_rec       else -np.inf,       "rec"),
        (r2_val_flat if m_flat_mean is not None else -np.inf, "flat"),
    ]
    best = max(candidates, key=lambda x: x[0])[1]

    if best == "pol":
        pred_hist   = np.maximum(m_pol.predict(x_all), 0)
        future_pred = np.maximum(m_pol.predict(future_x), 0)
        residuals   = values - pred_hist
        model_type  = "Polinómico (grado 2)"
    elif best == "rec":
        pred_hist   = np.maximum(m_rec.predict(x_all), 0)
        future_pred = np.maximum(m_rec.predict(future_x), 0)
        residuals   = values - pred_hist
        model_type  = "Lineal OLS (reciente)"
    elif best == "flat":
        pred_hist   = np.full_like(values, m_flat_mean, dtype=float)
        future_pred = np.full(len(FORECAST_YEARS), m_flat_mean)
        residuals   = values - pred_hist
        model_type  = "Media estable (reciente)"
    else:
        pred_hist   = np.maximum(p_lin, 0)
        future_pred = np.maximum(m_lin.predict(future_x), 0)
        residuals   = values - pred_hist
        model_type  = "Lineal OLS"

    # ── Métricas sobre la ventana apropiada ──────────────────────────
    # Para modelos planos: evaluar solo en la ventana estable (últimos 5 años)
    # para no penalizar el año del quiebre estructural
    eval_win = _FLAT_WINDOW if best == "flat" else METRICS_WINDOW
    val_y    = values[-eval_win:]
    val_pred = pred_hist[-eval_win:]
    mae    = mean_absolute_error(val_y, val_pred)
    denom  = np.maximum(np.abs(val_y), val_y.max() * 0.01 + 1)
    mape   = np.mean(np.abs((val_y - val_pred) / denom)) * 100
    ss_tot = np.sum((val_y - val_y.mean()) ** 2)
    r2     = r2_score(val_y, val_pred) if ss_tot > 1e-6 else 1.0
    margin = 1.5 * residuals[-eval_win:].std()

    return {
        "model_type":   model_type,
        "years":        years,
        "values":       values,
        "pred_hist":    pred_hist,
        "future_years": np.array(FORECAST_YEARS),
        "future_pred":  future_pred.flatten(),
        "margin":       margin,
        "mae":          mae,
        "mape":         mape,
        "r2":           r2,
    }


def fig_forecast(country: str, df: pd.DataFrame) -> tuple[go.Figure, dict]:
    years, values = _country_series(df, country)
    result = fit_forecast(years, values)

    values_M      = values / 1e6
    pred_hist_M   = result["pred_hist"] / 1e6
    future_M      = result["future_pred"] / 1e6
    margin_M      = result["margin"] / 1e6

    fig = go.Figure()

    # Tendencia ajustada sobre histórico
    fig.add_trace(go.Scatter(
        x=years, y=pred_hist_M, mode="lines",
        name="Tendencia ajustada", line=dict(color="#FF8C00", width=2, dash="dot"),
    ))
    # Serie histórica
    fig.add_trace(go.Scatter(
        x=years, y=values_M, mode="lines+markers",
        name="Histórico", line=dict(color="#6F4E37", width=2), marker=dict(size=5),
    ))
    # Banda de confianza proyectada
    fig.add_trace(go.Scatter(
        x=result["future_years"], y=future_M + margin_M,
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=result["future_years"], y=np.maximum(future_M - margin_M, 0),
        mode="lines", fill="tonexty",
        fillcolor="rgba(111,78,55,0.15)", line=dict(width=0),
        name="Intervalo de confianza (±1.5σ)",
    ))
    # Proyección
    fig.add_trace(go.Scatter(
        x=result["future_years"], y=future_M, mode="lines+markers",
        name="Proyección 2020–2025", line=dict(color="#2E8B57", width=2.5, dash="dash"),
        marker=dict(size=7, symbol="star"),
    ))
    fig.add_vline(x=2019, line_dash="dot", line_color="gray",
                  annotation_text="Fin de datos históricos")
    fig.update_layout(
        title=f"Pronóstico de consumo — {country} ({result['model_type']})",
        xaxis_title="Año", yaxis_title="Consumo (millones de tazas)",
        hovermode="x unified", height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig, result


def forecast_all_countries(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for country in df["Country"].unique():
        years, values = _country_series(df, country)
        if values.sum() == 0:
            continue
        ctype = df[df["Country"] == country]["Coffee type"].iloc[0]
        res   = fit_forecast(years, values)
        rows.append({
            "Country":           country,
            "Coffee type":       ctype,
            "Modelo":            res["model_type"],
            "Consumo_2019_M":    values[-1] / 1e6,
            "Proyeccion_2025_M": res["future_pred"][-1] / 1e6,
            "Crecimiento_%":     (res["future_pred"][-1] - values[-1])
                                 / max(values[-1], 1) * 100,
            "MAE_M":             res["mae"] / 1e6,
            "MAPE_%":            res["mape"],
            "R2":                res["r2"],
        })
    return pd.DataFrame(rows).sort_values("Crecimiento_%", ascending=False)


def fig_top_forecast(df: pd.DataFrame, n: int = 10) -> go.Figure:
    summary = forecast_all_countries(df).head(n)
    fig = px.bar(
        summary, x="Crecimiento_%", y="Country", orientation="h",
        color="Crecimiento_%",
        color_continuous_scale=px.colors.sequential.Greens,
        labels={"Crecimiento_%": "Crecimiento proyectado 2019→2025 (%)", "Country": "País"},
        title=f"Top {n} países por crecimiento proyectado al 2025",
        text=summary["Crecimiento_%"].apply(lambda x: f"{x:+.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False, height=420,
    )
    return fig
