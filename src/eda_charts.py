import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from src.data_loader import YEAR_COLS, YEAR_NUMS, top_countries, coffee_type_summary, global_trend, growth_table, to_long

COFFEE_COLORS = {
    "Arabica": "#A0522D",
    "Robusta": "#D4956A",
    "Arabica/Robusta": "#8B5E3C",
    "Robusta/Arabica": "#C49A6C",
}

# Lighter caramel-to-brown scale so text labels stay readable
PALETTE = ["#F5E2C8", "#E8C49A", "#D4956A", "#BF7040", "#A05020", "#7A3810"]


def fig_top_countries(df: pd.DataFrame, n: int = 10) -> go.Figure:
    top = top_countries(df, n)
    top["consumption_M"] = top["Total_domestic_consumption"] / 1e6
    fig = px.bar(
        top, x="consumption_M", y="Country", orientation="h",
        color="consumption_M",
        color_continuous_scale=PALETTE,
        labels={"consumption_M": "Consumo Total (Millones de tazas)", "Country": "País"},
        title=f"Top {n} Países por Consumo Doméstico Total (1990–2020)",
        text=top["consumption_M"].apply(lambda x: f"{x:,.0f}M"),
    )
    fig.update_traces(textposition="outside", textfont=dict(color="#3E1F00", size=11))
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=420)
    return fig


def fig_coffee_type_pie(df: pd.DataFrame) -> go.Figure:
    ct = coffee_type_summary(df)
    ct["consumption_M"] = ct["Total_domestic_consumption"] / 1e6
    fig = px.pie(
        ct, values="consumption_M", names="Coffee type",
        title="Distribución del Consumo por Tipo de Café",
        color="Coffee type",
        color_discrete_map=COFFEE_COLORS,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def fig_coffee_type_bar(df: pd.DataFrame) -> go.Figure:
    ct = coffee_type_summary(df)
    ct["consumption_M"] = ct["Total_domestic_consumption"] / 1e6
    fig = px.bar(
        ct, x="Coffee type", y="consumption_M",
        color="Coffee type",
        color_discrete_map=COFFEE_COLORS,
        labels={"consumption_M": "Consumo (Millones tazas)", "Coffee type": "Tipo"},
        title="Consumo Total por Tipo de Café",
        text=ct["consumption_M"].apply(lambda x: f"{x:,.0f}M"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def fig_global_trend(df: pd.DataFrame) -> go.Figure:
    gt = global_trend(df)
    gt["consumption_B"] = gt["consumption"] / 1e9
    fig = px.line(
        gt, x="year", y="consumption_B",
        markers=True,
        labels={"consumption_B": "Consumo Global (Miles de millones de tazas)", "year": "Año"},
        title="Tendencia Global del Consumo Doméstico de Café (1990–2020)",
    )
    fig.update_traces(line_color="#6F4E37", line_width=3, marker_size=6)
    fig.update_layout(hovermode="x unified")
    return fig


def fig_country_trends(df: pd.DataFrame, n: int = 10) -> go.Figure:
    top = top_countries(df, n)["Country"].tolist()
    long = to_long(df)
    filtered = long[long["Country"].isin(top)].groupby(["Country", "year"])["consumption"].sum().reset_index()
    filtered["consumption_M"] = filtered["consumption"] / 1e6
    fig = px.line(
        filtered, x="year", y="consumption_M", color="Country",
        labels={"consumption_M": "Consumo (Millones tazas)", "year": "Año", "Country": "País"},
        title=f"Evolución del Consumo — Top {n} Países",
        markers=False,
    )
    fig.update_layout(hovermode="x unified", height=480)
    return fig


def fig_growth_rates(df: pd.DataFrame, n: int = 15) -> go.Figure:
    gt = growth_table(df).dropna(subset=["growth_pct"]).head(n)
    fig = px.bar(
        gt, x="growth_pct", y="Country", orientation="h",
        color="growth_pct",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        labels={"growth_pct": "Crecimiento Total 1990→2020 (%)", "Country": "País"},
        title=f"Top {n} Países por Crecimiento de Consumo 1990–2020",
        text=gt["growth_pct"].apply(lambda x: f"{x:+.0f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False, height=460)
    return fig


def fig_cagr_scatter(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("Country")[["Total_domestic_consumption", "cagr"]].agg(
        {"Total_domestic_consumption": "sum", "cagr": "mean"}
    ).reset_index().dropna()
    grp["size"] = grp["Total_domestic_consumption"] / grp["Total_domestic_consumption"].max() * 60 + 5
    grp["cagr_pct"] = grp["cagr"] * 100
    grp["consumption_M"] = grp["Total_domestic_consumption"] / 1e6
    fig = px.scatter(
        grp, x="consumption_M", y="cagr_pct",
        size="size", text="Country",
        labels={"consumption_M": "Consumo Total (M tazas)", "cagr_pct": "CAGR (%)"},
        title="Tamaño de Mercado vs Tasa de Crecimiento Anual Compuesto (CAGR)",
        color="cagr_pct",
        color_continuous_scale=px.colors.diverging.RdYlGn,
        log_x=True,
    )
    fig.update_traces(textposition="top center", marker_opacity=0.7)
    fig.update_layout(coloraxis_showscale=False, height=520)
    return fig


def fig_year_correlation(df: pd.DataFrame) -> go.Figure:
    corr = df[YEAR_COLS].corr()
    short_labels = [y.split("/")[0] for y in YEAR_COLS]
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=short_labels,
        y=short_labels,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 7},
    ))
    fig.update_layout(title="Correlación entre Años (Consumo por País)", height=550)
    return fig


def fig_boxplot_by_type(df: pd.DataFrame) -> go.Figure:
    long = to_long(df)
    long["consumption_M"] = long["consumption"] / 1e6
    fig = px.box(
        long, x="Coffee type", y="consumption_M",
        color="Coffee type",
        color_discrete_map=COFFEE_COLORS,
        labels={"consumption_M": "Consumo Anual (M tazas)", "Coffee type": "Tipo"},
        title="Distribución del Consumo Anual por Tipo de Café",
        points="outliers",
    )
    fig.update_layout(showlegend=False)
    return fig


def fig_heatmap_countries(df: pd.DataFrame, n: int = 15) -> go.Figure:
    top = top_countries(df, n)["Country"].tolist()
    sub = df[df["Country"].isin(top)].groupby("Country")[YEAR_COLS].sum()
    sub_M = sub / 1e6
    short_labels = [y.split("/")[0] for y in YEAR_COLS]
    fig = go.Figure(data=go.Heatmap(
        z=sub_M.values,
        x=short_labels,
        y=sub_M.index.tolist(),
        colorscale=PALETTE,
        colorbar_title="M tazas",
    ))
    fig.update_layout(
        title=f"Mapa de Calor: Consumo por Año — Top {n} Países",
        xaxis_title="Año", yaxis_title="País", height=500
    )
    return fig


def fig_cumulative_share(df: pd.DataFrame) -> go.Figure:
    top5 = top_countries(df, 5)["Country"].tolist()
    yearly = {}
    for col in YEAR_COLS:
        total = df[col].sum()
        yearly[col] = {c: df[df["Country"] == c][col].sum() / total * 100 for c in top5}
        yearly[col]["Otros"] = 100 - sum(yearly[col].values())
    rows = []
    for label, year in zip(YEAR_COLS, YEAR_NUMS):
        for country, pct in yearly[label].items():
            rows.append({"year": year, "Country": country, "share_pct": pct})
    share_df = pd.DataFrame(rows)
    fig = px.area(
        share_df, x="year", y="share_pct", color="Country",
        labels={"share_pct": "Participación (%)", "year": "Año"},
        title="Evolución de la Participación de Mercado — Top 5 + Otros",
    )
    fig.update_layout(hovermode="x unified", height=420)
    return fig
