import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.data_loader import YEAR_COLS, YEAR_NUMS, load_data

# Base names assigned by max log_total rank (rank 0 = cluster with largest country)
_NAMES_BY_RANK = [
    "Mercado Gigante Consolidado",     # Cluster con Brasil — grandes mercados establecidos
    "Mercado Dinámico de Alto Crecimiento",  # Indonesia, Filipinas, Vietnam — CAGR >7%
    "Mercado Pequeño en Declive",      # Mercados pequeños con CAGR negativo
    "Sin Consumo Registrado",          # Países sin datos significativos
]

# Palette supports up to 8 clusters
CLUSTER_COLORS_PALETTE = [
    "#6F4E37", "#2E8B57", "#C49A6C", "#1E3A5F",
    "#CC4400", "#9B59B6", "#1ABC9C", "#E67E22",
]


def _cluster_label(rank: int) -> str:
    if rank < len(_NAMES_BY_RANK):
        return _NAMES_BY_RANK[rank]
    return f"Segmento {rank + 1}"


def _cluster_color(rank: int) -> str:
    return CLUSTER_COLORS_PALETTE[rank % len(CLUSTER_COLORS_PALETTE)]


CLUSTER_NAMES = {i: _cluster_label(i) for i in range(4)}
CLUSTER_COLORS = CLUSTER_COLORS_PALETTE[:4]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("Country")

    total = grp["Total_domestic_consumption"].sum()
    last_year = grp["2019/20"].sum()
    first_year = grp["1990/91"].sum()
    avg = grp[YEAR_COLS].sum().mean(axis=1)

    # CAGR per country
    cagr = grp["cagr"].mean()

    # Volatility: coefficient of variation
    yoy_std = grp[YEAR_COLS].sum().std(axis=1)
    yoy_mean = grp[YEAR_COLS].sum().mean(axis=1)
    cv = yoy_std / yoy_mean.replace(0, np.nan)

    # YoY growth rate in last 5 years vs first 5 years
    years_early = YEAR_COLS[:5]
    years_late = YEAR_COLS[-5:]
    early_avg = grp[years_early].sum().mean(axis=1)
    late_avg = grp[years_late].sum().mean(axis=1)
    momentum = (late_avg - early_avg) / early_avg.replace(0, np.nan)

    feats = pd.DataFrame({
        "log_total": np.log1p(total),
        "log_last": np.log1p(last_year),
        "cagr": cagr.fillna(0),
        "cv": cv.fillna(cv.median()),
        "momentum": momentum.fillna(0),
    }).dropna()

    return feats


def elbow_data(feats: pd.DataFrame, max_k: int = 10) -> tuple[list, list]:
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    inertias = []
    ks = list(range(2, max_k + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return ks, inertias


def fig_elbow(feats: pd.DataFrame) -> go.Figure:
    ks, inertias = elbow_data(feats)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=inertias, mode="lines+markers",
        line=dict(color="#6F4E37", width=2),
        marker=dict(size=8),
    ))
    fig.add_vline(x=4, line_dash="dot", line_color="#2E8B57", annotation_text="k óptimo = 4")
    fig.update_layout(
        title="Método del Codo — Selección de k óptimo",
        xaxis_title="Número de Clusters (k)",
        yaxis_title="Inercia (Within-cluster SSE)",
        height=380,
    )
    return fig


def run_clustering(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    feats = build_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    result = feats.reset_index()  # moves Country from index to column
    result["cluster"] = labels
    result["pc1"] = coords[:, 0]
    result["pc2"] = coords[:, 1]

    # Assign semantic names: sort clusters by max log_total (so Brazil's cluster = Gigante)
    order = (
        result.groupby("cluster")["log_total"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    remap = {old: new for new, old in enumerate(order)}
    result["cluster"] = result["cluster"].map(remap)
    result["cluster_name"] = result["cluster"].apply(_cluster_label)
    result["color"] = result["cluster"].apply(_cluster_color)

    # Merge original metadata
    meta = df.groupby("Country").agg(
        Coffee_type=("Coffee type", "first"),
        Total_M=("Total_domestic_consumption", lambda x: x.sum() / 1e6),
    ).reset_index()
    result = result.merge(meta, on="Country", how="left")

    return result


def fig_cluster_scatter(result: pd.DataFrame) -> go.Figure:
    k = result["cluster"].nunique()
    colors = [_cluster_color(i) for i in range(k)]
    fig = px.scatter(
        result, x="pc1", y="pc2",
        color="cluster_name",
        color_discrete_sequence=colors,
        hover_data={"Country": True, "cluster_name": True, "Total_M": ":.1f",
                    "cagr": ":.3f", "pc1": False, "pc2": False},
        text="Country",
        labels={"pc1": "Componente Principal 1", "pc2": "Componente Principal 2",
                "cluster_name": "Segmento"},
        title="Segmentación de Mercados de Café — PCA 2D",
        size="Total_M",
        size_max=45,
    )
    fig.update_traces(textposition="top center", marker_opacity=0.8)
    fig.update_layout(height=580, legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    return fig


def fig_cluster_profiles(result: pd.DataFrame) -> go.Figure:
    profile = result.groupby("cluster_name")[["log_total", "cagr", "cv", "momentum"]].mean().reset_index()
    profile_melt = profile.melt(id_vars="cluster_name", var_name="feature", value_name="value")

    feature_labels = {
        "log_total": "Tamaño (log)",
        "cagr": "CAGR",
        "cv": "Volatilidad",
        "momentum": "Momentum",
    }
    profile_melt["feature"] = profile_melt["feature"].map(feature_labels)

    k = result["cluster"].nunique()
    colors = [_cluster_color(i) for i in range(k)]
    fig = px.bar(
        profile_melt, x="feature", y="value", color="cluster_name",
        barmode="group",
        color_discrete_sequence=colors,
        labels={"value": "Valor Promedio", "feature": "Variable", "cluster_name": "Segmento"},
        title="Perfil Promedio por Segmento de Mercado",
    )
    fig.update_layout(height=400)
    return fig


def cluster_summary_table(result: pd.DataFrame) -> pd.DataFrame:
    return (
        result.groupby("cluster_name")
        .agg(
            Países=("Country", "count"),
            Consumo_Total_M=("Total_M", "sum"),
            CAGR_Promedio=("cagr", "mean"),
            Momentum_Promedio=("momentum", "mean"),
        )
        .round(4)
        .reset_index()
        .rename(columns={"cluster_name": "Segmento"})
    )
