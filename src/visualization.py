"""
Funciones de visualización reutilizables — estilo oscuro unificado.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

DARK_BG      = '#0f1117'
DARK_PLOT    = '#1a1d27'
DARK_GRID    = '#2a2d3e'
DARK_TEXT    = '#e0e0e0'
DARK_TICKS   = '#b0b0b0'
HOLO_BG      = '#060c18'
HOLO_GRID    = '#1e3a5f'
HOLO_TEXT    = '#7eb8e8'

CLUSTER_COLORS = [
    '#4FC3F7', '#FF8A65', '#81C784', '#CE93D8', '#FFD54F',
    '#80CBC4', '#F48FB1', '#BCAAA4', '#90CAF9'
]


def dark_layout(fig: go.Figure, title: str = '', width: int = 950, height: int = 600,
                holo: bool = False) -> go.Figure:
    """Aplica el tema oscuro estándar del proyecto a una figura Plotly."""
    bg = HOLO_BG if holo else DARK_BG
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        paper_bgcolor=bg,
        plot_bgcolor=DARK_PLOT if not holo else HOLO_BG,
        font=dict(color=HOLO_TEXT if holo else DARK_TEXT),
        width=width, height=height
    )
    fig.update_xaxes(gridcolor=DARK_GRID)
    fig.update_yaxes(gridcolor=DARK_GRID)
    return fig


def scatter_3d_clusters(df: pd.DataFrame, x: str, y: str, z: str,
                        cluster_col: str = 'cluster_km',
                        sample_n: int = 15000,
                        title: str = 'Mapa de Operación 3D',
                        cluster_names: dict = None,
                        holo: bool = True) -> go.Figure:
    """Scatter 3D interactivo coloreado por cluster."""
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    if cluster_names is None:
        cluster_names = {c: f'Modo {c}' for c in sorted(df[cluster_col].unique())}

    fig = go.Figure()
    for cid in sorted(sample[cluster_col].unique()):
        sub = sample[sample[cluster_col] == cid]
        fig.add_trace(go.Scatter3d(
            x=sub[x], y=sub[y], z=sub[z],
            mode='markers',
            name=cluster_names.get(cid, f'Cluster {cid}'),
            marker=dict(size=2.5, color=CLUSTER_COLORS[cid], opacity=0.5)
        ))

    bg = HOLO_BG if holo else DARK_BG
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title=x, backgroundcolor=bg, gridcolor=HOLO_GRID, color=HOLO_TEXT, showbackground=True),
            yaxis=dict(title=y, backgroundcolor=bg, gridcolor=HOLO_GRID, color=HOLO_TEXT, showbackground=True),
            zaxis=dict(title=z, backgroundcolor=bg, gridcolor=HOLO_GRID, color=HOLO_TEXT, showbackground=True),
            bgcolor=HOLO_BG if holo else DARK_PLOT
        ),
        paper_bgcolor=bg,
        font=dict(color=HOLO_TEXT if holo else DARK_TEXT)
    )
    return fig


def efficiency_surface(df: pd.DataFrame, q_col: str = 'Fluj_Desc',
                       r_col: str = 'RPM', z_col: str = 'Potencia_especifica',
                       n_grid: int = 60) -> go.Figure:
    """Superficie de eficiencia interpolada en malla Q × RPM."""
    from scipy.interpolate import griddata

    data = df[[q_col, r_col, z_col]].dropna()
    q_q1, q_q99 = data[q_col].quantile([0.05, 0.95])
    r_q1, r_q99 = data[r_col].quantile([0.05, 0.95])
    z_q99 = data[z_col].quantile(0.99)
    data  = data[data[z_col] < z_q99]

    q_range = np.linspace(q_q1, q_q99, n_grid)
    r_range = np.linspace(r_q1, r_q99, n_grid)
    Q, R    = np.meshgrid(q_range, r_range)
    Z       = griddata(data[[q_col, r_col]].values, data[z_col].values, (Q, R), method='linear')

    fig = go.Figure(data=[go.Surface(
        x=Q, y=R, z=Z,
        colorscale='RdYlGn_r',
        colorbar=dict(title='kW/GPM<br>(↓ eficiente)'),
        opacity=0.85,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True))
    )])
    fig.update_layout(
        title='Superficie de Eficiencia Energética',
        scene=dict(
            xaxis=dict(title=q_col, backgroundcolor=HOLO_BG, gridcolor=HOLO_GRID,
                       color=HOLO_TEXT, showbackground=True),
            yaxis=dict(title=r_col, backgroundcolor=HOLO_BG, gridcolor=HOLO_GRID,
                       color=HOLO_TEXT, showbackground=True),
            zaxis=dict(title=z_col, backgroundcolor=HOLO_BG, gridcolor=HOLO_GRID,
                       color=HOLO_TEXT, showbackground=True),
            bgcolor=HOLO_BG
        ),
        paper_bgcolor=HOLO_BG,
        font=dict(color=HOLO_TEXT),
        width=1000, height=700
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, cols: list = None,
                        method: str = 'spearman', title: str = 'Correlación de Spearman') -> plt.Figure:
    """Mapa de calor de correlación con estilo oscuro."""
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[cols].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11), facecolor=DARK_BG)
    ax.set_facecolor(DARK_PLOT)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1,
                linewidths=0.5, linecolor=DARK_BG,
                annot_kws={'size': 6}, ax=ax)
    ax.set_title(title, color=DARK_TEXT, pad=15)
    ax.tick_params(colors=DARK_TICKS)
    return fig
