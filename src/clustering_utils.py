"""
Utilidades de clustering: validación, selección de k, y caracterización de clusters.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

RANDOM_STATE = 42


def evaluate_kmeans_k(X: np.ndarray, k_range=range(2, 11),
                      sample_size: int = 30000) -> pd.DataFrame:
    """
    Evalúa K-Means para diferentes valores de k.
    Retorna DataFrame con inertia, Silhouette, Davies-Bouldin y Calinski-Harabasz.
    """
    if len(X) > sample_size:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_eval)
        results.append({
            'k'                   : k,
            'inertia'             : km.inertia_,
            'silhouette'          : silhouette_score(X_eval, labels, sample_size=5000, random_state=RANDOM_STATE),
            'davies_bouldin'      : davies_bouldin_score(X_eval, labels),
            'calinski_harabasz'   : calinski_harabasz_score(X_eval, labels),
        })
        print(f'  k={k}: sil={results[-1][\"silhouette\"]:.4f}, db={results[-1][\"davies_bouldin\"]:.4f}')

    return pd.DataFrame(results)


def suggest_k(metrics_df: pd.DataFrame) -> int:
    """
    Sugiere k óptimo basándose en mayoría de métricas:
    - Silhouette máximo
    - Davies-Bouldin mínimo
    - Calinski-Harabasz máximo
    """
    k_sil = metrics_df.loc[metrics_df['silhouette'].idxmax(), 'k']
    k_db  = metrics_df.loc[metrics_df['davies_bouldin'].idxmin(), 'k']
    k_ch  = metrics_df.loc[metrics_df['calinski_harabasz'].idxmax(), 'k']
    votes = [k_sil, k_db, k_ch]
    # Mayoría simple
    from collections import Counter
    k_suggested = Counter(votes).most_common(1)[0][0]
    print(f'Sugerencia por métrica — Silhouette: {k_sil}, DB: {k_db}, CH: {k_ch}')
    print(f'k sugerido por mayoría: {k_suggested}')
    return k_suggested


def check_stability(X: np.ndarray, k: int, n_seeds: int = 10,
                    sample_size: int = 20000) -> dict:
    """Verifica estabilidad del clustering con múltiples semillas aleatorias."""
    if len(X) > sample_size:
        idx = np.random.RandomState(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    scores = []
    for seed in range(n_seeds):
        km = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=seed)
        labels = km.fit_predict(X_eval)
        s = silhouette_score(X_eval, labels, sample_size=5000, random_state=42)
        scores.append(s)

    cv = np.std(scores) / np.mean(scores) * 100
    stable = cv < 5.0

    return {
        'scores'     : scores,
        'mean'       : np.mean(scores),
        'std'        : np.std(scores),
        'cv_pct'     : cv,
        'is_stable'  : stable,
    }


def characterize_clusters(df_features: pd.DataFrame, cluster_col: str = 'cluster_km') -> pd.DataFrame:
    """Genera tabla de caracterización de clusters con estadísticas descriptivas."""
    numeric_cols = df_features.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != cluster_col and 'flag_' not in c]

    return df_features.groupby(cluster_col)[numeric_cols].agg(
        ['mean', 'median', 'std']
    ).round(2)
