import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from .features import ALL_FEAT_KEYS


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled = np.sqrt(
        ((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2)
    )
    return (np.mean(x) - np.mean(y)) / pooled if pooled > 1e-10 else np.nan


def effect_label(d: float) -> str:
    d = abs(d)
    if np.isnan(d): return 'N/A'
    if d < 0.2:     return 'despreciable'
    if d < 0.5:     return 'pequeño'
    if d < 0.8:     return 'mediano'
    return 'grande'


def run_statistical_tests(windows_df: pd.DataFrame, feat_keys=None) -> pd.DataFrame:
    """
    Mann-Whitney U + Cohen's d para cada feature (Baseline vs Stress).
    Retorna DataFrame ordenado por |d de Cohen| descendente.
    """
    if feat_keys is None:
        feat_keys = ALL_FEAT_KEYS
    base_df   = windows_df[windows_df.label == 0]
    stress_df = windows_df[windows_df.label == 1]
    records = []
    for feat in feat_keys:
        x = base_df[feat].dropna().values
        y = stress_df[feat].dropna().values
        if len(x) < 5 or len(y) < 5:
            continue
        _, p = mannwhitneyu(x, y, alternative='two-sided')
        d = cohens_d(x, y)
        records.append({
            'feature':       feat,
            'n_baseline':    len(x),
            'n_stress':      len(y),
            'mean_baseline': x.mean(),
            'mean_stress':   y.mean(),
            'p_value':       p,
            'significant':   p < 0.05,
            'cohens_d':      d,
            'effect':        effect_label(d),
        })
    return pd.DataFrame(records).sort_values('cohens_d', key=abs, ascending=False)


def normalize_intrasubject(windows_df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Z-score intra-sujeto: cada feature se normaliza por separado dentro de cada sujeto."""
    df = windows_df.copy()
    df[feat_cols] = df[feat_cols].astype(float)  # evita TypeError en columnas int (e.g. scr_count)
    for subj in df['subject'].unique():
        mask = df['subject'] == subj
        for col in feat_cols:
            vals = df.loc[mask, col]
            mu, sigma = vals.mean(), vals.std()
            if sigma > 1e-10:
                df.loc[mask, col] = (vals - mu) / sigma
    return df
