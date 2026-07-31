import warnings
import numpy as np
import pandas as pd
from .config import DATASET_NORMALIZED, FEATURES_SELECTED


def load_dataset(feature_set=None, path=None, strict=True):
    """
    Carga el dataset normalizado (z-score intra-sujeto).

    Parameters
    ----------
    feature_set : list[str] | None
        Features a cargar. Por defecto FEATURES_SELECTED (WESAD, M=12).
    path : str | Path | None
        Ruta al CSV. Por defecto DATASET_NORMALIZED.
    strict : bool
        True  → lanza ValueError si alguna feature falta en el CSV.
        False → usa solo las features presentes (útil para datasets externos).

    Returns
    -------
    X        : ndarray (n_windows, n_features)
    y        : ndarray (n_windows,) — 0=baseline, 1=stress
    groups   : ndarray (n_windows,) — subject ID (para LOSO)
    features : list[str] — features efectivamente usadas
    """
    if feature_set is None:
        feature_set = FEATURES_SELECTED
    if path is None:
        path = DATASET_NORMALIZED

    df = pd.read_csv(path)

    for col in ('label', 'subject'):
        if col not in df.columns:
            raise ValueError(f"El CSV debe tener columna '{col}'.")

    missing = [f for f in feature_set if f not in df.columns]
    if missing:
        if strict:
            raise ValueError(
                f"Features no encontradas en el dataset: {missing}\n"
                "Usa strict=False para ignorarlas automáticamente."
            )
        available = [f for f in feature_set if f in df.columns]
        warnings.warn(
            f"[load_dataset] Features no disponibles (ignoradas): {missing}\n"
            f"  Usando {len(available)} de {len(feature_set)}: {available}",
            UserWarning, stacklevel=2,
        )
        feature_set = available

    if not feature_set:
        raise ValueError("Ninguna feature del feature_set está presente en el dataset.")

    # P5: aviso si los datos no parecen normalizados (std global >> 1 en alguna feature)
    stds = df[feature_set].std()
    if (stds > 10).any():
        warnings.warn(
            "[load_dataset] Algunas features tienen std > 10. "
            "¿El dataset ya tiene z-score intra-sujeto aplicado? "
            "SVM y ANN son sensibles a la escala.",
            UserWarning, stacklevel=2,
        )

    X = df[feature_set].to_numpy(dtype=np.float64)
    y = df['label'].to_numpy(dtype=np.int32)
    groups = df['subject'].to_numpy()

    return X, y, groups, list(feature_set)


def dataset_summary(X, y, groups, features):
    """Imprime un resumen del dataset cargado."""
    subjects = sorted(set(groups))
    print(f"Ventanas totales : {len(y)}")
    print(f"Features         : {len(features)}")
    print(f"Sujetos (LOSO)   : {len(subjects)} — {subjects}")
    print(f"Clase 0 (baseline): {(y == 0).sum()}  |  Clase 1 (stress): {(y == 1).sum()}")
    print(f"Ratio desbalance  : {(y == 0).sum() / (y == 1).sum():.2f}:1")
