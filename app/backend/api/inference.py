"""
Lógica de inferencia del backend:
  - run_pipeline_on_files: procesa un directorio con EDA/BVP/ACC/TEMP.csv
  - predict_window:        clasifica una ventana ya extraída (feature dict)
"""
import numpy as np
import pandas as pd
from pathlib import Path

from pipeline.loader_e4 import load_e4_subject
from pipeline.windowing import build_dataset
from pipeline.stats import normalize_intrasubject


def run_pipeline_on_files(subject_dir: Path, model, features: list) -> dict:
    """
    Carga las señales E4 desde subject_dir, extrae features y aplica el modelo.

    Returns dict con:
        subject, n_windows, n_stress, n_rest, stress_ratio,
        windows: list de {window_id, t_start_s, pred_class, pred_label, pred_proba}
    """
    data = load_e4_subject(subject_dir)
    subject_id = data["subject"] if data["subject"] != subject_dir.name else "subject"

    df = build_dataset({subject_id: data})
    if df.empty:
        raise ValueError("No se generaron ventanas válidas. Verifica los archivos CSV.")

    avail = [f for f in features if f in df.columns]
    df_norm = normalize_intrasubject(df, avail)
    X = df_norm[avail].fillna(0).to_numpy()

    y_pred = model.predict(X)
    y_prob = (model.predict_proba(X)[:, 1]
              if hasattr(model, "predict_proba") else np.full(len(y_pred), None))

    windows = []
    for i, (pred, prob, t) in enumerate(
        zip(y_pred, y_prob, df_norm["t_start_s"].values)
    ):
        windows.append({
            "window_id":  int(i),
            "t_start_s":  float(t),
            "t_start_min": round(float(t) / 60, 2),
            "pred_class": int(pred),
            "pred_label": "stress" if pred == 1 else "no-stress",
            "pred_proba": round(float(prob), 4) if prob is not None else None,
        })

    n_stress = int((y_pred == 1).sum())
    n_rest   = int((y_pred == 0).sum())

    return {
        "subject":      subject_id,
        "n_windows":    len(windows),
        "n_stress":     n_stress,
        "n_rest":       n_rest,
        "stress_ratio": round(n_stress / max(len(windows), 1), 3),
        "duration_min": round(float(df_norm["t_start_s"].max()) / 60 + 1, 1),
        "windows":      windows,
    }


def predict_window(features_dict: dict, model, feature_names: list) -> dict:
    """
    Clasifica una única ventana ya extraída.
    features_dict: {feature_name: value} — puede tener claves extra o faltantes.
    Valores faltantes se imputan con 0.
    """
    x = np.array([features_dict.get(f, 0.0) for f in feature_names],
                 dtype=np.float64).reshape(1, -1)

    y_pred = int(model.predict(x)[0])
    y_prob = (float(model.predict_proba(x)[0, 1])
              if hasattr(model, "predict_proba") else None)

    return {
        "class":  y_pred,
        "label":  "stress" if y_pred == 1 else "no-stress",
        "proba":  round(y_prob, 4) if y_prob is not None else None,
    }
