"""
Validación cruzada del pipeline WESAD sobre el dataset Campanella 2024.

Modo 1 — Validación completa (dataset con etiquetas de protocolo):
    conda run -n tesis python scripts/run_campanella.py

Modo 2 — Inferencia sobre un sujeto sin etiquetas (uso aplicación móvil):
    conda run -n tesis python scripts/run_campanella.py \
        --subject data/Data_29_subjects/Subjects/subject_01

Argumentos opcionales:
    --data    Carpeta raíz de sujetos Campanella (default: data/Data_29_subjects/Subjects)
    --model   Modelo a usar: RF | SVM | ANN | XGBoost | LDA | all (default: all)

Salidas en results/campanella/:
    campanella_windowed_normalized.csv
    campanella_results.csv         ← métricas por modelo
    campanella_per_subject.csv     ← métricas por sujeto (mejor modelo)
    inference_<sujeto>_<modelo>.csv ← modo inferencia
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from pipeline.config import FEATURES_SELECTED, RESULTS_DIR
from pipeline.loader_e4 import load_campanella_all, load_e4_subject
from pipeline.stats import normalize_intrasubject
from pipeline.windowing import build_dataset

MODELS_DIR  = Path(__file__).parent.parent / "models"
CAMPA_DIR   = Path(__file__).parent.parent / "data" / "Data_29_subjects" / "Subjects"
MODEL_NAMES = ('RF', 'SVM', 'ANN', 'XGBoost', 'LDA')


def _load_models(names):
    models = {}
    for n in names:
        p = MODELS_DIR / f"{n}.joblib"
        if p.exists():
            models[n] = joblib.load(p)
        else:
            print(f"  AVISO: {p.name} no encontrado, omitido.")
    return models


def _metrics(model, X, y):
    y_pred = model.predict(X)
    has_proba = hasattr(model, 'predict_proba')
    y_prob = model.predict_proba(X)[:, 1] if has_proba else None
    return {
        'accuracy':          accuracy_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'f1_macro':          f1_score(y, y_pred, average='macro',    zero_division=0),
        'f1_weighted':       f1_score(y, y_pred, average='weighted', zero_division=0),
        'precision_stress':  precision_score(y, y_pred, pos_label=1, zero_division=0),
        'recall_stress':     recall_score(y, y_pred,    pos_label=1, zero_division=0),
        'roc_auc':           roc_auc_score(y, y_prob) if y_prob is not None and len(np.unique(y)) > 1 else np.nan,
        'n_windows':         len(y),
        'n_stress':          int((y == 1).sum()),
        'n_rest':            int((y == 0).sum()),
    }


def _prepare_dataset(all_data, out_dir):
    """Extrae features, normaliza y retorna (X, y, df_norm, feats)."""
    print('\n[2] Extrayendo features (ventanas 60s, overlap 50%)...')
    windows_df = build_dataset(all_data)
    n0 = (windows_df.label == 0).sum()
    n1 = (windows_df.label == 1).sum()
    print(f'  Ventanas: {len(windows_df)}  (no-estrés={n0}, estrés={n1})')

    print('\n[3] Normalizando (z-score intra-sujeto)...')
    feats = [f for f in FEATURES_SELECTED if f in windows_df.columns]
    missing = [f for f in FEATURES_SELECTED if f not in windows_df.columns]
    if missing:
        print(f'  AVISO: features no disponibles (ignoradas): {missing}')
    df_norm = normalize_intrasubject(windows_df, feats)
    df_norm.to_csv(out_dir / 'campanella_windowed_normalized.csv', index=False)
    print(f'  Features usadas: {len(feats)}/{len(FEATURES_SELECTED)}')

    X = df_norm[feats].fillna(0).to_numpy()
    y = df_norm['label'].to_numpy()
    return X, y, df_norm, feats


def _run_validation(args, out_dir):
    print('=' * 60)
    print('  VALIDACIÓN — Campanella 2024 (n=29)')
    print('  Modelos WESAD aplicados sin reentrenamiento')
    print('=' * 60)

    # 1. Cargar sujetos
    print(f'\n[1] Cargando sujetos desde: {args.data}')
    all_data = load_campanella_all(args.data)
    print(f'  Sujetos cargados: {len(all_data)}')

    # 2-3. Features + normalización
    X, y, df_norm, feats = _prepare_dataset(all_data, out_dir)

    # 4. Modelos
    names  = list(MODEL_NAMES) if args.model == 'all' else [args.model]
    print(f'\n[4] Cargando modelos: {names}')
    models = _load_models(names)

    # 5. Métricas globales
    print('\n[5] Métricas globales (todos los sujetos):')
    print(f'  {"Modelo":10s} {"BA":>8} {"F1-macro":>10} {"AUC":>8}')
    print('  ' + '-' * 42)
    rows = []
    for name, model in models.items():
        m = _metrics(model, X, y)
        rows.append({'model': name, **m})
        print(f'  {name:10s} {m["balanced_accuracy"]:>8.3f} {m["f1_macro"]:>10.3f} {m["roc_auc"]:>8.3f}')
    pd.DataFrame(rows).to_csv(out_dir / 'campanella_results.csv', index=False)

    # 6. Métricas por sujeto con el mejor modelo disponible
    ref_model_name = next((n for n in ('SVM', 'RF', 'ANN', 'XGBoost', 'LDA') if n in models), None)
    if ref_model_name is None:
        return
    ref_model = models[ref_model_name]
    print(f'\n[6] Métricas por sujeto ({ref_model_name}):')
    print(f'  {"Sujeto":12s} {"BA":>8} {"F1":>8} {"AUC":>8} {"Ventanas":>10}')
    print('  ' + '-' * 52)
    subj_rows = []
    for subj in sorted(df_norm['subject'].unique()):
        mask = df_norm['subject'] == subj
        Xi, yi = X[mask], y[mask]
        if len(np.unique(yi)) < 2:
            print(f'  {subj:12s}  — omitido (solo una clase)')
            continue
        m = _metrics(ref_model, Xi, yi)
        subj_rows.append({'subject': subj, **m})
        print(f'  {subj:12s} {m["balanced_accuracy"]:>8.3f} {m["f1_macro"]:>8.3f} {m["roc_auc"]:>8.3f} {m["n_windows"]:>10}')
    pd.DataFrame(subj_rows).to_csv(out_dir / 'campanella_per_subject.csv', index=False)

    print('\n' + '=' * 60)
    print(f'  Resultados guardados en: {out_dir}')
    print('=' * 60)


def _run_inference(args, out_dir):
    print('=' * 60)
    print('  INFERENCIA — sujeto único (sin etiquetas)')
    print('=' * 60)

    names  = list(MODEL_NAMES) if args.model == 'all' else [args.model]
    models = _load_models(names)

    print(f'\n[1] Cargando sujeto: {args.subject}')
    data       = load_e4_subject(args.subject)
    subject_id = data['subject']
    task4_s    = (len(data['EDA']) / 4.0) - 1920
    print(f'  Sujeto: {subject_id}  |  Duración: {len(data["EDA"])/4/60:.1f} min  |  Task4 estimada: {task4_s:.0f}s')

    print('\n[2] Extrayendo features...')
    df = build_dataset({subject_id: data})
    if df.empty:
        print('  ERROR: No se obtuvieron ventanas válidas.')
        sys.exit(1)

    feats  = [f for f in FEATURES_SELECTED if f in df.columns]
    df_n   = normalize_intrasubject(df, feats)
    X      = df_n[feats].fillna(0).to_numpy()
    print(f'  Ventanas válidas: {len(df)}  |  Features: {len(feats)}')

    for name, model in models.items():
        y_pred = model.predict(X)
        y_prob = (model.predict_proba(X)[:, 1]
                  if hasattr(model, 'predict_proba') else np.full(len(y_pred), np.nan))

        result = pd.DataFrame({
            'subject':    subject_id,
            'window_id':  range(len(y_pred)),
            't_start_s':  df_n['t_start_s'].values,
            'pred_class': y_pred,
            'pred_proba': np.round(y_prob, 4),
            'pred_label': ['stress' if p == 1 else 'no-stress' for p in y_pred],
        })
        out_csv = out_dir / f'inference_{subject_id}_{name}.csv'
        result.to_csv(out_csv, index=False)

        n_s = (y_pred == 1).sum()
        n_r = (y_pred == 0).sum()
        print(f'\n  [{name}] Stress: {n_s} ventanas  |  No-estrés: {n_r} ventanas')
        print(f'  Guardado: {out_csv}')

    print('\n' + '=' * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Validación Campanella 2024 con modelos entrenados en WESAD.')
    parser.add_argument('--data',    default=str(CAMPA_DIR), metavar='DIR',
                        help='Carpeta raíz con subject_XX/ (default: data/Data_29_subjects/Subjects)')
    parser.add_argument('--subject', default=None, metavar='DIR',
                        help='Sujeto único para inferencia sin etiquetas.')
    parser.add_argument('--model',   default='all',
                        choices=['RF', 'SVM', 'ANN', 'XGBoost', 'LDA', 'all'],
                        help='Modelo(s) a evaluar (default: all).')
    args = parser.parse_args()

    out_dir = RESULTS_DIR / 'campanella'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)

    if args.subject:
        _run_inference(args, out_dir)
    else:
        _run_validation(args, out_dir)


if __name__ == '__main__':
    main()
