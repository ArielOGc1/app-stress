"""
Pipeline principal: LOSO con RF, SVM, ANN, XGBoost, LDA.

Uso (WESAD):
    conda run -n tesis python scripts/run_pipeline.py

Uso con dataset externo:
    conda run -n tesis python scripts/run_pipeline.py --csv ruta.csv --name campanella
    → guarda en results/campanella/ sin tocar results/
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import RESULTS_DIR
from pipeline.data import dataset_summary, load_dataset
from pipeline.loso import MODEL_ORIGINS, run_loso_full
from pipeline.metrics import aggregate_results, print_summary
from pipeline.models import get_models, clone_model
from pipeline.plots import plot_metrics_boxplots, plot_roc_curves, plot_confusion_matrices
from pipeline.shap_analysis import compute_shap_xgboost, plot_shap_bar, plot_shap_summary


def parse_args():
    parser = argparse.ArgumentParser(description='Pipeline LOSO de clasificación de estrés.')
    parser.add_argument('--csv',  default=None, metavar='RUTA',
                        help='CSV de entrada (default: WESAD normalizado).')
    parser.add_argument('--name', default=None, metavar='NOMBRE',
                        help='Nombre del dataset → subcarpeta en results/.')
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = RESULTS_DIR / args.name if args.name else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)

    dataset_label = args.name.upper() if args.name else 'WESAD'
    print('=' * 60)
    print(f'  PIPELINE {dataset_label} — Clasificación de Estrés (LOSO)')
    print('=' * 60)
    print(f'  Salida: {out_dir}')

    # 1. Cargar dataset
    print('\n[1] Cargando dataset normalizado...')
    strict = args.csv is None
    X, y, groups, features = load_dataset(path=args.csv, strict=strict)
    dataset_summary(X, y, groups, features)

    # 2. LOSO
    n_subjects = len(set(groups))
    print(f'\n[2] Ejecutando LOSO ({n_subjects} folds × 5 modelos)...')
    models  = get_models()
    fold_df = run_loso_full(X, y, groups, models)

    # 3. Guardar resultados por fold
    metric_cols = ['model', 'origen', 'fold', 'test_subject', 'accuracy',
                   'balanced_accuracy', 'f1_macro', 'f1_weighted',
                   'precision_stress', 'recall_stress', 'roc_auc']
    save_cols = [c for c in metric_cols if c in fold_df.columns]
    fold_csv  = out_dir / 'loso_fold_results.csv'
    fold_df[save_cols].to_csv(fold_csv, index=False)
    print(f'\n  Resultados por fold guardados: {fold_csv}')

    # 4. Resumen
    print('\n[3] Resumen LOSO:')
    summary_df = aggregate_results(fold_df[save_cols])
    print_summary(summary_df)
    summary_df['origen'] = [MODEL_ORIGINS.get(m, 'propuesto') for m in summary_df.index]
    summary_df.to_csv(out_dir / 'loso_summary.csv')

    # 5. Visualizaciones
    print('\n[4] Generando figuras...')
    plot_metrics_boxplots(fold_df, out_dir)
    plot_roc_curves(fold_df, out_dir)
    plot_confusion_matrices(fold_df, out_dir)

    # 6. SHAP (XGBoost)
    print('\n[5] Análisis SHAP (XGBoost)...')
    xgb = clone_model('XGBoost')
    n_pos = (y == 1).sum()
    xgb.set_params(scale_pos_weight=(y == 0).sum() / n_pos if n_pos > 0 else 1.0)
    shap_values, _ = compute_shap_xgboost(X, y, features, xgb)
    plot_shap_summary(shap_values, X, features, results_dir=out_dir)
    importance_df = plot_shap_bar(shap_values, features, results_dir=out_dir)
    print('\n  Top features por SHAP:')
    print(importance_df.head(10).to_string(index=False))
    importance_df.to_csv(out_dir / 'shap_importance.csv', index=False)

    print('\n' + '=' * 60)
    print(f'  Pipeline completado. Resultados en: {out_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
