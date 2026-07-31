import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_fold_metrics(y_true, y_pred, y_proba):
    """Calcula métricas para un fold LOSO.

    roc_auc devuelve NaN si el fold de test contiene una sola clase
    (no es computable). Esto puede ocurrir con datasets pequeños.
    """
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = np.nan

    return {
        'accuracy':           accuracy_score(y_true, y_pred),
        'balanced_accuracy':  balanced_accuracy_score(y_true, y_pred),
        'f1_macro':           f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted':        f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_stress':   precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall_stress':      recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'roc_auc':            roc_auc,
    }


def aggregate_results(results_df):
    """
    Agrega resultados por fold en media ± std.

    Returns
    -------
    DataFrame indexado por modelo con columnas '<metric>_mean' y '<metric>_std'.
    """
    metric_cols = [
        'accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
        'precision_stress', 'recall_stress', 'roc_auc',
    ]
    rows = []
    for model_name, grp in results_df.groupby('model'):
        row = {'model': model_name}
        for col in metric_cols:
            row[f'{col}_mean'] = grp[col].mean()
            row[f'{col}_std']  = grp[col].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index('model')


def print_summary(summary_df, decimals=3):
    """Imprime la tabla de resumen formateada."""
    metrics = ['balanced_accuracy', 'f1_macro', 'f1_weighted', 'roc_auc', 'precision_stress', 'recall_stress']
    header = f"{'Modelo':<10}" + "".join(f"  {m:<26}" for m in metrics)
    print(header)
    print("-" * len(header))
    for model, row in summary_df.iterrows():
        vals = "".join(
            f"  {row[f'{m}_mean']:.{decimals}f} ± {row[f'{m}_std']:.{decimals}f}     "
            for m in metrics
        )
        print(f"{model:<10}{vals}")
