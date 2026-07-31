import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight

from .metrics import compute_fold_metrics

# Origen de cada modelo para la columna 'origen' del CSV de resultados.
MODEL_ORIGINS = {
    'RF':      'propuesto',
    'SVM':     'propuesto',
    'ANN':     'propuesto',
    'XGBoost': 'linea_base_interna',
    'LDA':     'literatura_schmidt2018',
}


def _fit_model(model_name, model, X_train, y_train):
    """
    Entrena el modelo manejando el desbalance de clases.
    - RF y SVM: class_weight='balanced' recalculado internamente por sklearn en cada fold.
    - ANN: sample_weight calculado desde y_train del fold via fit_params.
    - XGBoost: scale_pos_weight calculado dinámicamente desde y_train del fold.
    - LDA: priors fijados a la proporción de clases del fold de entrenamiento.
    """
    if model_name == 'ANN':
        sw = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, clf__sample_weight=sw)
    elif model_name == 'XGBoost':
        n_pos = (y_train == 1).sum()
        ratio = (y_train == 0).sum() / n_pos if n_pos > 0 else 1.0
        model.set_params(scale_pos_weight=ratio)
        model.fit(X_train, y_train)
    elif model_name == 'LDA':
        classes, counts = np.unique(y_train, return_counts=True)
        model.set_params(priors=counts / counts.sum())
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model


def run_loso(X, y, groups, models):
    """
    Ejecuta validación Leave-One-Subject-Out.

    Parameters
    ----------
    X      : ndarray (n_windows, n_features)
    y      : ndarray (n_windows,)
    groups : ndarray (n_windows,) — subject IDs
    models : dict  {name: estimator}  — instancias FRESCAS (serán clonadas por fold)

    Returns
    -------
    DataFrame con una fila por (modelo × fold), columnas de métricas.
    """
    logo = LeaveOneGroupOut()
    records = []

    for model_name, model_template in models.items():
        print(f"\n  [{model_name}] ", end='', flush=True)

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_subject = groups[test_idx[0]]
            print(f".", end='', flush=True)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Skip fold si entrenamiento tiene una sola clase
            if len(np.unique(y_train)) < 2:
                warnings.warn(
                    f"[LOSO] Fold {fold_idx} (sujeto {test_subject}) ignorado: "
                    "el conjunto de entrenamiento tiene una sola clase.",
                    UserWarning, stacklevel=2,
                )
                continue

            model = clone(model_template)
            model = _fit_model(model_name, model, X_train, y_train)

            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            fold_metrics = compute_fold_metrics(y_test, y_pred, y_proba)
            fold_metrics.update({
                'model':        model_name,
                'fold':         fold_idx,
                'test_subject': test_subject,
                'n_train':      len(y_train),
                'n_test':       len(y_test),
                'origen':       MODEL_ORIGINS.get(model_name, 'propuesto'),
            })
            records.append(fold_metrics)

        print()  # newline al terminar el modelo

    return pd.DataFrame(records)


def run_loso_full(X, y, groups, models):
    """
    Como run_loso() pero guarda y_true, y_pred, y_proba por fold.
    Necesario para generar curvas ROC y matrices de confusión.
    """
    logo = LeaveOneGroupOut()
    records = []

    for model_name, model_template in models.items():
        print(f"\n  [{model_name}] ", end='', flush=True)

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            test_subject = groups[test_idx[0]]
            print(".", end='', flush=True)

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                warnings.warn(
                    f"[LOSO] Fold {fold_idx} (sujeto {test_subject}) ignorado: "
                    "entrenamiento con una sola clase.",
                    UserWarning, stacklevel=2,
                )
                continue

            model = clone(model_template)
            model = _fit_model(model_name, model, X_train, y_train)

            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            fold_metrics = compute_fold_metrics(y_test, y_pred, y_proba)
            fold_metrics.update({
                'model':        model_name,
                'fold':         fold_idx,
                'test_subject': test_subject,
                'n_train':      len(y_train),
                'n_test':       len(y_test),
                'origen':       MODEL_ORIGINS.get(model_name, 'propuesto'),
                'y_true':       y_test,
                'y_pred':       y_pred,
                'y_proba':      y_proba,
            })
            records.append(fold_metrics)

        print()

    return pd.DataFrame(records)
