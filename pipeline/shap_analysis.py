import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from .config import RESULTS_DIR


def compute_shap_xgboost(X, y, feature_names, xgb_template):
    """
    Entrena XGBoost sobre el dataset completo y calcula SHAP values.
    Uso: interpretación global del modelo, no evaluación de performance.

    Parameters
    ----------
    X             : ndarray (n_windows, n_features)
    y             : ndarray (n_windows,)
    feature_names : list[str]
    xgb_template  : estimador XGBClassifier (será clonado)

    Returns
    -------
    shap_values : ndarray (n_windows, n_features)
    explainer   : shap.TreeExplainer
    """
    model = clone(xgb_template)
    model.fit(X, y)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values, explainer


def plot_shap_summary(shap_values, X, feature_names, results_dir=None, save=True):
    """Beeswarm plot — importancia y dirección de cada feature."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.title("SHAP — XGBoost (entrenado en dataset completo)")
    plt.tight_layout()
    if save:
        out = results_dir / "figures" / "shap_summary.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {out}")
    plt.close()


def plot_shap_bar(shap_values, feature_names, results_dir=None, save=True):
    """Bar plot — importancia media absoluta por feature."""
    if results_dir is None:
        results_dir = RESULTS_DIR
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({'feature': feature_names, 'importance': mean_abs})
        .sort_values('importance', ascending=True)
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance — XGBoost")
    plt.tight_layout()
    if save:
        out = results_dir / "figures" / "shap_bar.png"
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Guardado: {out}")
    plt.close()

    return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
