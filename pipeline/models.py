from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import ANN_PARAMS, RF_PARAMS, SVM_PARAMS, XGB_PARAMS

# SVM y ANN requieren RobustScaler (no invariantes a escala).
# RF y XGBoost son basados en árboles — no necesitan scaler.
# LDA: priors se calculan por fold en _fit_model; no necesita scaler.

_MODEL_TEMPLATES = {
    'RF':      RandomForestClassifier(**RF_PARAMS),
    'SVM':     Pipeline([
        ('scaler', RobustScaler()),
        ('clf', SVC(**SVM_PARAMS)),
    ]),
    'ANN':     Pipeline([
        ('scaler', RobustScaler()),
        ('clf', MLPClassifier(**ANN_PARAMS)),
    ]),
    'XGBoost': XGBClassifier(**XGB_PARAMS),
    'LDA':     LinearDiscriminantAnalysis(),
}


def get_models():
    """
    Retorna un dict con clones frescos de cada modelo.
    Siempre clonar antes de entrenar para evitar estado residual entre folds.
    """
    return {name: clone(m) for name, m in _MODEL_TEMPLATES.items()}


def clone_model(name):
    """Retorna un clon fresco del modelo indicado."""
    return clone(_MODEL_TEMPLATES[name])
