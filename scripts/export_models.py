"""
Exporta los modelos entrenados sobre el dataset WESAD completo.

Salida en models/:
  RF.joblib, SVM.joblib, ANN.joblib, XGBoost.joblib  ← pipelines sklearn completos
  ann_weights.json                                    ← pesos para conversión TFLite
  model_metadata.json                                 ← features, clases, hiperparámetros

Uso:
    conda run -n tesis python scripts/export_models.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.config import FEATURES_SELECTED, RANDOM_STATE, RESULTS_DIR
from pipeline.data import dataset_summary, load_dataset
from pipeline.loso import _fit_model
from pipeline.models import get_models

MODELS_DIR = Path(__file__).parent.parent / "models"


def train_full_dataset(X, y):
    """Entrena cada modelo sobre el dataset completo."""
    models = get_models()
    fitted = {}
    for name, template in models.items():
        print(f"  Entrenando {name}...", end=" ", flush=True)
        model = clone(template)
        model = _fit_model(name, model, X, y)
        fitted[name] = model
        print("ok")
    return fitted


def extract_ann_weights(ann_pipeline):
    """
    Extrae pesos del Pipeline(RobustScaler + MLPClassifier) a un dict JSON-serializable.
    Permite reconstruir la ANN en TensorFlow/Keras sin sklearn.
    """
    scaler = ann_pipeline.named_steps['scaler']
    mlp    = ann_pipeline.named_steps['clf']

    layers = []
    for W, b in zip(mlp.coefs_, mlp.intercepts_):
        layers.append({
            'weights': W.tolist(),
            'biases':  b.tolist(),
        })

    return {
        'scaler': {
            'center': scaler.center_.tolist(),
            'scale':  scaler.scale_.tolist(),
        },
        'hidden_layer_sizes': list(mlp.hidden_layer_sizes),
        'activation':         mlp.activation,          # 'relu'
        'output_activation':  mlp.out_activation_,     # 'logistic' para binario
        'n_features_in':      int(mlp.n_features_in_),
        'layers':             layers,
    }


def build_metadata(features, fitted_models):
    """Genera el JSON de metadatos para documentación y app móvil."""
    ann = fitted_models['ANN'].named_steps['clf']
    svm_clf = fitted_models['SVM'].named_steps['clf']

    return {
        'features': features,
        'n_features': len(features),
        'classes': {0: 'baseline', 1: 'stress'},
        'normalization': 'z-score intra-subject (aplicado antes del entrenamiento)',
        'models': {
            'RF': {
                'type': 'RandomForestClassifier',
                'n_estimators': 200,
                'class_weight': 'balanced',
            },
            'SVM': {
                'type': 'SVC (rbf, C=1.0) con RobustScaler',
                'kernel': 'rbf',
                'C': svm_clf.C,
                'gamma': 'scale',
            },
            'ANN': {
                'type': 'MLPClassifier con RobustScaler',
                'hidden_layer_sizes': list(ann.hidden_layer_sizes),
                'activation': ann.activation,
                'n_iter': ann.n_iter_,
                'best_validation_score': round(float(ann.best_validation_score_), 4),
            },
            'XGBoost': {
                'type': 'XGBClassifier',
                'n_estimators': 200,
                'max_depth': 4,
            },
        },
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  EXPORTACIÓN DE MODELOS — WESAD (dataset completo)")
    print("=" * 60)

    # 1. Cargar dataset
    print("\n[1] Cargando dataset normalizado...")
    X, y, groups, features = load_dataset()
    dataset_summary(X, y, groups, features)

    # 2. Entrenar sobre dataset completo
    print("\n[2] Entrenando modelos sobre dataset completo...")
    fitted = train_full_dataset(X, y)

    # 3. Exportar .joblib
    print("\n[3] Guardando modelos (.joblib)...")
    for name, model in fitted.items():
        out = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, out)
        size_kb = out.stat().st_size / 1024
        print(f"  {name:10s} → {out.name}  ({size_kb:.1f} KB)")

    # 4. Extraer pesos ANN para TFLite
    print("\n[4] Extrayendo pesos ANN para conversión TFLite...")
    weights = extract_ann_weights(fitted['ANN'])
    weights_path = MODELS_DIR / "ann_weights.json"
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f"  Guardado: {weights_path}")
    print(f"  Capas: {[len(l['weights'][0]) for l in weights['layers']]} unidades por capa")

    # 5. Metadatos
    print("\n[5] Guardando metadatos...")
    meta = build_metadata(features, fitted)
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Guardado: {meta_path}")

    # 6. Verificación rápida
    print("\n[6] Verificación de coherencia...")
    for name, model in fitted.items():
        y_pred = model.predict(X)
        ba = np.mean([
            np.mean(y_pred[y == c] == c) for c in np.unique(y)
        ])
        print(f"  {name:10s}  BA (train, referencia) = {ba:.3f}")

    print("\n" + "=" * 60)
    print(f"  Modelos exportados en: {MODELS_DIR}")
    print("=" * 60)
    print("\nPróximo paso — TFLite:")
    print("  conda run -n tesis pip install tensorflow-cpu")
    print("  conda run -n tesis python scripts/export_tflite.py")


if __name__ == "__main__":
    main()
