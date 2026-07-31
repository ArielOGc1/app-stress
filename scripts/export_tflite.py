"""
Verificación de los modelos TFLite vs joblib (ANN sklearn).
Usa ai-edge-litert (instalado en el env tesis).

Requisito previo:
    conda run -n tflite python scripts/convert_tflite.py

Uso:
    conda run -n tesis python scripts/export_tflite.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.data import load_dataset

MODELS_DIR = Path(__file__).parent.parent / "models"


def run_tflite(tflite_path, X_sample):
    """Inferencia con ai-edge-litert."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']

    probas = []
    for i in range(len(X_sample)):
        interp.set_tensor(inp_idx, X_sample[i:i+1].astype(np.float32))
        interp.invoke()
        probas.append(float(interp.get_tensor(out_idx)[0, 0]))
    return np.array(probas)


def main():
    print("=" * 60)
    print("  VERIFICACIÓN TFLite vs joblib (ANN)")
    print("=" * 60)

    fp32_path = MODELS_DIR / "ANN.tflite"
    int8_path = MODELS_DIR / "ANN_int8.tflite"

    for p in (fp32_path, int8_path):
        if not p.exists():
            print(f"\n  FALTA: {p}")
            print("  Ejecuta primero:  conda run -n tflite python scripts/convert_tflite.py")
            sys.exit(1)

    print("\n[1] Cargando dataset y predicciones sklearn...")
    X, y, _, _ = load_dataset()
    ann_pipeline      = joblib.load(MODELS_DIR / "ANN.joblib")
    y_proba_sklearn   = ann_pipeline.predict_proba(X)[:, 1]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), 50, replace=False)
    X_sample   = X[idx]
    ref_probas = y_proba_sklearn[idx]

    print(f"  {len(X)} ventanas totales  |  Verificando sobre {len(idx)} muestras")

    print("\n[2] Verificando ANN.tflite (fp32)...")
    probas_fp32  = run_tflite(fp32_path, X_sample)
    errors_fp32  = np.abs(probas_fp32 - ref_probas)
    print(f"  Error máx  : {errors_fp32.max():.6f}")
    print(f"  Error medio: {errors_fp32.mean():.6f}")
    status_fp32 = "OK ✓" if errors_fp32.max() < 1e-4 else "REVISAR ⚠"
    print(f"  Estado     : {status_fp32}")

    print("\n[3] Verificando ANN_int8.tflite (int8)...")
    probas_int8  = run_tflite(int8_path, X_sample)
    errors_int8  = np.abs(probas_int8 - ref_probas)
    print(f"  Error máx  : {errors_int8.max():.6f}")
    print(f"  Error medio: {errors_int8.mean():.6f}")
    status_int8 = "OK ✓" if errors_int8.max() < 0.05 else "REVISAR ⚠"
    print(f"  Estado     : {status_int8}")

    print("\n[4] Tamaños de modelo:")
    print(f"  ANN.joblib       : {(MODELS_DIR/'ANN.joblib').stat().st_size/1024:.1f} KB")
    print(f"  ANN.tflite       : {fp32_path.stat().st_size/1024:.1f} KB")
    print(f"  ANN_int8.tflite  : {int8_path.stat().st_size/1024:.1f} KB")

    print("\n" + "=" * 60)
    print(f"  fp32 {status_fp32}   int8 {status_int8}")
    print("=" * 60)


if __name__ == "__main__":
    main()
