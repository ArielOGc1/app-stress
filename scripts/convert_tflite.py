"""
Conversión ANN → TFLite (fp32 e int8).
Requiere Python ≤ 3.12 con tensorflow instalado.

Env dedicado (una sola vez):
    conda create -n tflite python=3.11 -y
    conda run -n tflite pip install tensorflow-cpu numpy

Uso:
    conda run -n tflite python scripts/convert_tflite.py

Entradas (models/):
    ann_weights.json      ← generado por export_models.py
    calibration_data.npy  ← generado por export_models.py o export_tflite.py

Salidas (models/):
    ANN.tflite        ← fp32, ~94 KB
    ANN_int8.tflite   ← int8 post-training, ~30 KB
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

MODELS_DIR = Path(__file__).parent.parent / "models"

print(f"TensorFlow {tf.__version__}  |  Python {sys.version.split()[0]}")
print("=" * 60)
print("  CONVERSIÓN TFLite — ANN (stress classifier)")
print("=" * 60)


def build_keras_model(weights_path):
    with open(weights_path) as f:
        w = json.load(f)

    scaler_center = np.array(w['scaler']['center'], dtype=np.float32)
    scaler_scale  = np.array(w['scaler']['scale'],  dtype=np.float32)
    n_features    = w['n_features_in']

    inputs = tf.keras.Input(shape=(n_features,), name='features')
    x = (inputs - scaler_center) / scaler_scale     # RobustScaler baked-in

    for i, layer_w in enumerate(w['layers'][:-1]):
        W = np.array(layer_w['weights'], dtype=np.float32)
        b = np.array(layer_w['biases'],  dtype=np.float32)
        dense = tf.keras.layers.Dense(b.shape[0], activation='relu', name=f'hidden_{i+1}')
        x = dense(x)
        dense.set_weights([W, b])

    out_w = w['layers'][-1]
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    outputs = output_layer(x)
    output_layer.set_weights([
        np.array(out_w['weights'], dtype=np.float32),
        np.array(out_w['biases'],  dtype=np.float32),
    ])

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='ANN_stress')


def convert_fp32(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    return len(tflite_bytes)


def convert_int8(model, calibration_data, out_path):
    def representative_dataset():
        for i in range(len(calibration_data)):
            yield [calibration_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    return len(tflite_bytes)


def main():
    weights_path = MODELS_DIR / "ann_weights.json"
    calib_path   = MODELS_DIR / "calibration_data.npy"

    for p in (weights_path, calib_path):
        if not p.exists():
            raise FileNotFoundError(f"Falta: {p}")

    print("\n[1] Construyendo modelo Keras desde ann_weights.json...")
    model = build_keras_model(weights_path)
    model.summary()

    calib_data = np.load(calib_path)
    print(f"\n[2] Datos de calibración: {calib_data.shape}")

    print("\n[3] Exportando ANN.tflite (fp32)...")
    fp32_path = MODELS_DIR / "ANN.tflite"
    size_fp32 = convert_fp32(model, fp32_path)
    print(f"  Guardado: {fp32_path}  ({size_fp32/1024:.1f} KB)")

    print("\n[4] Exportando ANN_int8.tflite (post-training int8)...")
    int8_path = MODELS_DIR / "ANN_int8.tflite"
    size_int8 = convert_int8(model, calib_data, int8_path)
    print(f"  Guardado: {int8_path}  ({size_int8/1024:.1f} KB)")

    print("\n" + "=" * 60)
    print(f"  fp32 : {size_fp32/1024:.1f} KB   →  {fp32_path.name}")
    print(f"  int8 : {size_int8/1024:.1f} KB   →  {int8_path.name}")
    print("  Conversión completada. Verificar con: scripts/verify_tflite.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
