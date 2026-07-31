import 'dart:convert';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// Inferencia ANN mediante TFLite (ANN_int8.tflite).
/// Fallback automático a forward pass Dart puro si TFLite no está disponible
/// (ej. plataforma web durante desarrollo).
class AnnInference {
  // TFLite
  Interpreter? _interpreter;
  bool _usingTflite = false;

  // Fallback Dart puro
  List<double>? _scalerCenter;
  List<double>? _scalerScale;
  List<List<double>>? _w0, _w1, _wOut;
  List<double>? _b0, _b1, _bOut;

  bool _initialized = false;

  static final AnnInference _instance = AnnInference._internal();
  AnnInference._internal();
  factory AnnInference() => _instance;

  bool get isInitialized => _initialized;
  bool get usingTflite   => _usingTflite;

  Future<void> initialize() async {
    if (_initialized) return;
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/ANN_int8.tflite',
      );
      _usingTflite = true;
    } catch (_) {
      await _loadDartPure();
    }
    _initialized = true;
  }

  /// Devuelve probabilidad de estrés (0–1).
  /// Entrada: 12 features con z-score intra-sujeto aplicado.
  double predict(List<double> features) {
    assert(_initialized, 'Llama initialize() primero');
    assert(features.length == 12);
    return _usingTflite ? _predictTflite(features) : _predictDart(features);
  }

  // ── TFLite ──────────────────────────────────────────────────────────────────

  double _predictTflite(List<double> features) {
    try {
      final input  = [features.toList()];           // shape [1, 12]
      final output = [List<double>.filled(1, 0.0)]; // shape [1, 1]
      _interpreter!.run(input, output);
      return (output[0][0] as num).toDouble().clamp(0.0, 1.0);
    } catch (_) {
      // TFLite falló (dispositivo incompatible / cuantización) — usar Dart puro
      return _predictDart(features);
    }
  }

  // ── Dart puro (fallback) ─────────────────────────────────────────────────────

  Future<void> _loadDartPure() async {
    final raw = await rootBundle.loadString('assets/models/ann_weights.json');
    final w   = jsonDecode(raw) as Map<String, dynamic>;

    _scalerCenter = _toDoubleList(w['scaler']['center'] as List);
    _scalerScale  = _toDoubleList(w['scaler']['scale']  as List);
    _w0   = _toMatrix(w['layers'][0]['weights']);
    _b0   = _toDoubleList(w['layers'][0]['biases'] as List);
    _w1   = _toMatrix(w['layers'][1]['weights']);
    _b1   = _toDoubleList(w['layers'][1]['biases'] as List);
    _wOut = _toMatrix(w['layers'][2]['weights']);
    _bOut = _toDoubleList(w['layers'][2]['biases'] as List);
  }

  double _predictDart(List<double> features) {
    final x  = List.generate(12, (i) =>
        (features[i] - _scalerCenter![i]) / _scalerScale![i]);
    final h1  = _dense(x,  _w0!,   _b0!,  relu: true);
    final h2  = _dense(h1, _w1!,   _b1!,  relu: true);
    final out = _dense(h2, _wOut!, _bOut!, relu: false);
    return 1.0 / (1.0 + exp(-out[0]));
  }

  List<double> _dense(List<double> x, List<List<double>> W,
      List<double> b, {required bool relu}) {
    final nOut = b.length;
    final out  = List<double>.filled(nOut, 0.0);
    for (int j = 0; j < nOut; j++) {
      double s = b[j];
      for (int i = 0; i < x.length; i++) s += x[i] * W[i][j];
      out[j] = relu ? (s > 0 ? s : 0.0) : s;
    }
    return out;
  }

  List<double> _toDoubleList(List raw) =>
      raw.map((v) => (v as num).toDouble()).toList();

  List<List<double>> _toMatrix(dynamic raw) =>
      (raw as List).map((row) => _toDoubleList(row as List)).toList();
}
