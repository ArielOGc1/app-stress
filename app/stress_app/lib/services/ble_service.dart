import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/prediction_result.dart';

enum BleState { idle, scanning, connected, streaming, error }

class BleService extends ChangeNotifier {
  BleState _state = BleState.idle;
  BluetoothDevice? _device;
  final List<ScanResult> _scanResults = [];
  final List<StreamPrediction> _predictions = [];
  String? _error;
  WebSocketChannel? _wsChannel;
  StreamSubscription? _wsSub;

  BleState get state => _state;
  BluetoothDevice? get device => _device;
  List<ScanResult> get scanResults => List.unmodifiable(_scanResults);
  List<StreamPrediction> get predictions => List.unmodifiable(_predictions);
  String? get error => _error;
  bool get isStreaming => _state == BleState.streaming;

  StreamPrediction? get lastPrediction =>
      _predictions.isEmpty ? null : _predictions.last;

  // ── Escaneo BLE ───────────────────────────────────────────────────────────

  Future<void> startScan() async {
    _scanResults.clear();
    _state = BleState.scanning;
    _error = null;
    notifyListeners();

    try {
      await FlutterBluePlus.startScan(timeout: const Duration(seconds: 10));
      FlutterBluePlus.scanResults.listen((results) {
        for (final r in results) {
          if (!_scanResults.any((s) => s.device.remoteId == r.device.remoteId)) {
            _scanResults.add(r);
            notifyListeners();
          }
        }
      });
      await Future.delayed(const Duration(seconds: 10));
      _state = BleState.idle;
      notifyListeners();
    } catch (e) {
      _error = 'Error de escaneo: $e';
      _state = BleState.error;
      notifyListeners();
    }
  }

  Future<void> stopScan() async {
    await FlutterBluePlus.stopScan();
    _state = BleState.idle;
    notifyListeners();
  }

  // ── Simulación vía WebSocket (modo online) ───────────────────────────────

  Future<void> startSimulatedStream({
    required WebSocketChannel channel,
    String selectedModel = 'SVM',
  }) async {
    _wsChannel  = channel;
    _predictions.clear();
    _state  = BleState.streaming;
    _error  = null;
    notifyListeners();

    _wsSub = channel.stream.listen(
      (data) {
        try {
          final json = jsonDecode(data as String) as Map<String, dynamic>;
          if (json['type'] == 'prediction') {
            _predictions.add(StreamPrediction.fromJson(json));
            if (_predictions.length > 120) _predictions.removeAt(0);
            notifyListeners();
          }
        } catch (_) {}
      },
      onError: (e) {
        _error = 'Error de stream: $e';
        _state = BleState.error;
        notifyListeners();
      },
      onDone: () {
        _state = BleState.idle;
        notifyListeners();
      },
    );
  }

  Future<void> stopStream() async {
    await _wsSub?.cancel();
    await _wsChannel?.sink.close();
    _wsChannel = null;
    _wsSub     = null;
    _state     = BleState.idle;
    notifyListeners();
  }

  // ── Simulación offline (modo demo sin servidor) ───────────────────────────

  Future<void> startOfflineSimulation() async {
    _predictions.clear();
    _state = BleState.streaming;
    _error = null;
    notifyListeners();
  }

  void addOfflinePrediction(StreamPrediction p) {
    _predictions.add(p);
    if (_predictions.length > 120) _predictions.removeAt(0);
    notifyListeners();
  }

  Future<void> stopOfflineSimulation() async {
    _state = BleState.idle;
    notifyListeners();
  }

  // ── Estadísticas de la sesión de streaming ────────────────────────────────

  Map<String, dynamic> get sessionStats {
    if (_predictions.isEmpty) return {};
    final nStress = _predictions.where((p) => p.isStress).length;
    final avgProba = _predictions
        .where((p) => p.proba != null)
        .map((p) => p.proba!)
        .fold(0.0, (a, b) => a + b) /
        _predictions.where((p) => p.proba != null).length;

    return {
      'n_total':      _predictions.length,
      'n_stress':     nStress,
      'n_rest':       _predictions.length - nStress,
      'stress_pct':   nStress / _predictions.length * 100,
      'avg_proba':    avgProba.isNaN ? 0.0 : avgProba,
    };
  }

  @override
  void dispose() {
    stopStream();
    super.dispose();
  }
}
