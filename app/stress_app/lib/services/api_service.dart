import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:web_socket_channel/web_socket_channel.dart';

import '../models/prediction_result.dart';

class ApiService extends ChangeNotifier {
  String _baseUrl = 'http://localhost:8000';
  bool _isConnected = false;
  bool _isLoading = false;
  String? _lastError;

  Future<void> loadSavedUrl() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString('api_url');
    if (saved != null && saved.isNotEmpty) {
      _baseUrl = saved;
      notifyListeners();
    }
  }

  String get baseUrl => _baseUrl;
  bool get isConnected => _isConnected;
  bool get isLoading => _isLoading;
  String? get lastError => _lastError;

  void setBaseUrl(String url) {
    _baseUrl = url.endsWith('/') ? url.substring(0, url.length - 1) : url;
    notifyListeners();
  }

  // ── Health check ──────────────────────────────────────────────────────────

  Future<bool> checkHealth() async {
    try {
      final resp = await http
          .get(Uri.parse('$_baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      _isConnected = resp.statusCode == 200;
      _lastError   = _isConnected ? null : 'Servidor no responde';
      notifyListeners();
      return _isConnected;
    } catch (e) {
      _isConnected = false;
      _lastError   = 'No se puede conectar: $e';
      notifyListeners();
      return false;
    }
  }

  // ── Predicción desde archivos CSV ─────────────────────────────────────────

  Future<PredictionResult> predictFromFiles({
    required File edaFile,
    required File bvpFile,
    required File accFile,
    required File tempFile,
    String model = 'SVM',
  }) async {
    _isLoading  = true;
    _lastError  = null;
    notifyListeners();

    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/predict/csv'),
      );

      request.files.addAll([
        await http.MultipartFile.fromPath('eda',  edaFile.path),
        await http.MultipartFile.fromPath('bvp',  bvpFile.path),
        await http.MultipartFile.fromPath('acc',  accFile.path),
        await http.MultipartFile.fromPath('temp', tempFile.path),
      ]);
      request.fields['model'] = model;

      final streamed = await request.send()
          .timeout(const Duration(seconds: 30));
      final body = await streamed.stream.bytesToString()
          .timeout(const Duration(seconds: 90));

      if (streamed.statusCode != 200) {
        final err = jsonDecode(body);
        throw Exception(err['detail'] ?? 'Error del servidor');
      }

      return PredictionResult.fromJson(jsonDecode(body) as Map<String, dynamic>);
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  // ── WebSocket para streaming ───────────────────────────────────────────────

  WebSocketChannel openStreamChannel() {
    final wsUrl = _baseUrl
        .replaceFirst('http://', 'ws://')
        .replaceFirst('https://', 'wss://');
    return WebSocketChannel.connect(Uri.parse('$wsUrl/ws/stream'));
  }

  /// Celular solo escucha — el PC streamer.py envía datos a /ws/stream.
  WebSocketChannel openMonitorChannel({String mode = 'synthetic', String sid = 'S4'}) {
    final wsUrl = _baseUrl
        .replaceFirst('http://', 'ws://')
        .replaceFirst('https://', 'wss://');
    return WebSocketChannel.connect(Uri.parse('$wsUrl/ws/monitor?mode=$mode&sid=$sid'));
  }
}
