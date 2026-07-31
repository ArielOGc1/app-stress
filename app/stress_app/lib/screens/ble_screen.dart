import 'dart:async';
import 'dart:convert';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../models/prediction_result.dart';
import '../services/ann_inference.dart';
import '../services/api_service.dart';
import '../services/ble_service.dart';

class BleScreen extends StatefulWidget {
  const BleScreen({super.key});

  @override
  State<BleScreen> createState() => _BleScreenState();
}

class _BleScreenState extends State<BleScreen> with SingleTickerProviderStateMixin {
  late final TabController _tabs;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabs.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Monitoreo en Tiempo Real'),
        backgroundColor: cs.secondary,
        foregroundColor: cs.onSecondary,
        bottom: TabBar(
          controller: _tabs,
          labelColor: cs.onSecondary,
          unselectedLabelColor: cs.onSecondary.withOpacity(0.6),
          indicatorColor: cs.onSecondary,
          tabs: const [
            Tab(icon: Icon(Icons.bluetooth), text: 'BLE Directo'),
            Tab(icon: Icon(Icons.play_circle_outline), text: 'Simulación'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabs,
        children: const [
          _BleDirectTab(),
          _SimulationTab(),
        ],
      ),
    );
  }
}

// ── Tab 1: BLE Directo ────────────────────────────────────────────────────────

class _BleDirectTab extends StatelessWidget {
  const _BleDirectTab();

  @override
  Widget build(BuildContext context) {
    final ble = context.watch<BleService>();
    final cs  = Theme.of(context).colorScheme;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // Estado + botón escaneo
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  Row(
                    children: [
                      Icon(Icons.bluetooth_searching_rounded,
                          color: cs.secondary, size: 28),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('Estado BLE',
                                style: Theme.of(context).textTheme.titleSmall),
                            Text(
                              _stateLabel(ble.state),
                              style: TextStyle(color: cs.secondary, fontSize: 13),
                            ),
                          ],
                        ),
                      ),
                      ElevatedButton.icon(
                        onPressed: ble.state == BleState.scanning
                            ? () => ble.stopScan()
                            : () => ble.startScan(),
                        icon: Icon(ble.state == BleState.scanning
                            ? Icons.stop_rounded
                            : Icons.search_rounded),
                        label: Text(ble.state == BleState.scanning
                            ? 'Detener'
                            : 'Escanear'),
                      ),
                    ],
                  ),
                  if (ble.error != null) ...[
                    const SizedBox(height: 8),
                    Text(ble.error!,
                        style: const TextStyle(color: Colors.red, fontSize: 12)),
                  ],
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),

          // Lista de dispositivos
          Expanded(
            child: ble.scanResults.isEmpty
                ? Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.bluetooth_disabled_rounded,
                            size: 64, color: cs.onSurface.withOpacity(0.2)),
                        const SizedBox(height: 12),
                        Text(
                          ble.state == BleState.scanning
                              ? 'Buscando dispositivos...'
                              : 'Presiona "Escanear" para buscar dispositivos E4.',
                          style: TextStyle(color: cs.onSurface.withOpacity(0.5)),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  )
                : ListView.builder(
                    itemCount: ble.scanResults.length,
                    itemBuilder: (_, i) {
                      final r    = ble.scanResults[i];
                      final name = r.device.platformName.isNotEmpty
                          ? r.device.platformName
                          : 'Dispositivo desconocido';
                      final isE4 = name.contains('E4') || name.contains('Empatica');
                      return ListTile(
                        leading: Icon(
                          Icons.bluetooth_rounded,
                          color: isE4 ? cs.secondary : cs.onSurface.withOpacity(0.4),
                        ),
                        title: Text(name),
                        subtitle: Text(r.device.remoteId.toString()),
                        trailing: isE4
                            ? const Chip(
                                label: Text('E4'),
                                visualDensity: VisualDensity.compact,
                              )
                            : null,
                      );
                    },
                  ),
          ),

          // Banner — hardware requerido
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.orange.shade50,
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: Colors.orange.shade200),
            ),
            child: Row(
              children: [
                Icon(Icons.hardware_rounded, color: Colors.orange.shade700, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Requiere Empatica E4 físico + E4 Streaming Server en PC. '
                    'Sin hardware, usa la pestaña "Simulación".',
                    style: TextStyle(fontSize: 12, color: Colors.orange.shade900),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  String _stateLabel(BleState s) => switch (s) {
        BleState.idle      => 'Inactivo',
        BleState.scanning  => 'Escaneando...',
        BleState.connected => 'Conectado',
        BleState.streaming => 'Transmitiendo',
        BleState.error     => 'Error',
      };
}

// ── Tab 2: Simulación ─────────────────────────────────────────────────────────

class _SimulationTab extends StatefulWidget {
  const _SimulationTab();

  @override
  State<_SimulationTab> createState() => _SimulationTabState();
}

class _SimulationTabState extends State<_SimulationTab> {
  String _selectedModel   = 'ANN';
  String _simMode         = 'demo';      // 'demo' | 'pc_stream'
  String _pcMode          = 'synthetic'; // 'synthetic' | 'subject'
  String _selectedSubject = 'S4';
  Timer? _sendTimer;
  int    _windowIdx = 0;
  List<Map<String, dynamic>> _demoWindows = [];

  bool   _hasLooped               = false;
  double _currentRecordingTimeSec = 0.0;

  static const _localModels  = ['ANN'];
  static const _serverModels = ['SVM', 'RF', 'ANN', 'XGBoost'];
  static const _wesadSubjects = [
    'S2','S3','S4','S5','S6','S7','S8',
    'S9','S10','S11','S13','S14','S15','S16','S17',
  ];

  // ── Carga las ventanas demo desde assets ─────────────────────────────────

  Future<void> _loadDemoWindows() async {
    if (_demoWindows.isNotEmpty) return;
    try {
      final raw = await rootBundle.loadString('assets/models/demo_$_selectedSubject.json');
      _demoWindows = List<Map<String, dynamic>>.from(jsonDecode(raw) as List);
    } catch (_) {
      _demoWindows = [];
    }
  }

  // ── Inicio de simulación ─────────────────────────────────────────────────

  Future<void> _startSimulation() async {
    final api = context.read<ApiService>();
    final ble = context.read<BleService>();

    // Streaming PC: siempre requiere servidor (botón deshabilitado si offline)
    if (_simMode == 'pc_stream') {
      final chan = api.openMonitorChannel(mode: _pcMode, sid: _selectedSubject);
      await ble.startSimulatedStream(channel: chan, selectedModel: _selectedModel);
      return;
    }

    // Demo local: siempre usa ANN local, sin servidor
    await _startOfflineSimulation(ble);
  }

  Future<void> _startOfflineSimulation(BleService ble) async {
    await _loadDemoWindows();
    await ble.startOfflineSimulation();

    final ann = AnnInference();
    await ann.initialize();

    const feats = [
      'scl_mean', 'scl_std', 'scr_count', 'scr_amp_mean', 'phasic_energy',
      'acc_mag_std', 'acc_movement',
      'hrv_mean_rr', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
      'temp_mean',
    ];

    _sendTimer = Timer.periodic(const Duration(seconds: 2), (_) {
      if (!ble.isStreaming || _demoWindows.isEmpty) return;
      final idx   = _windowIdx % _demoWindows.length;
      final w     = _demoWindows[idx];
      final vec   = feats.map((f) => (w[f] as num?)?.toDouble() ?? 0.0).toList();
      final proba = ann.predict(vec);
      final cls   = proba >= 0.5 ? 1 : 0;
      ble.addOfflinePrediction(StreamPrediction(
        predClass: cls,
        label:     cls == 1 ? 'stress' : 'no-stress',
        proba:     proba,
      ));
      final tSec = (w['t_start_s'] as num?)?.toDouble() ?? (idx * 30.0);
      if (mounted) setState(() {
        _windowIdx++;
        _currentRecordingTimeSec = tSec;
        if (_windowIdx % _demoWindows.length == 0) _hasLooped = true;
      });
    });
  }

  Future<void> _stopSimulation() async {
    _sendTimer?.cancel();
    final ble = context.read<BleService>();
    if (ble.isStreaming) {
      await ble.stopStream();
      await ble.stopOfflineSimulation();
    }
    setState(() {
      _windowIdx = 0;
      _hasLooped = false;
      _currentRecordingTimeSec = 0.0;
    });
  }

  @override
  void dispose() {
    _sendTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final api = context.watch<ApiService>();
    final ble = context.watch<BleService>();
    final cs  = Theme.of(context).colorScheme;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          // ── Modo de simulación ────────────────────────────────────────
          Card(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Modo', style: Theme.of(context).textTheme.titleSmall),
                  const SizedBox(height: 10),
                  SegmentedButton<String>(
                    segments: const [
                      ButtonSegment(
                        value: 'demo',
                        icon: Icon(Icons.play_circle_outline_rounded),
                        label: Text('Demo local'),
                      ),
                      ButtonSegment(
                        value: 'pc_stream',
                        icon: Icon(Icons.computer_rounded),
                        label: Text('Streaming PC'),
                      ),
                    ],
                    selected: {_simMode},
                    onSelectionChanged: ble.isStreaming
                        ? null
                        : (s) => setState(() => _simMode = s.first),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _simMode == 'demo'
                        ? 'Reproduce datos de un sujeto WESAD. Funciona sin PC.'
                        : 'El celular escucha mientras tu PC envía datos reales.',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(
                          color: cs.onSurface.withOpacity(0.6),
                        ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),

          // ── Controles ──────────────────────────────────────────────────
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  if (_simMode == 'demo') ...[
                    Row(
                      children: [
                        const Icon(Icons.model_training_rounded, size: 20),
                        const SizedBox(width: 8),
                        const Text('Modelo:'),
                        const SizedBox(width: 12),
                        Expanded(
                          child: DropdownButtonFormField<String>(
                            value: _localModels.contains(_selectedModel)
                                ? _selectedModel : 'ANN',
                            decoration: const InputDecoration(
                                border: OutlineInputBorder(), isDense: true),
                            items: _localModels
                                .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                                .toList(),
                            onChanged: ble.isStreaming
                                ? null
                                : (v) => setState(() => _selectedModel = v!),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    DropdownButtonFormField<String>(
                      value: _selectedSubject,
                      decoration: const InputDecoration(
                        labelText: 'Sujeto WESAD',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      items: _wesadSubjects
                          .map((s) => DropdownMenuItem(value: s, child: Text(s)))
                          .toList(),
                      onChanged: ble.isStreaming
                          ? null
                          : (v) => setState(() {
                                _selectedSubject = v!;
                                _demoWindows = [];
                              }),
                    ),
                    const SizedBox(height: 12),
                  ],
                  if (_simMode == 'pc_stream') ...[
                    // Modelo del servidor (SVM, RF, XGBoost, ANN, LDA)
                    Row(
                      children: [
                        const Icon(Icons.model_training_rounded, size: 20),
                        const SizedBox(width: 8),
                        const Text('Modelo:'),
                        const SizedBox(width: 12),
                        Expanded(
                          child: DropdownButtonFormField<String>(
                            value: _serverModels.contains(_selectedModel)
                                ? _selectedModel : 'SVM',
                            decoration: const InputDecoration(
                                border: OutlineInputBorder(), isDense: true),
                            items: _serverModels
                                .map((m) => DropdownMenuItem(value: m, child: Text(m)))
                                .toList(),
                            onChanged: (ble.isStreaming || !api.isConnected)
                                ? null
                                : (v) => setState(() => _selectedModel = v!),
                          ),
                        ),
                      ],
                    ),
                    Padding(
                      padding: const EdgeInsets.only(top: 4, bottom: 10),
                      child: Text(
                        'Todos los modelos corren en el servidor PC — '
                        'SVM, RF y XGBoost no están embebidos en la app.',
                        style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                      ),
                    ),
                    SegmentedButton<String>(
                      segments: const [
                        ButtonSegment(
                          value: 'synthetic',
                          icon: Icon(Icons.auto_awesome_rounded),
                          label: Text('Ruido gaussiano'),
                        ),
                        ButtonSegment(
                          value: 'subject',
                          icon: Icon(Icons.person_rounded),
                          label: Text('Sujeto'),
                        ),
                      ],
                      selected: {_pcMode},
                      onSelectionChanged: ble.isStreaming
                          ? null
                          : (s) => setState(() => _pcMode = s.first),
                    ),
                    if (_pcMode == 'subject') ...[
                      const SizedBox(height: 10),
                      DropdownButtonFormField<String>(
                        value: _selectedSubject,
                        decoration: const InputDecoration(
                          labelText: 'Sujeto WESAD',
                          border: OutlineInputBorder(),
                          isDense: true,
                        ),
                        items: _wesadSubjects
                            .map((s) => DropdownMenuItem(value: s, child: Text(s)))
                            .toList(),
                        onChanged: ble.isStreaming
                            ? null
                            : (v) => setState(() {
                                  _selectedSubject = v!;
                                  _demoWindows = [];
                                }),
                      ),
                    ],
                    const SizedBox(height: 10),
                  ],
                  ElevatedButton.icon(
                    onPressed: (!api.isConnected && _simMode == 'pc_stream')
                        ? null
                        : (ble.isStreaming ? _stopSimulation : _startSimulation),
                    icon: Icon(ble.isStreaming
                        ? Icons.stop_circle_outlined
                        : Icons.play_circle_outlined),
                    label: Text(ble.isStreaming ? 'Detener' : 'Iniciar simulación'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: ble.isStreaming
                          ? Colors.red.shade600
                          : cs.secondary,
                      foregroundColor: Colors.white,
                    ),
                  ),
                  if (_simMode == 'pc_stream' && !api.isConnected)
                    Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text(
                        'Requiere conexión al servidor',
                        style: TextStyle(
                            fontSize: 11, color: Colors.orange.shade700),
                        textAlign: TextAlign.center,
                      ),
                    ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),

          // ── Temporización: Demo local ─────────────────────────────────
          if (_simMode == 'demo')
            Card(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Temporización',
                        style: Theme.of(context).textTheme.titleSmall),
                    const SizedBox(height: 8),
                    _TimingRow(
                      icon: Icons.speed_rounded,
                      text: 'Velocidad ×15  —  1 s real = 15 s del registro',
                    ),
                    _TimingRow(
                      icon: Icons.timer_outlined,
                      text: 'Cada 2 s reales = 1 ventana = 30 s avanzados en el registro',
                    ),
                    _TimingRow(
                      icon: Icons.crop_free_rounded,
                      text: 'Ventana: 60 s del registro  |  solapamiento: 50%',
                    ),
                    if (ble.isStreaming && _demoWindows.isNotEmpty) ...[
                      const Divider(height: 16, thickness: 0.5),
                      _TimingRow(
                        icon: Icons.location_on_outlined,
                        text:
                            'Ventana ${(_windowIdx == 0 ? 1 : ((_windowIdx - 1) % _demoWindows.length) + 1)}'
                            ' / ${_demoWindows.length}'
                            '  →  t = ${(_currentRecordingTimeSec / 60).toStringAsFixed(1)} min del registro',
                        color: cs.secondary,
                        bold: true,
                      ),
                      if (_hasLooped)
                        _TimingRow(
                          icon: Icons.loop_rounded,
                          text: 'Bucle activo — la secuencia se reinicia desde el inicio',
                          color: Colors.orange.shade700,
                        ),
                    ],
                  ],
                ),
              ),
            ),

          // ── Temporización: Streaming PC ───────────────────────────────
          if (_simMode == 'pc_stream')
            Builder(builder: (ctx) {
              // Mide el intervalo real entre las últimas 2 predicciones del servidor
              double? measuredSec;
              if (ble.predictions.length >= 2) {
                final diff = ble.predictions.last.timestamp.difference(
                    ble.predictions[ble.predictions.length - 2].timestamp);
                measuredSec = diff.inMilliseconds / 1000.0;
              }
              return Card(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Temporización',
                          style: Theme.of(ctx).textTheme.titleSmall),
                      const SizedBox(height: 8),
                      _TimingRow(
                        icon: Icons.computer_rounded,
                        text: 'Velocidad controlada por el servidor PC',
                      ),
                      if (_pcMode == 'subject') ...[
                        _TimingRow(
                          icon: Icons.crop_free_rounded,
                          text: 'Ventana: 60 s del registro  |  solapamiento: 50%',
                        ),
                        _TimingRow(
                          icon: Icons.speed_rounded,
                          text: 'El servidor puede reproducir más rápido que ×15',
                        ),
                      ] else
                        _TimingRow(
                          icon: Icons.casino_outlined,
                          text: 'Ruido gaussiano generado en el servidor — sin referencia temporal',
                        ),
                      if (ble.isStreaming) ...[
                        const Divider(height: 16, thickness: 0.5),
                        _TimingRow(
                          icon: Icons.bar_chart_rounded,
                          text: 'Ventana ${ble.predictions.length} recibida del servidor',
                          color: cs.secondary,
                          bold: true,
                        ),
                        if (measuredSec != null)
                          _TimingRow(
                            icon: Icons.timer_outlined,
                            text: 'Intervalo medido: ${measuredSec.toStringAsFixed(2)} s/ventana'
                                '  (≈ ×${(30 / measuredSec.clamp(0.1, 300)).toStringAsFixed(0)} respecto al registro)',
                            color: cs.secondary,
                          ),
                      ],
                    ],
                  ),
                ),
              );
            }),
          const SizedBox(height: 8),

          // ── Predicción actual ─────────────────────────────────────────
          if (ble.lastPrediction != null) ...[
            _LivePredictionCard(prediction: ble.lastPrediction!),
            const SizedBox(height: 12),
          ],

          // ── Historial ─────────────────────────────────────────────────
          if (ble.predictions.isNotEmpty) ...[
            _StreamHistoryChart(predictions: ble.predictions),
            const SizedBox(height: 12),
            _SessionStatsCard(stats: ble.sessionStats),
          ],

          if (!ble.isStreaming && ble.predictions.isEmpty)
            Center(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 40),
                child: Column(
                  children: [
                    Icon(Icons.sensors_rounded,
                        size: 64, color: cs.onSurface.withOpacity(0.2)),
                    const SizedBox(height: 12),
                    Text(
                      'Inicia la simulación para ver predicciones en tiempo real.',
                      style: TextStyle(color: cs.onSurface.withOpacity(0.5)),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ── Widgets de la pantalla BLE ────────────────────────────────────────────────

class _LivePredictionCard extends StatelessWidget {
  final StreamPrediction prediction;
  const _LivePredictionCard({required this.prediction});

  @override
  Widget build(BuildContext context) {
    final isStress = prediction.isStress;
    final color    = isStress ? Colors.red.shade600 : Colors.green.shade600;
    final bg       = isStress ? Colors.red.shade50  : Colors.green.shade50;

    return Card(
      color: bg,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            Container(
              width: 64, height: 64,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: color.withOpacity(0.15),
              ),
              child: Icon(
                isStress ? Icons.warning_amber_rounded : Icons.self_improvement_rounded,
                color: color, size: 36,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    isStress ? 'ESTRÉS DETECTADO' : 'SIN ESTRÉS',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 18,
                      color: color,
                    ),
                  ),
                  if (prediction.proba != null)
                    Text(
                      'Confianza: ${(prediction.proba! * 100).toStringAsFixed(1)}%',
                      style: TextStyle(color: color.withOpacity(0.8)),
                    ),
                  Text(
                    'Última ventana — ${prediction.timestamp.hour}:${prediction.timestamp.minute.toString().padLeft(2, '0')}',
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StreamHistoryChart extends StatelessWidget {
  final List<StreamPrediction> predictions;
  const _StreamHistoryChart({required this.predictions});

  @override
  Widget build(BuildContext context) {
    final spots = predictions.asMap().entries.map((e) {
      final prob = e.value.proba ?? e.value.predClass.toDouble();
      return FlSpot(e.key.toDouble(), prob);
    }).toList();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Historial de probabilidad',
                style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 12),
            SizedBox(
              height: 120,
              child: LineChart(
                LineChartData(
                  minY: 0, maxY: 1,
                  clipData: const FlClipData.all(),
                  gridData: const FlGridData(show: false),
                  borderData: FlBorderData(show: false),
                  titlesData: const FlTitlesData(show: false),
                  lineBarsData: [
                    LineChartBarData(
                      spots: spots,
                      isCurved: true,
                      color: Colors.red.shade400,
                      barWidth: 2,
                      dotData: const FlDotData(show: false),
                      belowBarData: BarAreaData(
                        show: true,
                        gradient: LinearGradient(
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                          colors: [
                            Colors.red.shade300.withOpacity(0.4),
                            Colors.transparent,
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SessionStatsCard extends StatelessWidget {
  final Map<String, dynamic> stats;
  const _SessionStatsCard({required this.stats});

  @override
  Widget build(BuildContext context) {
    final pct    = (stats['stress_pct'] as double?)?.toStringAsFixed(1) ?? '0.0';
    final nTotal = stats['n_total']  ?? 0;
    final nStress= stats['n_stress'] ?? 0;
    final nRest  = stats['n_rest']   ?? 0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Estadísticas de sesión',
                style: Theme.of(context).textTheme.titleSmall),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _SmallStat(label: 'Total', value: '$nTotal'),
                _SmallStat(label: 'Estrés', value: '$nStress',
                    color: Colors.red.shade600),
                _SmallStat(label: 'Reposo', value: '$nRest',
                    color: Colors.green.shade600),
                _SmallStat(label: '% Estrés', value: '$pct%',
                    color: double.parse(pct) > 50
                        ? Colors.red.shade600
                        : Colors.green.shade600),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SmallStat extends StatelessWidget {
  final String label;
  final String value;
  final Color? color;
  const _SmallStat({required this.label, required this.value, this.color});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(value,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 18,
              color: color,
            )),
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.grey)),
      ],
    );
  }
}

class _TimingRow extends StatelessWidget {
  final IconData icon;
  final String   text;
  final Color?   color;
  final bool     bold;
  const _TimingRow({
    required this.icon,
    required this.text,
    this.color,
    this.bold = false,
  });

  @override
  Widget build(BuildContext context) {
    final c = color ?? Theme.of(context).colorScheme.onSurface.withOpacity(0.7);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 15, color: c),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                fontSize: 12,
                color: c,
                fontWeight: bold ? FontWeight.w600 : FontWeight.normal,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
