import 'dart:async';
import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/foundation.dart' show compute;
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path_provider/path_provider.dart';

import '../models/prediction_result.dart';
import '../services/ann_inference.dart';
import '../services/feature_extractor.dart';
import '../services/notification_service.dart';

// Top-level: requerido por compute() para correr en isolate separado
Future<List<Map<String, double>>> _extractWindowsTask(
    List<String> paths) async {
  return FeatureExtractor.extractWindows(
    edaFile:  File(paths[0]),
    bvpFile:  File(paths[1]),
    accFile:  File(paths[2]),
    tempFile: File(paths[3]),
  );
}

// ── Resultado de un sujeto (para modo dataset) ─────────────────────────────

class _SubjectResult {
  final String subjectId;
  final PredictionResult ann;
  _SubjectResult({required this.subjectId, required this.ann});
}

// ─────────────────────────────────────────────────────────────────────────────

class CsvScreen extends StatefulWidget {
  const CsvScreen({super.key});

  @override
  State<CsvScreen> createState() => _CsvScreenState();
}

class _CsvScreenState extends State<CsvScreen>
    with SingleTickerProviderStateMixin {

  late final TabController _tabs;

  // ── Modo sujeto ────────────────────────────────────────────────────────────
  final Map<String, File?> _files = {
    'EDA': null, 'BVP': null, 'ACC': null, 'TEMP': null,
  };
  bool get _allFilesSelected => _files.values.every((f) => f != null);
  PredictionResult? _annResult;

  // ── Modo dataset ───────────────────────────────────────────────────────────
  File? _zipFile;
  List<_SubjectResult> _datasetResults = [];
  bool _processingDataset = false;
  String _datasetStatus = '';

  // ── Shared ─────────────────────────────────────────────────────────────────
  bool   _analyzing       = false;
  double _analyzeProgress = 0.0;
  String _analyzeStatus   = '';

  // ── Cancel ─────────────────────────────────────────────────────────────────
  bool   _cancelRequested = false;
  Timer? _elapsedTimer;
  int    _elapsedSecs    = 0;

  @override
  void initState() {
    super.initState();
    _tabs = TabController(length: 2, vsync: this);
  }

  @override
  void dispose() {
    _tabs.dispose();
    _elapsedTimer?.cancel();
    super.dispose();
  }

  // ── Sujeto individual ──────────────────────────────────────────────────────

  Future<void> _pickFile(String signal) async {
    final r = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['csv'],
      dialogTitle: 'Selecciona $signal.csv',
    );
    if (r != null && r.files.single.path != null) {
      setState(() => _files[signal] = File(r.files.single.path!));
    }
  }

  void _setProgress(double p, String status) {
    if (mounted) setState(() { _analyzeProgress = p; _analyzeStatus = status; });
  }

  void _cancelAnalysis() {
    _elapsedTimer?.cancel();
    _elapsedTimer = null;
    if (mounted) setState(() {
      _cancelRequested = true;
      _analyzing       = false;
      _analyzeProgress = 0.0;
      _analyzeStatus   = '';
      _elapsedSecs     = 0;
    });
  }

  Future<void> _analyzeSubject() async {
    if (!_allFilesSelected) return;
    _cancelRequested = false;
    _elapsedSecs     = 0;

    setState(() {
      _analyzing       = true;
      _analyzeProgress = 0.0;
      _analyzeStatus   = 'Iniciando...';
      _annResult       = null;
    });

    // Render the progress bar before heavy work starts
    await Future.delayed(const Duration(milliseconds: 100));
    if (_cancelRequested) return;

    // Elapsed counter visible throughout the analysis
    _elapsedTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted || _cancelRequested) return;
      setState(() => _elapsedSecs++);
    });

    try {
      await _analyzeSubjectOffline(
        _files['EDA']!, _files['BVP']!, _files['ACC']!, _files['TEMP']!,
        subjectLabel: 'sujeto',
        notify: true,
        onProgress: _setProgress,
        isCancelled: () => _cancelRequested,
        onDone: (ann) {
          if (mounted && !_cancelRequested)
            setState(() { _annResult = ann; });
        },
      );
    } catch (e) {
      if (mounted && !_cancelRequested) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Error al analizar: $e'),
          backgroundColor: Colors.red,
        ));
      }
    } finally {
      _elapsedTimer?.cancel();
      _elapsedTimer = null;
      if (mounted) setState(() {
        _analyzing       = false;
        _analyzeProgress = 0.0;
        _analyzeStatus   = '';
        _elapsedSecs     = 0;
      });
    }
  }

  Future<void> _analyzeSubjectOffline(
    File eda, File bvp, File acc, File temp, {
    required String subjectLabel,
    required void Function(PredictionResult) onDone,
    bool notify = false,
    void Function(double, String)? onProgress,
    bool Function()? isCancelled,
  }) async {
    if (isCancelled?.call() == true) return;

    onProgress?.call(0.05, 'Inicializando modelo...');
    final annSvc = AnnInference();
    await annSvc.initialize();

    if (isCancelled?.call() == true) return;
    onProgress?.call(0.10, 'Leyendo archivos CSV...');

    final rawWindows = await compute(
      _extractWindowsTask,
      [eda.path, bvp.path, acc.path, temp.path],
    );

    if (isCancelled?.call() == true) return;

    if (rawWindows.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
          content: Text('Señales muy cortas — se necesitan al menos 60 s por señal.'),
          backgroundColor: Colors.orange,
        ));
      }
      return;
    }

    onProgress?.call(0.60, 'Normalizando features (${rawWindows.length} ventanas)...');
    final normalized = FeatureExtractor.normalizeZScore(rawWindows);

    if (isCancelled?.call() == true) return;

    final annWindows = <WindowPrediction>[];
    final total = normalized.length;

    for (int i = 0; i < total; i++) {
      if (isCancelled?.call() == true) return;

      final vec  = FeatureExtractor.toVector(normalized[i]);
      final tSec = normalized[i]['t_start_s'] ?? (i * 30.0);

      final annP = annSvc.predict(vec);
      annWindows.add(WindowPrediction(
        windowId: i, tStartS: tSec, tStartMin: tSec / 60,
        predClass: annP >= 0.5 ? 1 : 0,
        predLabel: annP >= 0.5 ? 'stress' : 'no-stress',
        predProba: annP,
      ));

      if (i % 5 == 0 || i == total - 1) {
        onProgress?.call(
          0.65 + 0.34 * (i + 1) / total,
          'Clasificando ventana ${i + 1}/$total...',
        );
        await Future.delayed(Duration.zero);
      }
    }

    if (isCancelled?.call() == true) return;
    onProgress?.call(0.99, 'Finalizando...');

    PredictionResult buildResult(String model, List<WindowPrediction> wins) {
      final nS = wins.where((w) => w.isStress).length;
      return PredictionResult(
        subject: '$subjectLabel-$model',
        nWindows: wins.length, nStress: nS, nRest: wins.length - nS,
        stressRatio: nS / wins.length,
        durationMin: wins.isEmpty ? 0 : (wins.last.tStartS / 60 + 1),
        windows: wins,
      );
    }

    final annResult = buildResult('ANN', annWindows);
    onDone(annResult);

    if (notify) {
      try {
        await NotificationService.showAnalysisDone(
          nWindows: annResult.nWindows,
          stressPct: annResult.stressRatio * 100,
          subject: subjectLabel,
        );
      } catch (_) { /* notificaciones opcionales — no interrumpen el resultado */ }
    }
  }

  // ── Dataset ZIP ────────────────────────────────────────────────────────────

  Future<void> _pickZip() async {
    final r = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['zip'],
      dialogTitle: 'Selecciona el banco de datos (.zip)',
    );
    if (r != null && r.files.single.path != null) {
      setState(() {
        _zipFile = File(r.files.single.path!);
        _datasetResults = [];
        _datasetStatus = '';
      });
    }
  }

  Future<void> _analyzeDataset() async {
    if (_zipFile == null) return;
    setState(() {
      _processingDataset = true;
      _datasetResults    = [];
      _datasetStatus     = 'Extrayendo ZIP...';
      _analyzeProgress   = 0.0;
    });

    try {
      final tmpDir   = await getTemporaryDirectory();
      final outDir   = Directory('${tmpDir.path}/dariel_dataset_${DateTime.now().millisecondsSinceEpoch}');
      await outDir.create(recursive: true);

      final bytes  = await _zipFile!.readAsBytes();
      final archive = ZipDecoder().decodeBytes(bytes);
      for (final file in archive) {
        final outPath = '${outDir.path}/${file.name}';
        if (file.isFile) {
          final outFile = File(outPath);
          await outFile.create(recursive: true);
          await outFile.writeAsBytes(file.content as List<int>);
        }
      }

      final subjects = <Directory>[];
      await for (final entity in outDir.list(recursive: true)) {
        if (entity is Directory) {
          final has = await Future.wait([
            File('${entity.path}/EDA.csv').exists(),
            File('${entity.path}/BVP.csv').exists(),
            File('${entity.path}/ACC.csv').exists(),
            File('${entity.path}/TEMP.csv').exists(),
          ]);
          if (has.every((b) => b)) subjects.add(entity);
        }
      }

      if (subjects.isEmpty) {
        setState(() { _datasetStatus = 'No se encontraron sujetos válidos en el ZIP.\nEsperado: carpetas con EDA.csv, BVP.csv, ACC.csv, TEMP.csv.'; _processingDataset = false; });
        return;
      }

      subjects.sort((a, b) => a.path.compareTo(b.path));
      final total = subjects.length;
      setState(() {
        _datasetStatus   = 'Procesando $total sujetos...';
        _analyzeProgress = 0.05;
      });

      final results = <_SubjectResult>[];
      for (int i = 0; i < total; i++) {
        final subj = subjects[i];
        final name = subj.path.split('/').last;
        setState(() {
          _datasetStatus   = '$name  (${i + 1}/$total)';
          _analyzeProgress = 0.05 + 0.93 * i / total;
        });

        try {
          PredictionResult? annR;
          await _analyzeSubjectOffline(
            File('${subj.path}/EDA.csv'), File('${subj.path}/BVP.csv'),
            File('${subj.path}/ACC.csv'), File('${subj.path}/TEMP.csv'),
            subjectLabel: name,
            onDone: (a) { annR = a; },
          );
          if (annR != null) {
            results.add(_SubjectResult(subjectId: name, ann: annR!));
          }
        } catch (_) { /* sujeto con error — continúa */ }
      }

      await outDir.delete(recursive: true);

      setState(() {
        _datasetResults   = results;
        _datasetStatus    = '${results.length}/${subjects.length} sujetos procesados.';
        _processingDataset = false;
      });

      if (results.isNotEmpty) {
        final avgStress = results
            .map((r) => r.ann.stressRatio)
            .reduce((a, b) => a + b) / results.length * 100;
        await NotificationService.showAnalysisDone(
          nWindows: results.fold(0, (s, r) => s + r.ann.nWindows),
          stressPct: avgStress,
          subject: '${results.length} sujetos',
        );
      }
    } catch (e) {
      if (mounted) setState(() => _datasetStatus = 'Error: $e');
    } finally {
      if (mounted) setState(() { _processingDataset = false; _analyzeProgress = 0.0; });
    }
  }

  // ── Build ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Análisis de Señales E4'),
        backgroundColor: cs.primary,
        foregroundColor: cs.onPrimary,
        bottom: TabBar(
          controller: _tabs,
          labelColor: Colors.white,
          unselectedLabelColor: Colors.white60,
          indicatorColor: const Color(0xFFFFD580),
          tabs: const [
            Tab(icon: Icon(Icons.person_outline_rounded), text: 'Sujeto'),
            Tab(icon: Icon(Icons.folder_zip_outlined), text: 'Banco de datos'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabs,
        children: [
          _buildSubjectTab(cs),
          _buildDatasetTab(cs),
        ],
      ),
    );
  }

  // ── Tab: Sujeto individual ─────────────────────────────────────────────────

  Widget _buildSubjectTab(ColorScheme cs) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Selección de archivos
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Señales E4 — sujeto individual',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  Text('Selecciona los 4 archivos CSV exportados por el Empatica E4.',
                      style: Theme.of(context).textTheme.bodySmall),
                  const SizedBox(height: 12),
                  ..._files.keys.map((sig) => _FileRow(
                        signal: sig, file: _files[sig],
                        enabled: !_analyzing,
                        onPick: () => _pickFile(sig),
                        onClear: () => setState(() => _files[sig] = null),
                      )),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Modelo activo
          Card(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              child: Row(
                children: [
                  Icon(Icons.memory_rounded, size: 16, color: cs.primary),
                  const SizedBox(width: 8),
                  Text('Modelo: ANN (TFLite int8 + Dart forward pass)',
                      style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant)),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          // Botones Analizar / Cancelar
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: _allFilesSelected && !_analyzing ? _analyzeSubject : null,
                  icon: _analyzing
                      ? const SizedBox(width: 18, height: 18,
                          child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                      : const Icon(Icons.analytics_rounded),
                  label: Text(_analyzing ? 'Analizando...' : 'Analizar'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: cs.primary, foregroundColor: cs.onPrimary,
                  ),
                ),
              ),
              if (_analyzing) ...[
                const SizedBox(width: 8),
                OutlinedButton.icon(
                  onPressed: _cancelAnalysis,
                  icon: const Icon(Icons.stop_circle_outlined, color: Colors.red),
                  label: const Text('Cancelar', style: TextStyle(color: Colors.red)),
                  style: OutlinedButton.styleFrom(
                    side: const BorderSide(color: Colors.red),
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
                  ),
                ),
              ],
            ],
          ),

          // Barra de progreso (visible durante el análisis)
          if (_analyzing) ...[
            const SizedBox(height: 12),
            Card(
              color: cs.primaryContainer.withOpacity(0.35),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text('Procesando señales...',
                            style: TextStyle(fontWeight: FontWeight.bold,
                                color: cs.primary, fontSize: 13)),
                        const Spacer(),
                        Text('${_elapsedSecs}s',
                            style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
                      ],
                    ),
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: _analyzeProgress > 0 ? _analyzeProgress : null,
                        minHeight: 8,
                        backgroundColor: cs.primaryContainer,
                        color: cs.primary,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        Expanded(
                          child: Text(_analyzeStatus,
                              style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant)),
                        ),
                        if (_analyzeProgress > 0)
                          Text('${(_analyzeProgress * 100).toInt()}%',
                              style: TextStyle(fontSize: 12,
                                  fontWeight: FontWeight.bold, color: cs.primary)),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],

          // Resultados
          if (_annResult != null) ...[
            const SizedBox(height: 20),
            _SingleModelBanner(result: _annResult!, modelName: 'ANN',
                color: Colors.indigo.shade600),
            const SizedBox(height: 12),
            _TimelineChart(windows: _annResult!.windows),
            const SizedBox(height: 12),
            _WindowsTable(windows: _annResult!.windows, title: 'Ventanas — ANN'),
          ],
        ],
      ),
    );
  }

  // ── Tab: Banco de datos ZIP ────────────────────────────────────────────────

  Widget _buildDatasetTab(ColorScheme cs) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Banco de datos completo',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 4),
                  Text('ZIP con carpetas por sujeto. Cada carpeta debe contener\nEDA.csv, BVP.csv, ACC.csv y TEMP.csv.',
                      style: Theme.of(context).textTheme.bodySmall),
                  const SizedBox(height: 14),
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          _zipFile != null
                              ? _zipFile!.path.split('/').last
                              : 'Sin archivo seleccionado',
                          style: TextStyle(
                            fontSize: 13,
                            color: _zipFile != null ? cs.onSurface : cs.onSurface.withOpacity(0.4),
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      const SizedBox(width: 8),
                      OutlinedButton.icon(
                        onPressed: _processingDataset ? null : _pickZip,
                        icon: const Icon(Icons.folder_zip_outlined, size: 18),
                        label: const Text('Seleccionar ZIP'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),

          ElevatedButton.icon(
            onPressed: _zipFile != null && !_processingDataset ? _analyzeDataset : null,
            icon: _processingDataset
                ? const SizedBox(width: 18, height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : const Icon(Icons.dataset_rounded),
            label: Text(_processingDataset ? 'Procesando...' : 'Procesar banco de datos'),
            style: ElevatedButton.styleFrom(
              backgroundColor: cs.secondary, foregroundColor: cs.onSecondary,
            ),
          ),

          if (_processingDataset) ...[
            const SizedBox(height: 12),
            Card(
              color: cs.secondaryContainer.withOpacity(0.35),
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Procesando dataset...',
                        style: TextStyle(fontWeight: FontWeight.bold,
                            color: cs.secondary, fontSize: 13)),
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: _analyzeProgress > 0 ? _analyzeProgress : null,
                        minHeight: 8,
                        backgroundColor: cs.secondaryContainer,
                        color: cs.secondary,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        Expanded(
                          child: Text(_datasetStatus,
                              style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant)),
                        ),
                        if (_analyzeProgress > 0)
                          Text('${(_analyzeProgress * 100).toInt()}%',
                              style: TextStyle(fontSize: 12,
                                  fontWeight: FontWeight.bold, color: cs.secondary)),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],

          if (_datasetStatus.isNotEmpty && !_processingDataset) ...[
            const SizedBox(height: 8),
            Text(_datasetStatus,
                style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant),
                textAlign: TextAlign.center),
          ],

          if (_datasetResults.isNotEmpty) ...[
            const SizedBox(height: 16),
            _DatasetResultsTable(results: _datasetResults),
          ],
        ],
      ),
    );
  }
}

// ── Widgets ───────────────────────────────────────────────────────────────────

class _FileRow extends StatelessWidget {
  final String signal;
  final File? file;
  final VoidCallback onPick;
  final VoidCallback onClear;
  final bool enabled;
  const _FileRow({required this.signal, required this.file,
      required this.onPick, required this.onClear, this.enabled = true});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        children: [
          SizedBox(
            width: 44,
            child: Text(signal,
                style: TextStyle(fontWeight: FontWeight.bold,
                    color: enabled ? cs.primary : cs.primary.withOpacity(0.4))),
          ),
          Expanded(
            child: Text(
              file != null ? file!.path.split('/').last : 'Sin seleccionar',
              style: TextStyle(
                color: file != null
                    ? (enabled ? cs.onSurface : cs.onSurface.withOpacity(0.5))
                    : cs.onSurface.withOpacity(0.4),
                fontSize: 13,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          if (file != null)
            IconButton(
              icon: Icon(Icons.close, size: 16,
                  color: enabled ? null : cs.onSurface.withOpacity(0.3)),
              onPressed: enabled ? onClear : null,
              padding: EdgeInsets.zero, constraints: const BoxConstraints(),
            ),
          const SizedBox(width: 4),
          TextButton(
            onPressed: enabled ? onPick : null,
            child: const Text('Seleccionar'),
          ),
        ],
      ),
    );
  }
}


// ── Banner de resultado único ─────────────────────────────────────────────────

class _SingleModelBanner extends StatelessWidget {
  final PredictionResult result;
  final String modelName;
  final Color color;
  const _SingleModelBanner(
      {required this.result, required this.modelName, required this.color});

  @override
  Widget build(BuildContext context) {
    final isStress   = result.stressRatio > 0.5;
    final stressColor = isStress ? Colors.red.shade600 : Colors.green.shade600;

    return Card(
      color: color.withOpacity(0.07),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(width: 10, height: 10,
                    decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
                const SizedBox(width: 8),
                Text('Resultados — Modelo $modelName',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold, color: color)),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
              decoration: BoxDecoration(
                color: stressColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: stressColor.withOpacity(0.3)),
              ),
              child: Row(
                children: [
                  Icon(isStress ? Icons.warning_amber_rounded : Icons.check_circle_outline,
                      color: stressColor, size: 20),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(isStress ? 'Estrés detectado' : 'Sin estrés significativo',
                            style: TextStyle(fontWeight: FontWeight.bold,
                                color: stressColor, fontSize: 13)),
                        Text(
                          '${(result.stressRatio * 100).toStringAsFixed(1)}% ventanas con estrés'
                          '  •  ${result.nWindows} ventanas',
                          style: TextStyle(fontSize: 11, color: stressColor),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            LinearProgressIndicator(
              value: result.stressRatio,
              backgroundColor: Colors.green.shade100,
              color: stressColor, minHeight: 6,
              borderRadius: BorderRadius.circular(3),
            ),
          ],
        ),
      ),
    );
  }
}


// ── Gráfica ANN ───────────────────────────────────────────────────────────────

class _TimelineChart extends StatelessWidget {
  final List<WindowPrediction> windows;
  const _TimelineChart({required this.windows});

  @override
  Widget build(BuildContext context) {
    final spots = windows
        .map((w) => FlSpot(w.tStartMin, w.predProba ?? w.predClass.toDouble()))
        .toList();

    final maxTime = windows.isEmpty
        ? 10.0
        : windows.fold(0.0, (m, w) => w.tStartMin > m ? w.tStartMin : m);
    final xInterval = maxTime <= 30
        ? 5.0
        : maxTime <= 60
            ? 10.0
            : maxTime <= 90
                ? 15.0
                : 20.0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text('Probabilidad de estrés en el tiempo',
                      style: Theme.of(context).textTheme.titleSmall),
                ),
                const SizedBox(width: 12),
                _LegendDot(color: Colors.indigo.shade400, label: 'ANN'),
              ],
            ),
            const SizedBox(height: 12),
            SizedBox(
              height: 150,
              child: LineChart(LineChartData(
                minY: 0, maxY: 1,
                gridData: FlGridData(
                  drawHorizontalLine: true, drawVerticalLine: false,
                  horizontalInterval: 0.5,
                  getDrawingHorizontalLine: (_) =>
                      FlLine(color: Colors.grey.shade200, strokeWidth: 1, dashArray: [4, 4]),
                ),
                borderData: FlBorderData(show: false),
                titlesData: FlTitlesData(
                  leftTitles: AxisTitles(sideTitles: SideTitles(
                    showTitles: true, reservedSize: 30, interval: 0.5,
                    getTitlesWidget: (v, _) =>
                        Text('${(v * 100).toInt()}%', style: const TextStyle(fontSize: 10)),
                  )),
                  bottomTitles: AxisTitles(sideTitles: SideTitles(
                    showTitles: true, reservedSize: 26, interval: xInterval,
                    getTitlesWidget: (v, _) {
                      if (v.round() % xInterval.round() != 0) return const SizedBox.shrink();
                      return Padding(
                        padding: const EdgeInsets.only(top: 4),
                        child: Text('${v.toInt()}m', style: const TextStyle(fontSize: 10)),
                      );
                    },
                  )),
                  topTitles:   const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                  rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                ),
                lineBarsData: [
                  if (spots.isNotEmpty)
                    LineChartBarData(
                      spots: spots, isCurved: true, barWidth: 2,
                      color: Colors.indigo.shade400,
                      dotData: const FlDotData(show: false),
                      belowBarData: BarAreaData(show: true,
                          color: Colors.indigo.shade100.withOpacity(0.25)),
                    ),
                  if (windows.isNotEmpty)
                    LineChartBarData(
                      spots: [
                        FlSpot(windows.first.tStartMin, 0.5),
                        FlSpot(windows.last.tStartMin, 0.5),
                      ],
                      color: Colors.grey.shade400, barWidth: 1,
                      dashArray: [4, 4], dotData: const FlDotData(show: false),
                    ),
                ],
              )),
            ),
          ],
        ),
      ),
    );
  }
}

class _LegendDot extends StatelessWidget {
  final Color color;
  final String label;
  const _LegendDot({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(width: 10, height: 3,
            decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(2))),
        const SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 11, color: color, fontWeight: FontWeight.bold)),
      ],
    );
  }
}

// ── Tabla de ventanas ─────────────────────────────────────────────────────────

class _WindowsTable extends StatefulWidget {
  final List<WindowPrediction> windows;
  final String title;
  const _WindowsTable({required this.windows, required this.title});

  @override
  State<_WindowsTable> createState() => _WindowsTableState();
}

class _WindowsTableState extends State<_WindowsTable> {
  bool _showAll = false;

  @override
  Widget build(BuildContext context) {
    final shown = _showAll ? widget.windows : widget.windows.take(10).toList();
    return Card(
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
            child: Row(
              children: [
                Text(widget.title, style: Theme.of(context).textTheme.titleSmall),
                const Spacer(),
                Text('${widget.windows.length} ventanas',
                    style: Theme.of(context).textTheme.bodySmall),
              ],
            ),
          ),
          const Divider(height: 1),
          ...shown.map((w) => _WindowTile(window: w)),
          if (widget.windows.length > 10)
            TextButton(
              onPressed: () => setState(() => _showAll = !_showAll),
              child: Text(_showAll ? 'Mostrar menos' : 'Ver todas'),
            ),
        ],
      ),
    );
  }
}

class _WindowTile extends StatelessWidget {
  final WindowPrediction window;
  const _WindowTile({required this.window});

  @override
  Widget build(BuildContext context) {
    final color = window.isStress ? Colors.red.shade600 : Colors.green.shade600;
    final pct   = window.predProba != null
        ? '${(window.predProba! * 100).toStringAsFixed(1)}%'
        : '';
    return ListTile(
      dense: true,
      leading: CircleAvatar(
        radius: 14,
        backgroundColor: color.withOpacity(0.15),
        child: Icon(
          window.isStress ? Icons.warning_amber_rounded : Icons.check_circle_outline,
          size: 16, color: color,
        ),
      ),
      title: Text(window.isStress ? 'Estrés' : 'Reposo',
          style: TextStyle(fontWeight: FontWeight.bold, color: color, fontSize: 13)),
      subtitle: Text('t = ${window.tStartMin} min'),
      trailing: pct.isNotEmpty
          ? Text(pct, style: TextStyle(fontWeight: FontWeight.bold, color: color))
          : null,
    );
  }
}

// ── Tabla de resultados dataset ───────────────────────────────────────────────

class _DatasetResultsTable extends StatelessWidget {
  final List<_SubjectResult> results;
  const _DatasetResultsTable({required this.results});

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
            child: Text('Resumen por sujeto',
                style: Theme.of(context).textTheme.titleSmall),
          ),
          const Divider(height: 1),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Row(
              children: [
                Expanded(flex: 3, child: Text('Sujeto',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 12, color: cs.primary))),
                Expanded(flex: 2, child: Text('ANN %estrés',
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 11),
                    textAlign: TextAlign.center)),
                Expanded(flex: 2, child: Text('Ventanas',
                    style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 11),
                    textAlign: TextAlign.center)),
              ],
            ),
          ),
          const Divider(height: 1),
          ...results.map((r) {
            final annPct = (r.ann.stressRatio * 100).toStringAsFixed(0);
            final isStress = r.ann.stressRatio > 0.5;
            final color = isStress ? Colors.red.shade600 : Colors.green.shade600;
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              child: Row(
                children: [
                  Expanded(flex: 3, child: Row(
                    children: [
                      Icon(isStress ? Icons.warning_amber_rounded : Icons.check_circle_outline,
                          size: 14, color: color),
                      const SizedBox(width: 4),
                      Flexible(child: Text(r.subjectId,
                          style: const TextStyle(fontSize: 12),
                          overflow: TextOverflow.ellipsis)),
                    ],
                  )),
                  Expanded(flex: 2, child: Text('$annPct%',
                      style: TextStyle(fontSize: 12, color: Colors.indigo.shade600,
                          fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center)),
                  Expanded(flex: 2, child: Text('${r.ann.nWindows}',
                      style: const TextStyle(fontSize: 12),
                      textAlign: TextAlign.center)),
                ],
              ),
            );
          }),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
