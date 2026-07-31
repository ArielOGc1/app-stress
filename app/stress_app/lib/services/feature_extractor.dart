import 'dart:io';
import 'dart:math';

/// Extrae las 12 features del pipeline WESAD desde archivos CSV del Empatica E4.
/// Señales: EDA (4Hz), BVP (64Hz), ACC (32Hz, 3 cols), TEMP (4Hz).
/// Ventanas de 60s con 50% de solapamiento (step=30s).
class FeatureExtractor {
  static const int fsEda  = 4;
  static const int fsBvp  = 64;
  static const int fsAcc  = 32;
  static const int fsTemp = 4;

  static const int _windowSec = 60;
  static const int _stepSec   = 30;

  static const int _winEda  = _windowSec * fsEda;   // 240
  static const int _winBvp  = _windowSec * fsBvp;   // 3840
  static const int _winAcc  = _windowSec * fsAcc;   // 1920
  static const int _winTemp = _windowSec * fsTemp;  // 240

  static const int _stepEda  = _stepSec * fsEda;   // 120
  static const int _stepBvp  = _stepSec * fsBvp;   // 1920
  static const int _stepAcc  = _stepSec * fsAcc;   // 960
  static const int _stepTemp = _stepSec * fsTemp;  // 120

  static const List<String> featureNames = [
    'scl_mean', 'scl_std', 'scr_count', 'scr_amp_mean', 'phasic_energy',
    'acc_mag_std', 'acc_movement',
    'hrv_mean_rr', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
    'temp_mean',
  ];

  // ── Parsing ───────────────────────────────────────────────────────────────

  static Future<List<double>> _parseSingleCol(File file) async {
    final lines = await file.readAsLines();
    final vals = <double>[];
    int headerCount = 0;
    for (final line in lines) {
      final v = double.tryParse(line.trim());
      if (v == null) continue;
      // Saltar las 2 primeras líneas del formato E4 (unix timestamp + sample rate)
      if (headerCount < 2) { headerCount++; continue; }
      vals.add(v);
    }
    return vals;
  }

  static Future<List<List<double>>> _parseAcc(File file) async {
    final lines = await file.readAsLines();
    final vals = <List<double>>[];
    int skipped = 0;
    for (final line in lines) {
      if (line.trim().isEmpty) continue;
      final parts = line.split(',');
      // Saltar las 2 primeras líneas del formato E4 (1 columna cada una)
      if (parts.length < 3) { skipped++; continue; }
      // Solo las primeras 2 líneas de 1 col son headers; el resto son datos válidos
      // La verificación de parts.length >= 3 ya las excluye correctamente
      vals.add(parts.take(3)
          .map((p) => double.tryParse(p.trim()) ?? 0.0)
          .toList());
    }
    return vals;
  }

  // ── EDA features ──────────────────────────────────────────────────────────

  static Map<String, double> _edaFeatures(List<double> eda) {
    // SCL ≈ moving average (ventana de 4*fs = 16 muestras)
    final scl = _movingAvg(eda, fsEda * 4);
    final phasic = List.generate(eda.length, (i) => eda[i] - scl[i]);

    final sclMean = _mean(scl);
    final sclStd  = _std(scl);

    final peaks = _findPeaks(phasic, minDist: fsEda, threshold: 0.01);
    final scrCount   = peaks.length.toDouble();
    final scrAmpMean = peaks.isEmpty
        ? 0.0
        : peaks.map((i) => phasic[i]).reduce((a, b) => a + b) / peaks.length;
    final phasicEnergy = phasic.fold(0.0, (s, v) => s + v * v);

    return {
      'scl_mean':      sclMean,
      'scl_std':       sclStd,
      'scr_count':     scrCount,
      'scr_amp_mean':  scrAmpMean,
      'phasic_energy': phasicEnergy,
    };
  }

  // ── ACC features ──────────────────────────────────────────────────────────

  static Map<String, double> _accFeatures(List<List<double>> acc) {
    final mag = acc
        .map((r) => sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]))
        .toList();
    final magStd = _std(mag);
    double movement = 0.0;
    for (int i = 1; i < mag.length; i++) {
      movement += (mag[i] - mag[i - 1]).abs();
    }
    movement /= max(1, mag.length - 1);
    return {'acc_mag_std': magStd, 'acc_movement': movement};
  }

  // ── HRV features ──────────────────────────────────────────────────────────

  static Map<String, double> _hrvFeatures(List<double> bvp) {
    const zero = {
      'hrv_mean_rr': 0.0,
      'hrv_sdnn': 0.0,
      'hrv_rmssd': 0.0,
      'hrv_pnn50': 0.0,
    };

    final sd = _std(bvp);
    if (sd == 0) return zero;
    final mn   = _mean(bvp);
    final norm = bvp.map((v) => (v - mn) / sd).toList();

    // Picos BVP con distancia mínima ~0.4s
    final peaks = _findPeaks(norm, minDist: (fsBvp * 0.4).toInt(), threshold: 0.5);
    if (peaks.length < 3) return zero;

    // Intervalos RR en ms
    final rr = <double>[];
    for (int i = 1; i < peaks.length; i++) {
      rr.add((peaks[i] - peaks[i - 1]) / fsBvp * 1000.0);
    }
    // Filtro fisiológico: 300–2000 ms
    final rrF = rr.where((r) => r >= 300 && r <= 2000).toList();
    if (rrF.length < 2) return zero;

    final meanRr = _mean(rrF);
    final sdnn   = _std(rrF);

    double sumSq = 0;
    int pnn50n = 0;
    for (int i = 1; i < rrF.length; i++) {
      final d = rrF[i] - rrF[i - 1];
      sumSq += d * d;
      if (d.abs() > 50) pnn50n++;
    }
    final rmssd = sqrt(sumSq / (rrF.length - 1));
    final pnn50 = pnn50n / (rrF.length - 1) * 100.0;

    return {
      'hrv_mean_rr': meanRr,
      'hrv_sdnn':    sdnn,
      'hrv_rmssd':   rmssd,
      'hrv_pnn50':   pnn50,
    };
  }

  // ── Pipeline principal ────────────────────────────────────────────────────

  /// Extrae features de todas las ventanas posibles.
  /// Devuelve lista de mapas con featureNames + 't_start_s'.
  static Future<List<Map<String, double>>> extractWindows({
    required File edaFile,
    required File bvpFile,
    required File accFile,
    required File tempFile,
  }) async {
    final eda  = await _parseSingleCol(edaFile);
    final bvp  = await _parseSingleCol(bvpFile);
    final acc  = await _parseAcc(accFile);
    final temp = await _parseSingleCol(tempFile);

    if (eda.length < _winEda || bvp.length < _winBvp ||
        acc.length < _winAcc || temp.length < _winTemp) {
      return [];
    }

    final nW = [
      (eda.length  - _winEda)  ~/ _stepEda  + 1,
      (bvp.length  - _winBvp)  ~/ _stepBvp  + 1,
      (acc.length  - _winAcc)  ~/ _stepAcc  + 1,
      (temp.length - _winTemp) ~/ _stepTemp + 1,
    ].reduce(min);

    final windows = <Map<String, double>>[];
    for (int i = 0; i < nW; i++) {
      final feats = <String, double>{};
      feats.addAll(_edaFeatures(
          eda.sublist(i * _stepEda, i * _stepEda + _winEda)));
      feats.addAll(_accFeatures(
          acc.sublist(i * _stepAcc, i * _stepAcc + _winAcc)));
      feats.addAll(_hrvFeatures(
          bvp.sublist(i * _stepBvp, i * _stepBvp + _winBvp)));
      feats['temp_mean'] = _mean(
          temp.sublist(i * _stepTemp, i * _stepTemp + _winTemp));
      feats['t_start_s'] = (i * _stepSec).toDouble();
      windows.add(feats);
    }
    return windows;
  }

  /// Normalización z-score intra-sujeto (igual que el pipeline Python).
  static List<Map<String, double>> normalizeZScore(
      List<Map<String, double>> windows) {
    if (windows.isEmpty) return [];
    final out = windows.map((w) => Map<String, double>.from(w)).toList();
    for (final feat in featureNames) {
      final vals = windows.map((w) => w[feat] ?? 0.0).toList();
      final m = _mean(vals);
      final s = _std(vals);
      for (final w in out) {
        w[feat] = s > 0 ? ((w[feat] ?? 0.0) - m) / s : 0.0;
      }
    }
    return out;
  }

  /// Convierte un mapa de features al vector ordenado que espera el modelo.
  static List<double> toVector(Map<String, double> feats) =>
      featureNames.map((f) => feats[f] ?? 0.0).toList();

  // ── Helpers matemáticos ───────────────────────────────────────────────────

  static List<double> _movingAvg(List<double> data, int win) {
    final out = List<double>.filled(data.length, 0.0);
    for (int i = 0; i < data.length; i++) {
      final s = max(0, i - win ~/ 2);
      final e = min(data.length, i + win ~/ 2 + 1);
      double sum = 0;
      for (int j = s; j < e; j++) sum += data[j];
      out[i] = sum / (e - s);
    }
    return out;
  }

  static List<int> _findPeaks(List<double> data,
      {required int minDist, double? threshold}) {
    final peaks = <int>[];
    for (int i = 1; i < data.length - 1; i++) {
      if (data[i] > data[i - 1] && data[i] > data[i + 1]) {
        if (threshold != null && data[i] < threshold) continue;
        if (peaks.isNotEmpty && i - peaks.last < minDist) {
          if (data[i] > data[peaks.last]) peaks[peaks.length - 1] = i;
        } else {
          peaks.add(i);
        }
      }
    }
    return peaks;
  }

  static double _mean(List<double> d) {
    if (d.isEmpty) return 0.0;
    return d.fold(0.0, (s, v) => s + v) / d.length;
  }

  static double _std(List<double> d) {
    if (d.length < 2) return 0.0;
    final m = _mean(d);
    final v = d.fold(0.0, (s, v) => s + (v - m) * (v - m)) / d.length;
    return sqrt(v);
  }
}
