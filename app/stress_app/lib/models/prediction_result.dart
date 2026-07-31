class WindowPrediction {
  final int windowId;
  final double tStartS;
  final double tStartMin;
  final int predClass;
  final String predLabel;
  final double? predProba;

  const WindowPrediction({
    required this.windowId,
    required this.tStartS,
    required this.tStartMin,
    required this.predClass,
    required this.predLabel,
    this.predProba,
  });

  bool get isStress => predClass == 1;

  factory WindowPrediction.fromJson(Map<String, dynamic> json) {
    return WindowPrediction(
      windowId:  json['window_id'] as int,
      tStartS:   (json['t_start_s'] as num).toDouble(),
      tStartMin: (json['t_start_min'] as num).toDouble(),
      predClass: json['pred_class'] as int,
      predLabel: json['pred_label'] as String,
      predProba: json['pred_proba'] != null
          ? (json['pred_proba'] as num).toDouble()
          : null,
    );
  }
}

class PredictionResult {
  final String subject;
  final int nWindows;
  final int nStress;
  final int nRest;
  final double stressRatio;
  final double durationMin;
  final List<WindowPrediction> windows;

  const PredictionResult({
    required this.subject,
    required this.nWindows,
    required this.nStress,
    required this.nRest,
    required this.stressRatio,
    required this.durationMin,
    required this.windows,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      subject:     json['subject'] as String,
      nWindows:    json['n_windows'] as int,
      nStress:     json['n_stress'] as int,
      nRest:       json['n_rest'] as int,
      stressRatio: (json['stress_ratio'] as num).toDouble(),
      durationMin: (json['duration_min'] as num).toDouble(),
      windows: (json['windows'] as List)
          .map((w) => WindowPrediction.fromJson(w as Map<String, dynamic>))
          .toList(),
    );
  }
}

class StreamPrediction {
  final int predClass;
  final String label;
  final double? proba;
  final DateTime timestamp;

  StreamPrediction({
    required this.predClass,
    required this.label,
    this.proba,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  bool get isStress => predClass == 1;

  factory StreamPrediction.fromJson(Map<String, dynamic> json) {
    return StreamPrediction(
      predClass: json['class'] as int,
      label:     json['label'] as String,
      proba:     json['proba'] != null ? (json['proba'] as num).toDouble() : null,
    );
  }
}
