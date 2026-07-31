import 'package:flutter_local_notifications/flutter_local_notifications.dart';

class NotificationService {
  static final _plugin = FlutterLocalNotificationsPlugin();
  static bool _initialized = false;

  static Future<void> initialize() async {
    if (_initialized) return;
    try {
      const android = AndroidInitializationSettings('@mipmap/ic_launcher');
      await _plugin.initialize(
        const InitializationSettings(android: android),
      );
      // Solicitar permiso en Android 13+
      await _plugin
          .resolvePlatformSpecificImplementation<
              AndroidFlutterLocalNotificationsPlugin>()
          ?.requestNotificationsPermission();
      _initialized = true;
    } catch (_) {
      // Si las notificaciones no están disponibles, no fallar la app
    }
  }

  static Future<void> showAnalysisDone({
    required int nWindows,
    required double stressPct,
    String subject = '',
  }) async {
    if (!_initialized) return;
    final pct   = stressPct.toStringAsFixed(1);
    final label = stressPct > 50 ? 'Estrés detectado' : 'Sin estrés significativo';
    final title = subject.isNotEmpty ? 'Análisis listo — $subject' : 'Análisis completado';
    const channel = AndroidNotificationDetails(
      'macs_analysis',
      'Resultados de análisis',
      channelDescription: 'Notificación cuando el análisis de señales finaliza',
      importance: Importance.high,
      priority: Priority.high,
      icon: '@mipmap/ic_launcher',
    );
    await _plugin.show(
      0,
      title,
      '$label · $pct% estrés en $nWindows ventanas',
      const NotificationDetails(android: channel),
    );
  }
}
