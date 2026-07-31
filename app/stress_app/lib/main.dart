import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'screens/home_screen.dart';
import 'screens/splash_screen.dart';
import 'services/notification_service.dart';
import 'services/api_service.dart';
import 'services/ble_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await NotificationService.initialize();
  final apiService = ApiService();
  await apiService.loadSavedUrl();
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => apiService),
        ChangeNotifierProvider(create: (_) => BleService()),
      ],
      child: const StressApp(),
    ),
  );
}

class StressApp extends StatelessWidget {
  const StressApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MACS',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF1565C0),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
        fontFamily: 'Roboto',
        cardTheme: CardThemeData(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
          ),
        ),
      ),
      home: const SplashScreen(),
    );
  }
}
