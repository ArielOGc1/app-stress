import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/api_service.dart';
import 'csv_screen.dart';
import 'ble_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ApiService>().checkHealth();
    });
  }

  @override
  Widget build(BuildContext context) {
    final cs    = Theme.of(context).colorScheme;
    final api   = context.watch<ApiService>();

    return Scaffold(
      backgroundColor: cs.surface,
      appBar: AppBar(
        title: const _MacsLogo(),
        centerTitle: true,
        backgroundColor: cs.primary,
        foregroundColor: cs.onPrimary,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const SettingsScreen()),
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // ── Estado del servidor ──────────────────────────────────────
              _ServerStatusBanner(api: api),
              const SizedBox(height: 28),

              // ── Título ───────────────────────────────────────────────────
              Text(
                'Selecciona el modo de análisis',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: cs.onSurfaceVariant,
                    ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),

              // ── Tarjetas de modo ─────────────────────────────────────────
              Expanded(
                child: Column(
                  children: [
                    _ModeCard(
                      icon: Icons.upload_file_rounded,
                      title: 'Análisis de Señales E4',
                      subtitle: 'Carga archivos CSV de un sujeto\no un banco de datos completo (ZIP).',
                      color: cs.primaryContainer,
                      iconColor: cs.primary,
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const CsvScreen()),
                      ),
                    ),
                    const SizedBox(height: 16),
                    _ModeCard(
                      icon: Icons.bluetooth_rounded,
                      title: 'Monitoreo en Tiempo Real',
                      subtitle: 'Conecta vía Bluetooth o simula\nun stream de datos E4 en vivo.',
                      color: cs.secondaryContainer,
                      iconColor: cs.secondary,
                      onTap: () => Navigator.push(
                        context,
                        MaterialPageRoute(builder: (_) => const BleScreen()),
                      ),
                    ),
                  ],
                ),
              ),

              // ── Footer ────────────────────────────────────────────────────
              Padding(
                padding: const EdgeInsets.only(top: 16),
                child: Text(
                  'Ingeniería Biomédica',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: cs.onSurfaceVariant.withOpacity(0.6),
                      ),
                  textAlign: TextAlign.center,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ── Logo ─────────────────────────────────────────────────────────────────────

class _MacsLogo extends StatelessWidget {
  const _MacsLogo();

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Image.asset('assets/images/macs_logo.png', height: 36),
        const SizedBox(width: 8),
        const Text(
          'MACS',
          style: TextStyle(
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.w900,
            letterSpacing: 2,
          ),
        ),
      ],
    );
  }
}

// ── Widgets auxiliares ────────────────────────────────────────────────────────

class _ServerStatusBanner extends StatelessWidget {
  final ApiService api;
  const _ServerStatusBanner({required this.api});

  @override
  Widget build(BuildContext context) {
    final connected = api.isConnected;

    return Material(
      color: connected
          ? Colors.green.shade50
          : Colors.blue.shade50,
      borderRadius: BorderRadius.circular(12),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            Icon(
              connected ? Icons.cloud_done_rounded : Icons.offline_bolt_rounded,
              color: connected ? Colors.green.shade700 : Colors.blue.shade700,
              size: 20,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                connected
                    ? 'Servidor conectado — ${api.baseUrl}'
                    : 'Modo offline — Inferencia local (ANN)',
                style: TextStyle(
                  color: connected
                      ? Colors.green.shade800
                      : Colors.blue.shade800,
                  fontSize: 13,
                ),
              ),
            ),
            IconButton(
              icon: const Icon(Icons.refresh, size: 18),
              onPressed: () => api.checkHealth(),
              color: connected ? Colors.green.shade700 : Colors.blue.shade700,
              padding: EdgeInsets.zero,
              constraints: const BoxConstraints(),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModeCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color color;
  final Color iconColor;
  final VoidCallback? onTap;

  const _ModeCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.color,
    required this.iconColor,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final disabled = onTap == null;

    return Expanded(
      child: Opacity(
        opacity: disabled ? 0.5 : 1.0,
        child: Card(
          color: color,
          child: InkWell(
            borderRadius: BorderRadius.circular(16),
            onTap: onTap,
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: iconColor.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Icon(icon, size: 36, color: iconColor),
                  ),
                  const SizedBox(width: 20),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          title,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                fontWeight: FontWeight.bold,
                              ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          subtitle,
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                color: Theme.of(context)
                                    .colorScheme
                                    .onSurface
                                    .withOpacity(0.7),
                              ),
                        ),
                      ],
                    ),
                  ),
                  Icon(
                    Icons.arrow_forward_ios_rounded,
                    size: 16,
                    color: iconColor,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
