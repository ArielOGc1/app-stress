import 'package:flutter/material.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/api_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _urlCtrl;
  bool _editingUrl = false;

  @override
  void initState() {
    super.initState();
    _urlCtrl = TextEditingController(text: context.read<ApiService>().baseUrl);
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    super.dispose();
  }

  Future<void> _scanQr() async {
    final scanned = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => const _QrScannerSheet(),
    );
    if (scanned != null && scanned.isNotEmpty) {
      final url = scanned.startsWith('http') ? scanned : 'http://$scanned';
      _urlCtrl.text = url;
      await _save();
    }
  }

  Future<void> _save() async {
    final api = context.read<ApiService>();
    final url = _urlCtrl.text.trim();
    if (url.isEmpty) return;

    api.setBaseUrl(url);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('api_url', url);
    setState(() => _editingUrl = false);

    final ok = await api.checkHealth();
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(ok ? 'Conectado correctamente' : 'No se pudo conectar'),
          backgroundColor: ok ? Colors.green : Colors.red,
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs  = Theme.of(context).colorScheme;
    final api = context.watch<ApiService>();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Ajustes'),
        backgroundColor: cs.primary,
        foregroundColor: cs.onPrimary,
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 20),
        children: [

          // ── Conexión ────────────────────────────────────────────────────
          _SectionLabel(label: 'Conexión'),
          Card(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            child: Column(
              children: [
                ListTile(
                  leading: CircleAvatar(
                    radius: 16,
                    backgroundColor: api.isConnected
                        ? Colors.green.shade50
                        : Colors.grey.shade100,
                    child: Icon(
                      api.isConnected
                          ? Icons.cloud_done_rounded
                          : Icons.cloud_off_rounded,
                      size: 18,
                      color: api.isConnected
                          ? Colors.green.shade700
                          : Colors.grey.shade500,
                    ),
                  ),
                  title: Text(
                    api.isConnected ? 'En línea' : 'Sin conexión',
                    style: TextStyle(
                      fontWeight: FontWeight.w600,
                      color: api.isConnected
                          ? Colors.green.shade700
                          : Colors.grey.shade600,
                    ),
                  ),
                  subtitle: Text(
                    api.isConnected ? api.baseUrl : 'Modo offline activo',
                    style: const TextStyle(fontSize: 12),
                  ),
                  trailing: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        tooltip: 'Verificar conexión',
                        icon: const Icon(Icons.refresh_rounded, size: 20),
                        onPressed: () => api.checkHealth(),
                      ),
                      IconButton(
                        tooltip: 'Escanear QR del servidor',
                        icon: const Icon(Icons.qr_code_scanner_rounded, size: 20),
                        onPressed: _scanQr,
                      ),
                      IconButton(
                        tooltip: 'Editar URL manualmente',
                        icon: Icon(
                          _editingUrl ? Icons.close_rounded : Icons.edit_outlined,
                          size: 20,
                        ),
                        onPressed: () => setState(() {
                          _editingUrl = !_editingUrl;
                          if (!_editingUrl) _urlCtrl.text = api.baseUrl;
                        }),
                      ),
                    ],
                  ),
                ),
                if (!api.isConnected && !_editingUrl) ...[
                  const Divider(height: 1),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 10, 16, 12),
                    child: Row(
                      children: [
                        Icon(Icons.wifi_rounded,
                            size: 15, color: Colors.grey.shade500),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            'Asegúrate de que el celular y la PC estén en la misma red WiFi.',
                            style: TextStyle(
                                fontSize: 11, color: Colors.grey.shade600),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
                if (_editingUrl) ...[
                  const Divider(height: 1),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        TextField(
                          controller: _urlCtrl,
                          decoration: const InputDecoration(
                            labelText: 'Dirección del servidor',
                            hintText: 'http://192.168.1.X:8000',
                            prefixIcon: Icon(Icons.dns_outlined),
                            border: OutlineInputBorder(),
                            isDense: true,
                          ),
                          keyboardType: TextInputType.url,
                          autofocus: true,
                          onSubmitted: (_) => _save(),
                        ),
                        const SizedBox(height: 10),
                        FilledButton(
                          onPressed: _save,
                          child: const Text('Guardar'),
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          ),

          const SizedBox(height: 24),

          // ── Acerca de ───────────────────────────────────────────────────
          _SectionLabel(label: 'Acerca de'),
          Card(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Row(
                children: [
                  Container(
                    width: 52, height: 52,
                    decoration: BoxDecoration(
                      color: cs.primaryContainer,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Icon(Icons.monitor_heart_rounded,
                        color: cs.primary, size: 28),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Stress Classifier',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, fontSize: 15),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          'Versión 1.0.0',
                          style: TextStyle(
                              fontSize: 12, color: Colors.grey.shade600),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Tesis — Ingeniería Biomédica 2026',
                          style: TextStyle(
                              fontSize: 12, color: Colors.grey.shade500),
                        ),
                      ],
                    ),
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

// ── QR Scanner ───────────────────────────────────────────────────────────────

class _QrScannerSheet extends StatefulWidget {
  const _QrScannerSheet();

  @override
  State<_QrScannerSheet> createState() => _QrScannerSheetState();
}

class _QrScannerSheetState extends State<_QrScannerSheet> {
  bool _scanned = false;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return SizedBox(
      height: MediaQuery.of(context).size.height * 0.6,
      child: Column(
        children: [
          const SizedBox(height: 12),
          Container(
            width: 40, height: 4,
            decoration: BoxDecoration(
              color: Colors.grey.shade300,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Apunta al QR del servidor',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 4),
          Text(
            'Ejecuta ./start.sh en la PC y escanea el QR',
            style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
          ),
          const SizedBox(height: 12),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: MobileScanner(
                onDetect: (capture) {
                  if (_scanned) return;
                  final raw = capture.barcodes.firstOrNull?.rawValue;
                  if (raw != null && raw.isNotEmpty) {
                    _scanned = true;
                    Navigator.pop(context, raw);
                  }
                },
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.info_outline_rounded,
                    size: 14, color: cs.primary.withOpacity(0.7)),
                const SizedBox(width: 6),
                Text(
                  'Busca el código QR en la terminal de la PC',
                  style: TextStyle(fontSize: 11, color: cs.primary.withOpacity(0.7)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ── Section label ─────────────────────────────────────────────────────────────

class _SectionLabel extends StatelessWidget {
  final String label;
  const _SectionLabel({required this.label});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, bottom: 8),
      child: Text(
        label.toUpperCase(),
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 1.2,
          color: Theme.of(context).colorScheme.primary,
        ),
      ),
    );
  }
}
