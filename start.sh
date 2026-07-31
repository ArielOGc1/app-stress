#!/bin/bash
# Servidor backend — Stress Classifier
# Ejecutar desde cualquier lugar, se ubica solo.

cd "$(dirname "$0")"

IP=$(hostname -I | awk '{print $1}')

# Liberar puerto si está ocupado
PID=$(ss -tlnp "sport = :8000" 2>/dev/null | awk 'NR>1 {match($0, /pid=([0-9]+)/, a); if (a[1]) print a[1]}' | head -1)
if [[ -n "$PID" ]]; then
  echo "Puerto 8000 ocupado (PID $PID) — liberando..."
  kill "$PID" 2>/dev/null && sleep 1
fi

clear
echo "╔══════════════════════════════════════════╗"
echo "║    Stress Classifier — Backend v1.0      ║"
echo "╠══════════════════════════════════════════╣"
echo "║                                          ║"
printf  "║  IP de esta PC:  %-24s║\n" "$IP"
printf  "║  Configura app:  %-24s║\n" "http://$IP:8000"
echo "║                                          ║"
echo "║  Escanea el QR de abajo con la app       ║"
echo "║  (Ajustes → ícono QR) para conectar.     ║"
echo "║                                          ║"
echo "║  El streaming arranca automáticamente    ║"
echo "║  cuando el celular presiona 'Iniciar'.   ║"
echo "║                                          ║"
echo "║  Ctrl+C para detener.                    ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Generar QR con la URL del servidor
/home/ariel/miniconda3/envs/tesis/bin/python3 -c "
import qrcode, sys
q = qrcode.QRCode(
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=1, border=1
)
q.add_data('http://${IP}:8000')
q.make(fit=True)
q.print_ascii(invert=True)
print()
" 2>/dev/null || echo "  (instala qrcode: pip install qrcode)"
echo ""

conda run -n tesis uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
