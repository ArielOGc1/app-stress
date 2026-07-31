"""
Simulador de streaming E4 para modo BLE.

Modos:
  --synthetic  Genera ventanas infinitas con ruido gaussiano (distribución WESAD).
               Recomendado para demo — no necesita archivos adicionales.

  --demo-json  Reproduce un archivo demo_windows.json en loop.

  --subject    Carga CSV de un sujeto Campanella (EDA/BVP/ACC/TEMP.csv).

Uso estándar (modo demo automático desde backend):
    python app/backend/streamer.py --synthetic --ws ws://localhost:8000/ws/stream
"""
import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

_DEMO_JSON = Path(__file__).parent.parent / "stress_app/assets/models/demo_windows.json"


FEATURES_ORDERED = [
    'scl_mean', 'scl_std', 'scr_count', 'scr_amp_mean', 'phasic_energy',
    'acc_mag_std', 'acc_movement',
    'hrv_mean_rr', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
    'temp_mean',
]

# Patrón de clases: bloques alternados baseline/stress (simula protocolo WESAD)
_PATTERN = [0]*10 + [1]*12 + [0]*8 + [1]*10 + [0]*5 + [1]*8 + [0]*7 + [1]*6


def _compute_stats(demo_path: Path) -> dict:
    """Calcula media y std por feature y clase desde demo_windows.json."""
    windows = json.loads(demo_path.read_text())
    vals: dict = {f: {0: [], 1: []} for f in FEATURES_ORDERED}
    for w in windows:
        cls = int(w.get('label', 0))
        for f in FEATURES_ORDERED:
            if f in w and w[f] is not None:
                vals[f][cls].append(float(w[f]))
    stats = {}
    for f in FEATURES_ORDERED:
        stats[f] = {}
        for cls in [0, 1]:
            v = vals[f][cls]
            if v:
                mean = sum(v) / len(v)
                std  = math.sqrt(sum((x - mean) ** 2 for x in v) / max(len(v) - 1, 1))
                stats[f][cls] = (mean, max(std * 0.3, 1e-4))
            else:
                stats[f][cls] = (0.0, 0.1)
    return stats


async def stream_synthetic_to_ws(ws_url: str, model: str = "SVM", speed: float = 3.0):
    """
    Genera ventanas sintéticas infinitas con ruido gaussiano (distribución WESAD)
    y las envía al backend. Se reconecta automáticamente si se pierde la conexión.
    """
    try:
        import websockets
    except ImportError:
        print("Instala websockets: pip install websockets")
        return

    demo_path = _DEMO_JSON if _DEMO_JSON.exists() else \
        Path(__file__).parent.parent.parent / "app/stress_app/assets/models/demo_windows.json"
    stats = _compute_stats(demo_path)
    delay = 2.0 / speed

    print(f"[synthetic] modelo={model}  speed={speed}x  delay={delay:.2f}s", flush=True)
    async with websockets.connect(ws_url) as ws:
        print(f"[synthetic] Conectado a {ws_url}", flush=True)
        i = 0
        while True:
            cls      = _PATTERN[i % len(_PATTERN)]
            feat_dict = {
                f: stats[f][cls][0] + random.gauss(0, stats[f][cls][1])
                for f in FEATURES_ORDERED
            }
            await ws.send(json.dumps({"type": "window", "model": model, "features": feat_dict}))
            resp = json.loads(await ws.recv())
            gt   = "stress  " if cls == 1 else "baseline"
            print(f"  [{i+1:4d}] {gt} → {resp.get('label','?'):10s}  "
                  f"p={resp.get('proba', 0):.3f}", flush=True)
            i += 1
            await asyncio.sleep(delay)


async def stream_json_to_ws(json_path: Path, ws_url: str,
                            model: str = "SVM", speed: float = 1.0):
    """Envía ventanas desde demo_windows.json al WebSocket del backend."""
    try:
        import websockets
    except ImportError:
        print("Instala websockets: pip install websockets")
        return

    windows = json.loads(json_path.read_text())
    delay   = 2.0 / speed  # demo usa ventanas de 2s de separación

    async with websockets.connect(ws_url) as ws:
        print(f"Conectado a {ws_url}")
        print(f"JSON: {json_path.name}  |  {len(windows)} ventanas  |  x{speed} velocidad")
        for i, w in enumerate(windows):
            feat_dict = {f: float(w[f]) for f in FEATURES_ORDERED if f in w}
            msg = {"type": "window", "model": model, "features": feat_dict}
            await ws.send(json.dumps(msg))
            resp = await ws.recv()
            pred = json.loads(resp)
            label_gt = "stress" if w.get("label") == 1 else "baseline"
            print(f"  [{i+1:3d}/{len(windows)}] gt={label_gt:8s} → "
                  f"{pred.get('label','?'):10s}  p={pred.get('proba', '?'):.3f}")
            await asyncio.sleep(delay)
        print("Streaming completado.")


async def stream_csv_to_ws(subject_dir: Path, ws_url: str,
                           model: str = "SVM", speed: float = 1.0):
    """Lee señales de un directorio E4 Campanella y envía ventanas al backend."""
    try:
        import websockets
    except ImportError:
        print("Instala websockets: pip install websockets")
        return

    from pipeline.config import FEATURES_SELECTED
    from pipeline.loader_e4 import load_e4_subject
    from pipeline.windowing import build_dataset, STEP_S
    from pipeline.stats import normalize_intrasubject

    data    = load_e4_subject(subject_dir)
    df      = build_dataset({data["subject"]: data})
    feats   = [f for f in FEATURES_SELECTED if f in df.columns]
    df_norm = normalize_intrasubject(df, feats)
    delay   = STEP_S / speed

    async with websockets.connect(ws_url) as ws:
        print(f"Conectado a {ws_url}")
        print(f"Sujeto: {data['subject']}  |  {len(df_norm)} ventanas  |  x{speed} velocidad")
        for _, row in df_norm.iterrows():
            feat_dict = {f: float(row[f]) for f in feats if not np.isnan(row[f])}
            msg = {"type": "window", "model": model, "features": feat_dict}
            await ws.send(json.dumps(msg))
            resp = await ws.recv()
            pred = json.loads(resp)
            t = row["t_start_s"]
            print(f"  t={t/60:.1f}min  → {pred.get('label','?'):10s}  "
                  f"p={pred.get('proba', '?'):.3f}")
            await asyncio.sleep(delay)
        print("Streaming completado.")


def main():
    parser = argparse.ArgumentParser(description="E4 stream simulator → WebSocket")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Genera ventanas sintéticas infinitas con ruido gaussiano (recomendado para demo)")
    parser.add_argument("--demo-json", default=None,
                        help="Path al demo_windows.json")
    parser.add_argument("--subject",   default=None,
                        help="Directorio del sujeto E4 Campanella (EDA/BVP/ACC/TEMP.csv)")
    parser.add_argument("--ws",        default="ws://localhost:8000/ws/stream",
                        help="URL del WebSocket del backend")
    parser.add_argument("--model",     default="SVM",
                        choices=["RF", "SVM", "ANN", "XGBoost", "LDA"])
    parser.add_argument("--speed",     type=float, default=1.0,
                        help="Multiplicador de velocidad (1.0 = tiempo real, 5.0 = 5× más rápido)")
    parser.add_argument("--loop",      action="store_true",
                        help="Repetir en bucle hasta Ctrl+C")
    args = parser.parse_args()

    if not args.synthetic and args.demo_json is None and args.subject is None:
        parser.print_help()
        sys.exit(1)

    if args.synthetic:
        while True:
            try:
                asyncio.run(stream_synthetic_to_ws(args.ws, args.model, args.speed))
            except Exception as exc:
                print(f"[synthetic] Desconectado: {exc} — reconectando en 2s...", flush=True)
                import time; time.sleep(2)
    elif args.demo_json:
        if args.loop:
            run = 0
            while True:
                run += 1
                print(f"\n── Pasada {run} ──")
                try:
                    asyncio.run(stream_json_to_ws(
                        Path(args.demo_json), args.ws, args.model, args.speed
                    ))
                except Exception as exc:
                    print(f"  Error: {exc} — reintentando en 2s...")
                    import time; time.sleep(2)
        else:
            asyncio.run(stream_json_to_ws(
                Path(args.demo_json), args.ws, args.model, args.speed
            ))
    else:
        asyncio.run(stream_csv_to_ws(
            Path(args.subject), args.ws, args.model, args.speed
        ))


if __name__ == "__main__":
    main()
