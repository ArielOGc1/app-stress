"""
Backend FastAPI — Pipeline de clasificación de estrés.

Endpoints:
    POST /predict/csv      — carga 4 archivos CSV (EDA, BVP, ACC, TEMP), retorna predicciones
    POST /predict/window   — ventana única como JSON (modo streaming/BLE)
    GET  /health           — estado del servidor y modelos cargados
    GET  /models           — lista de modelos disponibles
    WS   /ws/stream        — WebSocket para streaming en tiempo real (streamer.py → backend)
    WS   /ws/monitor       — WebSocket solo-lectura para el celular (modo Streaming PC)

Uso:
    conda run -n tesis uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
"""
import asyncio
import io
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEMO_JSON    = _PROJECT_ROOT / "app/stress_app/assets/models/demo_windows.json"
_STREAMER_PY  = _PROJECT_ROOT / "app/backend/streamer.py"

# ── Auto-streamer ────────────────────────────────────────────────────────────

class _AutoStreamer:
    """
    Lanza streamer.py automáticamente cuando el primer celular se conecta
    a /ws/monitor y lo detiene cuando el último desconecta.
    """
    def __init__(self):
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._clients: int = 0
        self._mode: str = "synthetic"
        self._sid:  str = "S4"

    async def on_connect(self, mode: str = "synthetic", sid: str = "S4"):
        self._mode = mode
        self._sid  = sid
        self._clients += 1
        if self._clients == 1:
            self._task = asyncio.create_task(self._run_loop())

    async def on_disconnect(self):
        self._clients = max(0, self._clients - 1)
        if self._clients == 0:
            await self._stop()

    def _build_streamer_args(self) -> list[str]:
        base = [sys.executable, str(_STREAMER_PY),
                "--ws",    "ws://localhost:8000/ws/stream",
                "--model", "SVM",
                "--speed", "3.0"]
        if self._mode == "subject":
            subject_json = _PROJECT_ROOT / f"app/stress_app/assets/models/demo_{self._sid}.json"
            if subject_json.exists():
                return base + ["--demo-json", str(subject_json), "--loop"]
            # fallback: synthetic si no existe JSON del sujeto
        return base + ["--synthetic"]

    async def _run_loop(self):
        while self._clients > 0:
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    *self._build_streamer_args(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await self._proc.wait()
            except Exception as exc:
                print(f"[autostreamer] error: {exc}", flush=True)
            self._proc = None
            if self._clients > 0:
                await asyncio.sleep(0.5)

    async def _stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._proc and self._proc.returncode is None:
            self._proc.terminate()
            self._proc = None

_auto_streamer = _AutoStreamer()

# ── Monitor broadcast manager ────────────────────────────────────────────────

class _MonitorManager:
    """Mantiene conexiones de celulares que solo escuchan predicciones."""
    def __init__(self):
        self._clients: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self._clients:
            self._clients.remove(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self._clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.remove(ws)

_monitors = _MonitorManager()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.config import FEATURES_SELECTED
from pipeline.loader_e4 import load_e4_subject, build_campanella_labels, FS_E4
from pipeline.stats import normalize_intrasubject
from pipeline.windowing import build_dataset
from app.backend.api.inference import run_pipeline_on_files, predict_window

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos al iniciar el servidor."""
    print("[startup] Cargando modelos...", flush=True)
    for name in ("RF", "SVM", "ANN", "XGBoost", "LDA"):
        p = MODELS_DIR / f"{name}.joblib"
        if p.exists():
            _models[name] = joblib.load(p)
            print(f"  {name} ✓", flush=True)
    print(f"[startup] {len(_models)} modelos listos.", flush=True)
    yield
    _models.clear()


app = FastAPI(
    title="Stress Classifier API",
    description="Pipeline WESAD — clasificación de estrés via señales fisiológicas E4",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health & info ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "models":  list(_models.keys()),
        "features": FEATURES_SELECTED,
    }


@app.get("/models")
def models():
    return {"available": list(_models.keys())}


# ── Predicción desde archivos CSV ────────────────────────────────────────────

@app.post("/predict/csv")
async def predict_csv(
    eda:   UploadFile = File(..., description="EDA.csv — 4Hz"),
    bvp:   UploadFile = File(..., description="BVP.csv — 64Hz"),
    acc:   UploadFile = File(..., description="ACC.csv — 32Hz, 3 columnas"),
    temp:  UploadFile = File(..., description="TEMP.csv — 4Hz"),
    model: str        = Form("SVM", description="Modelo: RF | SVM | ANN | XGBoost | LDA"),
):
    if model not in _models:
        raise HTTPException(400, f"Modelo '{model}' no disponible. Usa: {list(_models.keys())}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for upload, name in [(eda, "EDA.csv"), (bvp, "BVP.csv"),
                             (acc, "ACC.csv"), (temp, "TEMP.csv")]:
            (tmp / name).write_bytes(await upload.read())

        try:
            result = run_pipeline_on_files(tmp, _models[model], FEATURES_SELECTED)
        except Exception as exc:
            raise HTTPException(422, f"Error procesando señales: {exc}")

    return result


# ── Predicción de ventana única (streaming / BLE) ───────────────────────────

@app.post("/predict/window")
async def predict_single_window(
    payload: dict,
    model: str = "SVM",
):
    """
    Recibe una ventana pre-extraída como JSON:
        {"features": {"scl_mean": 0.12, "hrv_mean_rr": 820, ...}}
    Retorna la predicción inmediata.
    """
    if model not in _models:
        raise HTTPException(400, f"Modelo '{model}' no disponible.")

    feats = payload.get("features", {})
    try:
        result = predict_window(feats, _models[model], FEATURES_SELECTED)
    except Exception as exc:
        raise HTTPException(422, f"Error en predicción: {exc}")
    return result


# ── WebSocket — streaming en tiempo real ────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Protocolo WebSocket para streaming BLE:

    Cliente envía JSON:
        {"type": "window", "model": "SVM", "features": {...}}
        {"type": "ping"}

    Servidor responde JSON:
        {"type": "prediction", "class": 1, "label": "stress", "proba": 0.87}
        {"type": "pong"}
        {"type": "error", "detail": "..."}
    """
    import json
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "window")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            model_name = data.get("model", "SVM")
            if model_name not in _models:
                await websocket.send_json({
                    "type": "error",
                    "detail": f"Modelo '{model_name}' no disponible."
                })
                continue

            feats = data.get("features", {})
            try:
                pred = predict_window(feats, _models[model_name], FEATURES_SELECTED)
                await websocket.send_json({"type": "prediction", **pred})
                await _monitors.broadcast({"type": "prediction", **pred})
            except Exception as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})

    except WebSocketDisconnect:
        pass


@app.websocket("/ws/monitor")
async def ws_monitor(websocket: WebSocket, mode: str = "synthetic", sid: str = "S4"):
    """
    El celular se conecta aquí en modo Streaming PC.
    mode: 'synthetic' | 'subject'
    sid:  sujeto WESAD, e.g. 'S4' (solo si mode='subject')
    """
    await _monitors.connect(websocket)
    await _auto_streamer.on_connect(mode=mode, sid=sid)
    try:
        while True:
            await websocket.receive_text()   # mantiene la conexión viva
    except WebSocketDisconnect:
        pass
    finally:
        _monitors.disconnect(websocket)
        await _auto_streamer.on_disconnect()
