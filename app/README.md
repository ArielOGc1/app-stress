# Stress Classifier App

Aplicación móvil Flutter + backend Python FastAPI para clasificación automática de estados de estrés mediante señales fisiológicas del Empatica E4.

---

## Arquitectura

```
┌─────────────────────────────────┐      HTTP/WebSocket
│  Flutter App (stress_app/)      │ ◄──────────────────► FastAPI Backend
│  ├─ Pantalla CSV upload         │                       (backend/)
│  ├─ Pantalla BLE / Simulación   │                       Pipeline Python
│  └─ Ajustes (IP del servidor)   │                       Modelos WESAD
└─────────────────────────────────┘
```

---

## Setup

### 1. Backend (Python)

```bash
# Instalar dependencias
conda run -n tesis pip install fastapi "uvicorn[standard]" python-multipart aiofiles

# Iniciar el servidor (desde la raíz de tesis_1/)
conda run -n tesis uvicorn app.backend.main:app --host 0.0.0.0 --port 8000

# Verificar
curl http://localhost:8000/health
```

El servidor carga los 5 modelos al iniciar (RF, SVM, ANN, XGBoost, LDA).

---

### 2. Flutter (App móvil)

#### Instalar Flutter SDK (Linux)

```bash
# Descargar Flutter
cd ~
wget https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.27.4-stable.tar.xz
tar xf flutter_linux_3.27.4-stable.tar.xz

# Agregar al PATH (añadir a ~/.bashrc)
echo 'export PATH="$PATH:$HOME/flutter/bin"' >> ~/.bashrc
source ~/.bashrc

# Verificar
flutter doctor
```

#### Instalar Android SDK

```bash
# Opción A: Android Studio (recomendado)
# Descargar desde https://developer.android.com/studio

# Opción B: Solo command-line tools
mkdir -p ~/Android/cmdline-tools
cd ~/Android/cmdline-tools
wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip commandlinetools-linux-*.zip -d latest
mv latest/cmdline-tools latest/tools  # Mover al lugar correcto
mv latest ~/Android/cmdline-tools/
echo 'export ANDROID_HOME=$HOME/Android' >> ~/.bashrc
echo 'export PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin' >> ~/.bashrc
source ~/.bashrc
sdkmanager --install "platform-tools" "platforms;android-34" "build-tools;34.0.0"
```

#### Compilar y ejecutar la app

```bash
cd app/stress_app

# Obtener dependencias
flutter pub get

# Verificar configuración
flutter doctor

# Ejecutar en dispositivo conectado (USB debugging activado)
flutter run

# Compilar APK para instalar manualmente
flutter build apk --release
# APK en: build/app/outputs/flutter-apk/app-release.apk
```

---

## Uso

### Modo CSV (análisis de sesión completa)

1. Abre la app → **"Cargar Archivos CSV"**
2. Selecciona los 4 archivos exportados del E4: `EDA.csv`, `BVP.csv`, `ACC.csv`, `TEMP.csv`
3. Elige el modelo (recomendado: SVM)
4. Toca **"Analizar"**
5. Visualiza el gráfico de probabilidad de estrés y las predicciones por ventana

### Modo BLE / Simulación

**Simulación (sin hardware E4):**

1. Abre la app → **"Monitoreo en Tiempo Real"** → pestaña **"Simulación"**
2. Selecciona el modelo
3. En la PC, inicia el streamer en otra terminal:
   ```bash
   conda run -n tesis python app/backend/streamer.py \
       --subject data/Data_29_subjects/Subjects/subject_01 \
       --ws ws://localhost:8000/ws/stream \
       --speed 5.0
   ```
4. Toca **"Iniciar simulación"** en la app
5. Observa las predicciones en tiempo real

**BLE directo (con E4):**

Requiere E4 Streaming Server corriendo en la PC (Windows/Mac):
1. Abre la app → **"Monitoreo en Tiempo Real"** → pestaña **"BLE Directo"**
2. Toca **"Escanear"** para detectar el E4
3. Conecta el dispositivo

---

## Ajustes de red

La app necesita la IP del servidor backend. Si el teléfono y la PC están en la misma red WiFi:

1. En la PC: `ip addr` → busca la IP local (e.g., `192.168.1.100`)
2. En la app: **Ajustes** → URL del servidor → `http://192.168.1.100:8000`

---

## API del backend

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/health` | Estado del servidor y modelos |
| GET | `/models` | Modelos disponibles |
| POST | `/predict/csv` | Predicción desde 4 archivos CSV |
| POST | `/predict/window` | Predicción de ventana única (JSON) |
| WS | `/ws/stream` | WebSocket para streaming en tiempo real |

### Ejemplo curl

```bash
# Análisis de un sujeto Campanella
curl -X POST http://localhost:8000/predict/csv \
  -F "eda=@data/Data_29_subjects/Subjects/subject_01/EDA.csv" \
  -F "bvp=@data/Data_29_subjects/Subjects/subject_01/BVP.csv" \
  -F "acc=@data/Data_29_subjects/Subjects/subject_01/ACC.csv" \
  -F "temp=@data/Data_29_subjects/Subjects/subject_01/TEMP.csv" \
  -F "model=SVM"
```

---

## Estructura de archivos

```
app/
├── backend/
│   ├── main.py          ← FastAPI app (endpoints + WebSocket)
│   ├── streamer.py      ← Simulador de streaming E4
│   └── api/
│       └── inference.py ← Lógica de procesamiento y predicción
└── stress_app/
    ├── pubspec.yaml
    ├── android/         ← Permisos BLE + Internet
    └── lib/
        ├── main.dart
        ├── models/
        │   └── prediction_result.dart
        ├── screens/
        │   ├── home_screen.dart
        │   ├── csv_screen.dart
        │   ├── ble_screen.dart
        │   └── settings_screen.dart
        └── services/
            ├── api_service.dart   ← HTTP + WebSocket client
            └── ble_service.dart   ← BLE scan + streaming state
```
