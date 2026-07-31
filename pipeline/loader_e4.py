"""
Carga de datos E4 en formato CSV (Campanella 2024, Empatica E4).
Archivos sin cabecera: cada fila es una muestra directamente.

Referencia: Campanella et al., Sensors 2023. doi:10.3390/s23073565
"""
import numpy as np
from pathlib import Path


def _load_1d_csv(path: Path, dtype=np.float32) -> np.ndarray:
    """
    Carga CSV de una columna tolerando notación europea con separador de miles.
    '1.038.145' → 1.038145  (el punto de miles se fusiona con el decimal).
    """
    values = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.count('.') >= 2:
                parts = s.split('.')
                s = parts[0] + '.' + ''.join(parts[1:])
            values.append(float(s))
    return np.array(values, dtype=dtype)

FS_E4 = {'EDA': 4.0, 'BVP': 64.0, 'ACC': 32.0, 'TEMP': 4.0}

# Protocolo Campanella: (nombre, duracion_s, label)
# Labels WESAD-compatibles: 1=no-estrés, 2=estrés → LABEL_MAP = {1:0, 2:1}
# None = duración variable (Task4, sin límite de tiempo)
CAMPANELLA_PROTOCOL = [
    ('Rest1', 180,  1),   # 3 min  → no-estrés
    ('Task1', 600,  2),   # 10 min → estrés (Lego sin instrucciones)
    ('Rest2', 120,  1),   # 2 min
    ('Task2', 300,  2),   # 5 min  → estrés (Lego con instrucciones)
    ('Rest3', 120,  1),   # 2 min
    ('Task3', 180,  2),   # 3 min  → estrés (Lego grande + contar 180→0)
    ('Rest4', 120,  1),   # 2 min
    ('Task4', None, 2),   # variable → estrés (511-13, sin límite)
    ('Rest5', 120,  1),   # 2 min
    ('Task5',  60,  2),   # 1 min  → estrés (presentación CV)
    ('Rest6', 120,  1),   # 2 min
]

# Suma de fases con duración fija: 180+600+120+300+120+180+120+120+60+120 = 1920 s
_FIXED_DURATION_S = sum(d for _, d, _ in CAMPANELLA_PROTOCOL if d is not None)


def build_campanella_labels(n_samples: int, fs: float = 4.0) -> np.ndarray:
    """
    Reconstruye etiquetas a la frecuencia fs para n_samples muestras.
    La duración de Task4 se infiere del total: task4_s = total_s - 1920.
    Valores: 1 = no-estrés, 2 = estrés (WESAD-compatibles).
    """
    total_s = n_samples / fs
    task4_s = total_s - _FIXED_DURATION_S
    if task4_s < 30:
        raise ValueError(
            f"Task4 inferida = {task4_s:.1f}s < 30s "
            f"(total={total_s:.0f}s, fijo={_FIXED_DURATION_S}s). "
            "Verifica que EDA.csv esté completo."
        )

    labels = np.zeros(n_samples, dtype=np.int32)
    t = 0.0
    for _, dur, lbl in CAMPANELLA_PROTOCOL:
        d_s   = task4_s if dur is None else float(dur)
        start = int(round(t * fs))
        end   = min(int(round((t + d_s) * fs)), n_samples)
        labels[start:end] = lbl
        t += d_s
    return labels


def load_e4_subject(subject_dir) -> dict:
    """
    Carga señales E4 y genera etiquetas de protocolo para un sujeto Campanella.
    Formato de salida compatible con load_wesad_subject() de pipeline/windowing.py.

    Claves del dict:
        subject, EDA, BVP, TEMP, ACC (arrays numpy)
        label_EDA, label_BVP, label_ACC, label_TEMP (etiquetas sincronizadas)
        labels_700: None (no hay señal de pecho en Campanella)
    """
    d          = Path(subject_dir)
    subject_id = d.name

    eda  = _load_1d_csv(d / 'EDA.csv')
    bvp  = np.loadtxt(d / 'BVP.csv',  dtype=np.float32)
    acc  = np.loadtxt(d / 'ACC.csv',  dtype=np.float32, delimiter=',')
    temp = np.loadtxt(d / 'TEMP.csv', dtype=np.float32)

    label_eda = build_campanella_labels(len(eda), FS_E4['EDA'])

    def _sync(lbl_ref, target_len):
        """Interpola etiquetas de referencia (4Hz) a target_len muestras."""
        src = len(lbl_ref)
        if src == target_len:
            return lbl_ref.copy()
        idx = np.round(np.arange(target_len) * (src - 1) / max(target_len - 1, 1)).astype(int)
        return lbl_ref[idx]

    return {
        'subject':    subject_id,
        'EDA':        eda,
        'BVP':        bvp,
        'TEMP':       temp,
        'ACC':        acc,
        'labels_700': None,
        'label_EDA':  label_eda,
        'label_BVP':  _sync(label_eda, len(bvp)),
        'label_ACC':  _sync(label_eda, len(acc)),
        'label_TEMP': label_eda.copy(),
    }


def load_campanella_all(campanella_root) -> dict:
    """
    Carga todos los sujetos del directorio Campanella.
    Returns: dict {subject_id: data_dict}
    """
    root = Path(campanella_root)
    subject_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not subject_dirs:
        raise FileNotFoundError(f"No se encontraron subdirectorios en {root}")

    all_data = {}
    for sd in subject_dirs:
        try:
            all_data[sd.name] = load_e4_subject(sd)
        except Exception as exc:
            print(f"  AVISO: {sd.name} omitido — {exc}")
    return all_data
