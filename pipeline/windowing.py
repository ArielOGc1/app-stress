import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from .features import (
    ALL_FEAT_KEYS,
    extract_eda_features, extract_acc_features,
    extract_hrv_from_bvp, extract_temp_features,
)

FS        = {'EDA': 4.0, 'BVP': 64.0, 'ACC': 32.0, 'TEMP': 4.0}
FS_LABEL  = 700
LABEL_MAP = {1: 0, 2: 1}   # baseline→0, stress→1
WINDOW_S  = 60
STEP_S    = 30
MIN_PURITY = 0.90


def sync_labels(labels_700: np.ndarray, fs_signal: float) -> np.ndarray:
    """Downsample labels de 700 Hz al fs de la señal objetivo usando índices temporales."""
    step = FS_LABEL / fs_signal
    idx  = np.round(np.arange(0, len(labels_700), step)).astype(int)
    idx  = idx[idx < len(labels_700)]
    return labels_700[idx]


def window_label(labels_signal: np.ndarray, start: int, end: int) -> int:
    """
    Label de una ventana por mayoría (≥90% mismo label).
    Retorna -1 si la ventana es mixta o contiene solo label 0 (undefined).
    """
    seg = labels_signal[start:end]
    if len(seg) == 0:
        return -1
    if (seg == 0).any():
        non_zero = seg[seg != 0]
        if len(non_zero) / len(seg) < MIN_PURITY:
            return -1
        seg = non_zero
    vals, counts = np.unique(seg, return_counts=True)
    dominant = vals[np.argmax(counts)]
    if counts.max() / len(seg) < MIN_PURITY:
        return -1
    return int(dominant)


def load_wesad_subject(wesad_dir: Path, subject_id: str) -> dict:
    """
    Carga el pickle de un sujeto WESAD y sincroniza los labels al fs de cada señal wrist.
    """
    pkl_path = wesad_dir / subject_id / f'{subject_id}.pkl'
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    labels_700 = raw['label'].astype(int)
    wrist = raw['signal']['wrist']
    return {
        'subject':    subject_id,
        'EDA':        wrist['EDA'].flatten(),
        'BVP':        wrist['BVP'].flatten(),
        'TEMP':       wrist['TEMP'].flatten(),
        'ACC':        wrist['ACC'],
        'labels_700': labels_700,
        'label_EDA':  sync_labels(labels_700, FS['EDA']),
        'label_BVP':  sync_labels(labels_700, FS['BVP']),
        'label_ACC':  sync_labels(labels_700, FS['ACC']),
        'label_TEMP': sync_labels(labels_700, FS['TEMP']),
    }


def build_dataset(all_data: dict) -> pd.DataFrame:
    """
    Construye el dataset ventaneado extrayendo features de cada ventana.

    Parameters
    ----------
    all_data : dict  {subject_id: dict cargado con load_wesad_subject}

    Returns
    -------
    DataFrame con columnas: subject, label, raw_label, window_id, t_start_s, <features>
    """
    records = []
    for subj, d in all_data.items():
        win_n_eda  = int(WINDOW_S * FS['EDA'])
        step_n_eda = int(STEP_S   * FS['EDA'])
        win_n_bvp  = int(WINDOW_S * FS['BVP'])
        win_n_acc  = int(WINDOW_S * FS['ACC'])

        win_idx = 0
        for i_eda in range(0, len(d['EDA']) - win_n_eda + 1, step_n_eda):
            raw_label = window_label(d['label_EDA'], i_eda, i_eda + win_n_eda)
            if raw_label not in LABEL_MAP:
                win_idx += 1
                continue
            class_label = LABEL_MAP[raw_label]
            i_bvp = int(i_eda * FS['BVP'] / FS['EDA'])
            i_acc = int(i_eda * FS['ACC'] / FS['EDA'])

            rec = {
                'subject':   subj,
                'label':     class_label,
                'raw_label': raw_label,
                'window_id': win_idx,
                't_start_s': round(i_eda / FS['EDA'], 1),
            }
            rec.update(extract_eda_features(d['EDA'][i_eda:i_eda + win_n_eda], FS['EDA']))
            rec.update(extract_acc_features(d['ACC'][i_acc:i_acc + win_n_acc]))
            rec.update(extract_hrv_from_bvp(d['BVP'][i_bvp:i_bvp + win_n_bvp], FS['BVP']))
            rec.update(extract_temp_features(d['TEMP'][i_eda:i_eda + win_n_eda]))
            records.append(rec)
            win_idx += 1

    return pd.DataFrame(records)
