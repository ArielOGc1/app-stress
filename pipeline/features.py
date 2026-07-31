import numpy as np
import neurokit2 as nk
from .preprocess import preprocess_eda, preprocess_bvp

EDA_KEYS  = ['scl_mean', 'scl_std', 'scr_count', 'scr_density',
             'scr_amp_mean', 'phasic_mean', 'phasic_energy']
ACC_KEYS  = ['acc_mag_mean', 'acc_mag_std', 'acc_mag_energy', 'acc_movement']
HRV_KEYS  = ['hrv_mean_rr', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50']
TEMP_KEYS = ['temp_mean']
ALL_FEAT_KEYS = EDA_KEYS + ACC_KEYS + HRV_KEYS + TEMP_KEYS


def extract_eda_features(eda_raw: np.ndarray, fs: float = 4.0) -> dict:
    empty = {k: np.nan for k in EDA_KEYS}
    if len(eda_raw) < 8:
        return empty
    try:
        _, eda_tonic, eda_phasic = preprocess_eda(eda_raw, fs)
        feats = {
            'scl_mean':      float(np.mean(eda_tonic)),
            'scl_std':       float(np.std(eda_tonic)),
            'phasic_mean':   float(np.mean(eda_phasic)),
            'phasic_energy': float(np.sum(eda_phasic**2) / len(eda_phasic)),
        }
        _, info = nk.eda_process(eda_raw, sampling_rate=int(fs))
        scr_peaks   = info.get('SCR_Peaks', [])
        scr_heights = info.get('SCR_Height', [])
        n_scr = len(scr_peaks) if scr_peaks is not None else 0
        dur_s = len(eda_raw) / fs
        feats['scr_count']    = n_scr
        feats['scr_density']  = n_scr / (dur_s / 60.0) if dur_s > 0 else 0.0
        feats['scr_amp_mean'] = float(np.nanmean(scr_heights)) if n_scr > 0 else 0.0
        return feats
    except Exception:
        return empty


def extract_acc_features(acc_raw: np.ndarray) -> dict:
    empty = {k: np.nan for k in ACC_KEYS}
    if acc_raw.shape[0] < 2:
        return empty
    try:
        mag = np.sqrt((acc_raw**2).sum(axis=1))
        return {
            'acc_mag_mean':   float(np.mean(mag)),
            'acc_mag_std':    float(np.std(mag)),
            'acc_mag_energy': float(np.mean(mag**2)),
            'acc_movement':   float(np.sum(np.abs(np.diff(mag))) / len(mag)),
        }
    except Exception:
        return empty


def extract_hrv_from_bvp(bvp_raw: np.ndarray, fs: float = 64.0) -> dict:
    empty = {k: np.nan for k in HRV_KEYS}
    if len(bvp_raw) < fs * 10:
        return empty
    try:
        bvp_f = preprocess_bvp(bvp_raw, fs)
        _, info = nk.ppg_process(bvp_f, sampling_rate=int(fs))
        peaks = info.get('PPG_Peaks', [])
        if peaks is None or len(peaks) < 5:
            return empty
        rr_ms = np.diff(peaks) / fs * 1000
        rr_ms = rr_ms[(rr_ms >= 300) & (rr_ms <= 2000)]
        if len(rr_ms) < 4:
            return empty
        diff_rr = np.diff(rr_ms)
        return {
            'hrv_mean_rr': float(np.mean(rr_ms)),
            'hrv_sdnn':    float(np.std(rr_ms, ddof=1)),
            'hrv_rmssd':   float(np.sqrt(np.mean(diff_rr**2))),
            'hrv_pnn50':   float(np.sum(np.abs(diff_rr) > 50) / len(diff_rr) * 100),
        }
    except Exception:
        return empty


def extract_temp_features(temp_raw: np.ndarray) -> dict:
    if len(temp_raw) == 0:
        return {'temp_mean': np.nan}
    return {'temp_mean': float(np.nanmean(temp_raw))}
