import numpy as np
from scipy.signal import butter, sosfilt, savgol_filter, cheby2


def preprocess_eda(eda_raw: np.ndarray, fs: float = 4.0):
    """
    Butterworth 5th order LPF 1.99 Hz → Savitzky-Golay (ventana 4s).
    Returns: (eda_filtered, eda_tonic, eda_phasic)
    """
    nyq = fs / 2
    sos = butter(N=5, Wn=1.99 / nyq, btype='low', output='sos')
    eda_f = sosfilt(sos, eda_raw)
    wl = int(fs * 4)
    if wl % 2 == 0:
        wl += 1
    eda_tonic  = savgol_filter(eda_f, window_length=wl, polyorder=3)
    eda_phasic = eda_f - eda_tonic
    return eda_f, eda_tonic, eda_phasic


def preprocess_bvp(bvp_raw: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """Chebyshev Type II orden 4, bandpass 0.5–10 Hz."""
    nyq = fs / 2
    sos = cheby2(N=4, rs=40, Wn=[0.5 / nyq, 10.0 / nyq], btype='bandpass', output='sos')
    return sosfilt(sos, bvp_raw)
