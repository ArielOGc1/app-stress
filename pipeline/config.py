from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "eda_outputs"
RESULTS_DIR = BASE_DIR / "results"

DATASET_NORMALIZED = DATA_DIR / "dataset_windowed_normalized.csv"

# 16 features candidatas extraídas en EDA_WESAD.ipynb
FEATURES_ALL = [
    'scl_mean', 'scl_std', 'phasic_mean', 'phasic_energy',
    'scr_count', 'scr_density', 'scr_amp_mean',
    'acc_mag_mean', 'acc_mag_std', 'acc_mag_energy', 'acc_movement',
    'hrv_mean_rr', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
    'temp_mean',
]

# 12 features seleccionadas post-EDA (p<0.05, |d|>=0.5)
# scr_density eliminada: r=1.0 con scr_count en ventanas de duración fija (60s)
# scr_density = scr_count/60 → tras z-score intra-sujeto son idénticas
FEATURES_SELECTED = [
    'scl_mean', 'scl_std', 'scr_count', 'scr_amp_mean', 'phasic_energy',
    'acc_mag_std', 'acc_movement',
    'hrv_mean_rr', 'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
    'temp_mean',
]

# Ablación 1: sin ACC (evaluar sesgo comportamental del TSST)
FEATURES_NO_ACC = [f for f in FEATURES_SELECTED if f not in ('acc_mag_std', 'acc_movement')]

# Ablación 2: solo hrv_mean_rr (evaluar artefacto PPG en HRV)
FEATURES_HRV_MEAN_ONLY = [f for f in FEATURES_SELECTED if f not in ('hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50')]

ABLATION_FEATURE_SETS = {
    'full_12':        FEATURES_SELECTED,
    'no_acc':         FEATURES_NO_ACC,
    'hrv_mean_only':  FEATURES_HRV_MEAN_ONLY,
}

# Ratio clase mayoritaria / minoritaria (562 baseline / 307 stress)
CLASS_RATIO = 562 / 307

RANDOM_STATE = 42

# Hiperparámetros
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

SVM_PARAMS = dict(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=RANDOM_STATE,
)

ANN_PARAMS = dict(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    max_iter=500,
    early_stopping=True,
    n_iter_no_change=15,
    random_state=RANDOM_STATE,
)

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=CLASS_RATIO,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
)
