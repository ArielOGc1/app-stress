import pandas as pd
from .config import ABLATION_FEATURE_SETS, RESULTS_DIR
from .data import load_dataset
from .loso import run_loso
from .metrics import aggregate_results
from .models import get_models


def run_ablation(path=None, results_dir=None):
    """
    Ejecuta LOSO para cada conjunto de features definido en ABLATION_FEATURE_SETS.

    Experimentos:
    - full_12      : las 12 features seleccionadas post-EDA (baseline de comparación)
    - no_acc       : sin acc_mag_std ni acc_movement (evalúa sesgo comportamental TSST)
    - hrv_mean_only: solo hrv_mean_rr (evalúa impacto del artefacto PPG en HRV)

    Parameters
    ----------
    path        : str | Path | None — CSV de entrada (None = WESAD por defecto)
    results_dir : Path | None       — directorio de salida (None = RESULTS_DIR de config)

    Returns
    -------
    fold_df    : DataFrame con resultados por fold de todos los experimentos
    summary_df : DataFrame con media ± std por experimento y modelo
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    all_folds   = []
    all_summary = []

    for experiment_name, feature_set in ABLATION_FEATURE_SETS.items():
        try:
            X, y, groups, features = load_dataset(
                feature_set=feature_set, path=path, strict=False
            )
        except ValueError as e:
            print(f"\n=== Ablación: {experiment_name} — SKIP ({e}) ===")
            continue

        if not features:
            print(f"\n=== Ablación: {experiment_name} — SKIP (ninguna feature disponible) ===")
            continue

        print(f"\n=== Ablación: {experiment_name} ({len(features)} features) ===")
        models = get_models()
        fold_df = run_loso(X, y, groups, models)
        fold_df['experiment'] = experiment_name

        summary = aggregate_results(fold_df)
        summary['experiment'] = experiment_name
        summary['n_features'] = len(feature_set)

        all_folds.append(fold_df)
        all_summary.append(summary.reset_index())

    fold_results    = pd.concat(all_folds,   ignore_index=True)
    summary_results = pd.concat(all_summary, ignore_index=True)

    out_folds   = results_dir / "ablation" / "ablation_fold_results.csv"
    out_summary = results_dir / "ablation" / "ablation_summary.csv"
    out_folds.parent.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(out_folds, index=False)
    summary_results.to_csv(out_summary, index=False)
    print(f"\nResultados guardados en:\n  {out_folds}\n  {out_summary}")

    return fold_results, summary_results


def print_ablation_comparison(summary_df, metric='f1_macro'):
    """
    Imprime tabla comparativa: experimento × modelo para la métrica indicada.
    """
    mean_col = f'{metric}_mean'
    std_col  = f'{metric}_std'
    pivot_mean = summary_df.pivot(index='model', columns='experiment', values=mean_col)
    pivot_std  = summary_df.pivot(index='model', columns='experiment', values=std_col)

    print(f"\nComparación de ablación — {metric} (mean ± std):\n")
    print(f"{'Modelo':<10}", end='')
    for col in pivot_mean.columns:
        print(f"  {col:<20}", end='')
    print()
    print("-" * (10 + 22 * len(pivot_mean.columns)))
    for model in pivot_mean.index:
        print(f"{model:<10}", end='')
        for col in pivot_mean.columns:
            m = pivot_mean.loc[model, col]
            s = pivot_std.loc[model, col]
            print(f"  {m:.3f} ± {s:.3f}         ", end='')
        print()
