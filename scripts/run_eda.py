#!/usr/bin/env python
"""
Pipeline EDA automatizado — preprocesamiento y construcción del dataset ventaneado.

Uso:
  conda run -n tesis python scripts/run_eda.py
  conda run -n tesis python scripts/run_eda.py --data data/WESAD/WESAD --out eda_outputs
"""
import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

from pipeline.features import ALL_FEAT_KEYS
from pipeline.windowing import FS_LABEL, LABEL_MAP, load_wesad_subject, build_dataset
from pipeline.stats import run_statistical_tests, normalize_intrasubject
from pipeline.plots import (
    plot_raw_signals, plot_class_balance, plot_preprocessing,
    plot_distributions, plot_separability, plot_correlation,
)


def main():
    parser = argparse.ArgumentParser(description='Pipeline EDA automatizado WESAD')
    parser.add_argument('--data', type=Path, default=Path('data/WESAD/WESAD'),
                        help='Directorio raíz del dataset (contiene S2/, S3/, ...)')
    parser.add_argument('--out',  type=Path, default=Path('eda_outputs'),
                        help='Directorio de salida')
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'figures').mkdir(exist_ok=True)

    subjects = sorted([d.name for d in args.data.iterdir() if d.is_dir()])
    if not subjects:
        raise ValueError(f"No se encontraron sujetos en {args.data}")
    print(f"Sujetos encontrados ({len(subjects)}): {subjects}")

    # 1. Carga
    print("\n[1/6] Cargando datos...")
    all_data = {}
    for subj in subjects:
        all_data[subj] = load_wesad_subject(args.data, subj)
        lbl = all_data[subj]['labels_700']
        print(f"  {subj}: baseline={(lbl==1).sum()/FS_LABEL:.0f}s  stress={(lbl==2).sum()/FS_LABEL:.0f}s")

    # 2. Figuras de señales y balance
    print("\n[2/6] Generando figuras de señales y balance...")
    first = list(all_data.values())[0]
    plot_raw_signals(first, out_dir)
    plot_class_balance(all_data, out_dir)
    plot_preprocessing(first, out_dir)

    # 3. Ventaneo y extracción de features
    print("\n[3/6] Extrayendo features (ventanas 60s, 50% overlap)...")
    windows_df = build_dataset(all_data)
    windows_df.to_csv(out_dir / 'dataset_windowed.csv', index=False)
    n0 = (windows_df.label == 0).sum()
    n1 = (windows_df.label == 1).sum()
    print(f"  Ventanas: {len(windows_df)}  (baseline={n0}, stress={n1}, ratio={n0/max(n1,1):.2f}:1)")

    # 4. Pruebas estadísticas (Mann-Whitney U + Cohen's d)
    print("\n[4/6] Mann-Whitney U + Cohen's d...")
    test_df = run_statistical_tests(windows_df)
    test_df.to_csv(out_dir / 'statistical_tests.csv', index=False)
    print(f"\n  {'Feature':25s} {'p-valor':>10} {'d Cohen':>9} {'Efecto':>14} {'Sig':>5}")
    print('  ' + '-' * 65)
    for _, row in test_df.iterrows():
        sig = '✓' if row['significant'] else '✗'
        print(f"  {row['feature']:25s} {row['p_value']:>10.4f} {row['cohens_d']:>9.3f} "
              f"{row['effect']:>14} {sig:>5}")

    # 5. Figuras de features
    print("\n[5/6] Generando figuras de features...")
    plot_distributions(windows_df, out_dir)
    plot_separability(windows_df, test_df, out_dir)
    plot_correlation(windows_df, ALL_FEAT_KEYS, out_dir)

    # 6. Normalización z-score intra-sujeto
    print("\n[6/6] Normalización z-score intra-sujeto...")
    nan_pct = windows_df[ALL_FEAT_KEYS].isna().mean() * 100
    available_feats = [f for f in ALL_FEAT_KEYS if nan_pct[f] < 20]
    windows_norm = normalize_intrasubject(windows_df, available_feats)
    windows_norm.to_csv(out_dir / 'dataset_windowed_normalized.csv', index=False)

    sig_n   = int(test_df['significant'].sum())
    large_n = int((test_df['effect'] == 'grande').sum())
    print('\n' + '=' * 65)
    print('RESUMEN EDA COMPLETADO')
    print('=' * 65)
    print(f'Sujetos        : {len(all_data)}')
    print(f'Ventanas       : {len(windows_df)} (baseline={n0}, stress={n1})')
    print(f'Sig. (p<0.05)  : {sig_n}/{len(test_df)}  |  Efecto grande: {large_n}')
    print(f'Dataset norm.  : {out_dir / "dataset_windowed_normalized.csv"}')


if __name__ == '__main__':
    main()
