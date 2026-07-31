"""
Ablation studies: evalúa el impacto de ACC y HRV en la performance.

Uso (WESAD):
    conda run -n tesis python scripts/run_ablation.py

Uso con dataset externo:
    conda run -n tesis python scripts/run_ablation.py --csv ruta.csv --name campanella
    → guarda en results/campanella/ablation/
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.ablation import print_ablation_comparison, run_ablation
from pipeline.config import RESULTS_DIR
from pipeline.plots import plot_ablation_bars


def parse_args():
    parser = argparse.ArgumentParser(description='Ablation studies del pipeline.')
    parser.add_argument('--csv',  default=None, metavar='RUTA',
                        help='CSV de entrada (default: WESAD normalizado).')
    parser.add_argument('--name', default=None, metavar='NOMBRE',
                        help='Nombre del dataset → subcarpeta en results/.')
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = RESULTS_DIR / args.name if args.name else RESULTS_DIR
    dataset_label = args.name.upper() if args.name else 'WESAD'

    print('=' * 60)
    print(f'  ABLATION STUDIES — {dataset_label}')
    print('=' * 60)
    print(f'  Salida: {out_dir / "ablation"}')

    fold_df, summary_df = run_ablation(path=args.csv, results_dir=out_dir)

    print('\n--- Comparación por F1 macro ---')
    print_ablation_comparison(summary_df, metric='f1_macro')

    print('\n--- Comparación por Balanced Accuracy ---')
    print_ablation_comparison(summary_df, metric='balanced_accuracy')

    print('\n[Figuras] Generando barplots...')
    plot_ablation_bars(summary_df, out_dir, metric='f1_macro')
    plot_ablation_bars(summary_df, out_dir, metric='balanced_accuracy')

    print('\nAblation completado.')


if __name__ == '__main__':
    main()
