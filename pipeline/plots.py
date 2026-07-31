"""
Funciones de visualización para EDA, LOSO y ablación.
Requiere matplotlib.use('Agg') antes de importar este módulo.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve

from .preprocess import preprocess_eda, preprocess_bvp
from .features import EDA_KEYS, ACC_KEYS, HRV_KEYS, TEMP_KEYS

COLORS = {
    'baseline': '#2196F3', 'stress': '#F44336',
    'c0': '#2196F3',       'c1': '#F44336',
}
MODEL_ORDER  = ['RF', 'SVM', 'ANN', 'XGBoost', 'LDA']
MODEL_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8E44AD']

FS = {'EDA': 4.0, 'BVP': 64.0, 'ACC': 32.0, 'TEMP': 4.0}


def _label_spans(lbl_arr, t_arr, target):
    spans, in_span, t0 = [], False, 0
    for t, v in zip(t_arr, lbl_arr):
        if v == target and not in_span:
            in_span, t0 = True, t
        elif v != target and in_span:
            spans.append((t0, t))
            in_span = False
    if in_span:
        spans.append((t0, t_arr[-1]))
    return spans


# ── EDA ───────────────────────────────────────────────────────────────────────
def plot_raw_signals(d: dict, out_dir: Path):
    """Señales wrist raw con regiones de label (primer sujeto como ejemplo)."""
    t_eda  = np.arange(len(d['EDA']))  / FS['EDA']  / 60
    t_bvp  = np.arange(len(d['BVP']))  / FS['BVP']  / 60
    t_acc  = np.arange(len(d['ACC']))  / FS['ACC']  / 60
    t_temp = np.arange(len(d['TEMP'])) / FS['TEMP'] / 60

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)
    fig.suptitle(f"Señales wrist Empatica E4 — {d['subject']} (raw)", fontsize=13, fontweight='bold')
    pairs = [
        (axes[0], t_eda,  d['EDA'],                           d['label_EDA'],  'EDA (μS)',   'EDA'),
        (axes[1], t_bvp,  d['BVP'],                           d['label_BVP'],  'BVP (a.u.)', 'BVP'),
        (axes[2], t_acc,  np.sqrt((d['ACC']**2).sum(axis=1)), d['label_ACC'],  'ACC mag',    'ACC'),
        (axes[3], t_temp, d['TEMP'],                          d['label_TEMP'], 'TEMP (°C)',  'TEMP'),
    ]
    for ax, t, sig, lbl, ylabel, name in pairs:
        ax.plot(t, sig, lw=0.5, color='#333', alpha=0.8)
        for t0, t1 in _label_spans(lbl, t, 1):
            ax.axvspan(t0, t1, alpha=0.15, color=COLORS['baseline'])
        for t0, t1 in _label_spans(lbl, t, 2):
            ax.axvspan(t0, t1, alpha=0.20, color=COLORS['stress'])
        ax.set_ylabel(ylabel); ax.set_xlabel('Tiempo (min)'); ax.set_title(name)
        ax.grid(True, alpha=0.3)
    p1 = mpatches.Patch(color=COLORS['baseline'], alpha=0.4, label='Baseline (clase 0)')
    p2 = mpatches.Patch(color=COLORS['stress'],   alpha=0.5, label='Stress TSST (clase 1)')
    axes[0].legend(handles=[p1, p2], loc='upper right')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / '01_seniales_raw.png', bbox_inches='tight')
    plt.close()


def plot_class_balance(all_data: dict, out_dir: Path,
                       label_map=None, label_names=None, fs_label: int = 700):
    if label_map   is None: label_map   = {1: 0, 2: 1}
    if label_names is None: label_names = {0: 'Baseline', 1: 'Stress (TSST)'}
    records = []
    for subj, d in all_data.items():
        lbl = d['labels_700']
        for orig, cls in label_map.items():
            records.append({'subject': subj,
                            'label_name': label_names[cls],
                            'duration_min': (lbl == orig).sum() / fs_label / 60})
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribución de clases — señales wrist', fontsize=12, fontweight='bold')
    pivot = df.pivot(index='subject', columns='label_name', values='duration_min')
    pivot.plot(kind='bar', ax=axes[0], color=[COLORS['baseline'], COLORS['stress']], alpha=0.8)
    axes[0].set_title('Duración por sujeto'); axes[0].set_ylabel('Minutos')
    axes[0].tick_params(axis='x', rotation=45); axes[0].grid(True, alpha=0.3, axis='y')
    totals = df.groupby('label_name')['duration_min'].sum()
    bars = axes[1].bar(totals.index, totals.values,
                       color=[COLORS['baseline'], COLORS['stress']], alpha=0.8)
    for bar, v in zip(bars, totals.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f'{v:.0f} min', ha='center', fontweight='bold')
    axes[1].set_title(f'Total acumulado ({len(all_data)} sujetos)')
    axes[1].set_ylabel('Minutos'); axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / '02_balance_clases.png', bbox_inches='tight')
    plt.close()


def plot_preprocessing(d: dict, out_dir: Path):
    """EDA tónico/fásico y BVP filtrado para el sujeto de ejemplo."""
    eda_f, eda_t, eda_p = preprocess_eda(d['EDA'], FS['EDA'])
    bvp_f = preprocess_bvp(d['BVP'], FS['BVP'])
    t_eda = np.arange(len(d['EDA'])) / FS['EDA'] / 60
    t_bvp = np.arange(len(d['BVP'])) / FS['BVP'] / 60
    lbl_e = d['label_EDA']

    fig, axes = plt.subplots(3, 1, figsize=(16, 9))
    fig.suptitle(f"Preprocesamiento de señales — {d['subject']}", fontsize=12, fontweight='bold')
    axes[0].plot(t_eda, d['EDA'], alpha=0.5, color='grey', lw=0.8, label='EDA raw')
    axes[0].plot(t_eda, eda_t, color='#E91E63', lw=1.5, label='Tónico (SCL)')
    axes[0].set_title('EDA — componente tónica (SCL)'); axes[0].set_ylabel('μS')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_eda, eda_p, color='#9C27B0', lw=0.8, label='EDA fásico (SCR)')
    axes[1].set_title('EDA — componente fásica (SCR)'); axes[1].set_ylabel('μS')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    for ax in axes[:2]:
        for t0, t1 in _label_spans(lbl_e, t_eda, 1):
            ax.axvspan(t0, t1, alpha=0.10, color=COLORS['baseline'])
        for t0, t1 in _label_spans(lbl_e, t_eda, 2):
            ax.axvspan(t0, t1, alpha=0.15, color=COLORS['stress'])
    seg = slice(0, int(30 * FS['BVP']))
    axes[2].plot(t_bvp[seg], d['BVP'][seg], alpha=0.4, color='grey', lw=0.6, label='BVP raw')
    axes[2].plot(t_bvp[seg], bvp_f[seg], color='#00796B', lw=1.2, label='BVP filtrado')
    axes[2].set_title('BVP — primeros 30 s (Chebyshev II bandpass 0.5–10 Hz)')
    axes[2].set_ylabel('a.u.'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    for ax in axes:
        ax.set_xlabel('Tiempo (min)')
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / '03_preprocesamiento.png', bbox_inches='tight')
    plt.close()


def plot_distributions(windows_df: pd.DataFrame, out_dir: Path):
    """Histogramas de densidad por clase para cada grupo de features."""
    feat_groups = {'EDA': EDA_KEYS, 'ACC': ACC_KEYS, 'HRV': HRV_KEYS, 'TEMP': TEMP_KEYS}
    for group, feats in feat_groups.items():
        avail = [f for f in feats if windows_df[f].notna().sum() > 10]
        if not avail:
            continue
        ncols = min(4, len(avail))
        nrows = (len(avail) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        axes = np.array(axes).flatten()
        fig.suptitle(f'Distribución por clase — Features {group}', fontsize=12, fontweight='bold')
        for idx, feat in enumerate(avail):
            ax = axes[idx]
            for cls, color, name in [(0, COLORS['c0'], 'Baseline'), (1, COLORS['c1'], 'Stress')]:
                vals = windows_df[windows_df.label == cls][feat].dropna()
                ax.hist(vals, bins=30, alpha=0.55, color=color, label=name, density=True)
            ax.set_title(feat, fontsize=9); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        for idx in range(len(avail), len(axes)):
            axes[idx].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_dir / 'figures' / f'04_dist_{group.lower()}.png', bbox_inches='tight')
        plt.close()


def plot_separability(windows_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path):
    """Violin plots de las 12 features seleccionadas (FEATURES_SELECTED), por |d de Cohen| descendente."""
    from .config import FEATURES_SELECTED
    sub_test = test_df[test_df.feature.isin(FEATURES_SELECTED)].copy()
    sub_test['abs_d'] = sub_test['cohens_d'].abs()
    top_feats = sub_test.sort_values('abs_d', ascending=False)['feature'].tolist()
    top_feats = [f for f in top_feats if windows_df[f].notna().sum() > 10]
    if not top_feats:
        return
    ncols = 4
    nrows = (len(top_feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.array(axes).flatten()
    fig.suptitle('Top features — separabilidad Baseline vs Stress TSST', fontsize=12, fontweight='bold')
    plot_df = windows_df[['label'] + top_feats].copy()
    plot_df['Clase'] = plot_df['label'].map({0: 'Baseline', 1: 'Stress'})
    for idx, feat in enumerate(top_feats):
        ax = axes[idx]
        sub = plot_df[['Clase', feat]].dropna()
        row_s = test_df[test_df.feature == feat].iloc[0]
        sns.violinplot(data=sub, x='Clase', y=feat, ax=ax,
                       palette={'Baseline': COLORS['c0'], 'Stress': COLORS['c1']},
                       inner='quartile', alpha=0.8)
        ax.set_title(f'{feat}\nd={row_s["cohens_d"]:.2f} ({row_s["effect"]})', fontsize=9)
        ax.set_xlabel(''); ax.grid(True, alpha=0.3, axis='y')
    for idx in range(len(top_feats), len(axes)):
        axes[idx].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / '05_separabilidad.png', bbox_inches='tight')
    plt.close()


def plot_correlation(windows_df: pd.DataFrame, feat_cols: list, out_dir: Path):
    """Heatmap de correlación de Spearman entre features."""
    corr_data = windows_df[feat_cols].dropna()
    if len(corr_data) <= 10:
        return
    corr_matrix = corr_data.corr(method='spearman')
    n = len(feat_cols)
    fig, ax = plt.subplots(figsize=(max(8, n), max(7, n - 1)))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, annot_kws={'size': 8})
    ax.set_title('Correlación de Spearman entre features (ventanas 60s)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / 'figures' / '06_correlacion_features.png', bbox_inches='tight')
    plt.close()


# ── PIPELINE (LOSO) ───────────────────────────────────────────────────────────
def plot_metrics_boxplots(fold_df: pd.DataFrame, results_dir: Path):
    """Boxplots de BA, F1-macro y ROC-AUC por modelo."""
    metrics = ['balanced_accuracy', 'f1_macro', 'roc_auc']
    n_folds = fold_df['fold'].nunique() if 'fold' in fold_df.columns else '?'
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 5))
    for ax, metric in zip(axes, metrics):
        data = [fold_df[fold_df['model'] == m][metric].values for m in MODEL_ORDER]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=MODEL_ORDER)
        for patch, color in zip(bp['boxes'], MODEL_COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim(0, 1); ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f"Distribución de métricas por fold (LOSO — {n_folds} folds)", y=1.01)
    plt.tight_layout()
    out = results_dir / "figures" / "metrics_boxplots.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out}")
    plt.close()


def plot_roc_curves(fold_df: pd.DataFrame, results_dir: Path):
    """Curvas ROC promediadas por fold. Folds con NaN en roc_auc son ignorados."""
    base_fpr = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, color in zip(MODEL_ORDER, MODEL_COLORS):
        grp = fold_df[fold_df['model'] == model_name]
        tprs, aucs = [], []
        for _, row in grp.iterrows():
            if np.isnan(row['roc_auc']):
                continue
            fpr, tpr, _ = roc_curve(row['y_true'], row['y_proba'])
            tprs.append(np.interp(base_fpr, fpr, tpr))
            aucs.append(row['roc_auc'])
        if not tprs:
            continue
        mean_auc, std_auc = np.mean(aucs), np.std(aucs)
        ax.plot(base_fpr, np.mean(tprs, axis=0),
                label=f"{model_name} (AUC={mean_auc:.3f}±{std_auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    n_folds = fold_df['fold'].nunique() if 'fold' in fold_df.columns else '?'
    ax.set_title(f"ROC Curves — promedio LOSO ({n_folds} folds)")
    ax.legend(loc='lower right')
    plt.tight_layout()
    out = results_dir / "figures" / "roc_curves.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out}")
    plt.close()


def plot_confusion_matrices(fold_df: pd.DataFrame, results_dir: Path):
    """Matrices de confusión acumuladas (suma de todos los folds) por modelo."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, model_name in zip(axes, MODEL_ORDER):
        grp = fold_df[fold_df['model'] == model_name]
        y_true_all = np.concatenate(grp['y_true'].values)
        y_pred_all = np.concatenate(grp['y_pred'].values)
        cm = confusion_matrix(y_true_all, y_pred_all)
        ConfusionMatrixDisplay(cm, display_labels=['Baseline', 'Stress']).plot(
            ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(model_name)
    fig.suptitle("Matrices de confusión (suma de todos los folds LOSO)")
    plt.tight_layout()
    out = results_dir / "figures" / "confusion_matrices.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out}")
    plt.close()


# ── ABLACIÓN ──────────────────────────────────────────────────────────────────
def plot_ablation_bars(summary_df: pd.DataFrame, results_dir: Path, metric: str = 'f1_macro'):
    """Barplot comparativo de experimentos de ablación por modelo."""
    model_order = ['RF', 'SVM', 'ANN', 'XGBoost']
    experiments = list(summary_df['experiment'].unique())
    colors = ['#4C72B0', '#DD8452', '#55A868'][:len(experiments)]
    x, width = range(len(model_order)), 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (exp, color) in enumerate(zip(experiments, colors)):
        grp = summary_df[summary_df['experiment'] == exp].set_index('model')
        means = [grp.loc[m, f'{metric}_mean'] if m in grp.index else 0 for m in model_order]
        stds  = [grp.loc[m, f'{metric}_std']  if m in grp.index else 0 for m in model_order]
        offset = (i - len(experiments) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], means, width,
               yerr=stds, label=exp, color=color, alpha=0.8, error_kw=dict(capsize=4))
    ax.set_xticks(list(x)); ax.set_xticklabels(model_order)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f"Ablation Study — {metric.replace('_', ' ').title()} (LOSO)")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out = results_dir / "ablation" / f"ablation_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out}")
    plt.close()
