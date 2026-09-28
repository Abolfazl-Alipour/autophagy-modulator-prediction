#!/usr/bin/env python3
"""
Journal-quality figures (Nature/Bioinformatics style) for the NAFS2 paper.

Reads cached data (paper/fig1_data.npz for fig1) and produces
paper/figN_*.pdf (vector) + paper/figN_*.png (300 dpi).

Run extract_fig1_data.py first to build the fig1 cache.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm

WORK_DIR = Path(__file__).resolve().parent
PAPER_DIR = WORK_DIR / 'paper'

# Okabe-Ito colorblind-safe palette
OI = {
    'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9',
    'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermillion': '#D55E00', 'purple': '#CC79A7', 'grey': '#999999',
}

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'font.size': 8,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def add_panel_letter(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top', ha='right', color='black')


def save(fig, name):
    pdf = PAPER_DIR / f'{name}.pdf'
    png = PAPER_DIR / f'{name}.png'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(png, dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'Saved {pdf} and {png}')


# ---------------------------------------------------------------- fig1
def make_fig1():
    d = np.load(PAPER_DIR / 'fig1_data.npz')
    x_m, y_m = d['pred_rapamycin'], d['pred_wortmannin']
    has_gt = 'gt_rapamycin' in d.files and np.isfinite(d['gt_rapamycin']).all()

    fig, axes = plt.subplots(1, 2 if has_gt else 1,
                             figsize=(7.0, 3.2) if has_gt else (3.5, 3.2))
    if not has_gt:
        axes = [axes]

    panels = [(axes[0], x_m, y_m, 'Model predictions (V6)',
               'Predicted logFC: Rapamycin (mTOR inhibitor)',
               'Predicted logFC: Wortmannin (PI3K inhibitor)')]
    if has_gt:
        panels.append((axes[1], d['gt_rapamycin'], d['gt_wortmannin'],
                       'Ground truth (L1000)',
                       'Observed logFC: Rapamycin (mTOR inhibitor)',
                       'Observed logFC: Wortmannin (PI3K inhibitor)'))

    for letter, (ax, x, y, title, xlab, ylab) in zip('ab', panels):
        ax.scatter(x, y, s=1, alpha=0.3, edgecolors='none',
                   color=OI['blue'], rasterized=True, zorder=2)
        # symmetric view limits (1st/99th percentile of |values|) so that
        # heavy-tailed expression outliers don't compress the cloud
        lim = np.percentile(np.abs(np.concatenate([x, y])), 99)
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_ylim(-lim * 1.05, lim * 1.05)
        ax.plot([-lim, lim], [-lim, lim], '--', color=OI['grey'], linewidth=0.8,
                zorder=3, label='y = x')
        rho, _ = spearmanr(x, y)
        ax.text(0.12, 0.98, f'Spearman ρ = {rho:+.3f}',
                transform=ax.transAxes, fontsize=8, va='top', ha='left')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc='lower right', fontsize=7)
        ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top', ha='left')

    fig.tight_layout(w_pad=2.2)
    save(fig, 'fig1_v6_correlation')
    return rho if has_gt else None


# ---------------------------------------------------------------- fig2
def make_fig2():
    curves, aurocs = {}, {}
    for path in ['hamdb_binary_roc_curves.npz', 'binary_roc_curves.npz',
                 'paper/reconstructed_xgb_roc.npz']:
        data = np.load(WORK_DIR / path, allow_pickle=True)
        for key in data.files:
            if key == 'aurocs':
                continue
            arr = data[key]
            curves[key] = (arr[0], arr[1])
        for row in data['aurocs']:
            aurocs[row[0]] = float(row[1])

    fig, ax = plt.subplots(figsize=(3.5, 3.3))

    styles = {
        'XGB Morgan only':        dict(color=OI['green'], linewidth=1.6, linestyle='-', zorder=5),
        'XGB Morgan + CLAMP':     dict(color=OI['blue'], linewidth=1.6, linestyle='-', zorder=5),
        'Morgan+context':         dict(color=OI['grey'], linewidth=0.8, linestyle='--', zorder=3),
        'CLAMP+context':          dict(color=OI['grey'], linewidth=0.8, linestyle=':', zorder=3),
        'Ridge+context (Morgan)': dict(color='#BBBBBB', linewidth=0.8, linestyle='-.', zorder=3),
        'Ridge+context (CLAMP)':  dict(color='#BBBBBB', linewidth=0.8, linestyle=(0, (3, 1, 1, 1)), zorder=3),
    }
    plot_order = ['Ridge+context (CLAMP)', 'Ridge+context (Morgan)',
                  'CLAMP+context', 'Morgan+context',
                  'XGB Morgan only', 'XGB Morgan + CLAMP']

    for name in plot_order:
        if name not in curves:
            continue
        fpr, tpr = curves[name]
        label = f'{name} (AUROC = {aurocs.get(name, np.nan):.3f})'
        ax.plot(fpr, tpr, label=label, **styles[name])

    ax.plot([0, 1], [0, 1], ':', color=OI['grey'], linewidth=0.8,
            label='Random (AUROC = 0.500)', zorder=2)

    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right', fontsize=7)
    save(fig, 'fig2_binary_classifier_roc')


# ---------------------------------------------------------------- fig3
def wilson_ci(pct, n, z=1.959963984540054):
    """Wilson score 95% CI for a percentage given count x out of n."""
    x = pct / 100.0 * n
    p = x / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def make_fig3():
    df = pd.read_csv(WORK_DIR / 'chembl_enrichment_morgan_vs_clamp.csv')
    metrics = [('autophagy_active_pct', 'ChEMBL autophagy-active annotations', 0.18),
               ('mtor_pi3k_pct', 'ChEMBL mTOR/PI3K target annotations', 0.24)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    colors = {'Morgan + CLAMP': OI['blue'], 'Morgan only': OI['orange']}

    for ax, (metric, title, baseline) in zip(axes, metrics):
        for model in ['Morgan only', 'Morgan + CLAMP']:
            sub = df[df['model'] == model].sort_values('threshold')
            pct = sub[metric].values
            n = sub['n_compounds'].values
            lo, hi = wilson_ci(pct, n)
            yerr = np.vstack([pct - lo, hi - pct])
            ax.errorbar(sub['threshold'], pct, yerr=yerr, marker='o',
                        markersize=3, linewidth=1.2, capsize=2, capthick=0.7,
                        elinewidth=0.7, color=colors[model], label=model,
                        zorder=3)
        ax.axhline(baseline, linestyle='--', color=OI['grey'], linewidth=0.7,
                   alpha=0.6, zorder=1)
        ax.text(0.955, baseline, f'baseline {baseline:.2f}', transform=ax.get_yaxis_transform(),
                fontsize=6.5, va='bottom', ha='right', color=OI['grey'])
        ax.set_xlabel('Modulator score threshold')
        ax.set_ylabel('Annotation rate (%)')
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        ax.legend(loc='upper left', fontsize=7)
    axes[0].text(-0.14, 1.06, 'a', transform=axes[0].transAxes, fontsize=10,
                 fontweight='bold', va='top', ha='right')
    axes[1].text(-0.10, 1.06, 'b', transform=axes[1].transAxes, fontsize=10,
                 fontweight='bold', va='top', ha='right')

    fig.tight_layout(w_pad=2.2)
    save(fig, 'fig3_chembl_enrichment')


# ---------------------------------------------------------------- fig4
def make_fig4():
    data = {'Morgan only': (0.900, 0.532),
            'Morgan + CLAMP': (0.900, 0.455)}

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    models = list(data.keys())
    x = np.arange(len(models))
    width = 0.35

    acc = [data[m][0] for m in models]
    f1 = [data[m][1] for m in models]
    b1 = ax.bar(x - width / 2, acc, width, label='Accuracy', color=OI['green'])
    b2 = ax.bar(x + width / 2, f1, width, label='Macro F1', color=OI['blue'])

    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7.5)
    save(fig, 'fig4_multiclass_comparison')


# ---------------------------------------------------------------- fig5
def make_fig5():
    df = pd.read_csv(WORK_DIR / 'paper_feature_importance.csv')
    df = df.sort_values('gain', ascending=False).head(15)
    df = df.iloc[::-1]  # largest gain on top for barh

    labels = []
    for _, row in df.iterrows():
        smarts = str(row['smarts'])
        if len(smarts) > 22:
            smarts = smarts[:21] + '…'
        labels.append(f"bit {int(row['bit'])} · {smarts}")

    colors = [OI['green'] if fe >= 1 else OI['vermillion']
              for fe in df['fold_enrichment']]

    fig, ax = plt.subplots(figsize=(3.5, 4.2))
    ax.barh(np.arange(len(df)), df['gain'], color=colors, height=0.7)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(0, color='black', linewidth=0.5, zorder=3)
    ax.set_xlabel('XGBoost gain')

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=OI['green'], label='Enriched in modulators (fold ≥ 1)'),
        Patch(facecolor=OI['vermillion'], label='Depleted in modulators (fold < 1)'),
    ]
    ax.legend(handles=legend_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.10), fontsize=7)
    save(fig, 'fig5_feature_importance')


if __name__ == '__main__':
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    if (PAPER_DIR / 'fig1_data.npz').exists():
        make_fig1()
    else:
        print('fig1_data.npz not found; run extract_fig1_data.py first, '
              'then re-run make_fig1 from this script.')
