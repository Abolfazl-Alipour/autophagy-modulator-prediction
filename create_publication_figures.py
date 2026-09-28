#!/usr/bin/env python3
"""
Generate publication-quality figures for the NAFS2 autophagy modulator project.

Figures produced:
  - fig1_v6_correlation.png          : V6 model collapse scatter
  - fig2_binary_classifier_roc.png   : ROC curves for binary classifiers
  - fig3_chembl_enrichment.png       : ChEMBL enrichment Morgan vs CLAMP
  - fig4_multiclass_comparison.png   : 3-class model performance
  - linkedin_infographic.png         : Two-panel summary infographic
"""
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from cmapPy.pandasGEXpress.parse import parse
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings('ignore')

WORK_DIR = Path(__file__).resolve().parent
ROC_CURVES_PATH = WORK_DIR / 'hamdb_binary_roc_curves.npz'
STRUCTURE_ONLY_ROC_PATH = WORK_DIR / 'binary_roc_curves.npz'
ENRICHMENT_PATH = WORK_DIR / 'chembl_enrichment_morgan_vs_clamp.csv'
MULTICLASS_PATH = WORK_DIR / 'multiclass_metrics.csv'
OUTPUT_DIR = WORK_DIR

RANDOM_STATE = 42
N_JOBS = 12


def setup_style():
    sns.set_context('poster', font_scale=0.65)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.grid'] = False


def extract_v6_predictions():
    """Return dict of V6 predictions for Rapamycin and Wortmannin."""
    sys.path.append(str(WORK_DIR))
    from models_12k import NAFS2SingleHeadModel
    from dataset_l1000_fixed import L1000DataManager
    from batch_correction import load_baselines

    dm = L1000DataManager(
        gctx_path="data/l1000/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        sig_info_path="data/l1000/GSE92742_Broad_LINCS_sig_info.txt.gz",
        pert_info_path="data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz"
    )
    dm.load_metadata()

    cell_types = dm.meta['cell_id'].dropna().unique().tolist()
    cell_type_map = {ct: i for i, ct in enumerate(cell_types)}
    baselines = load_baselines('cell_type_baselines.npz')

    model = NAFS2SingleHeadModel(
        unimol_dim=512,
        n_genes=12328,
        n_cell_types=len(cell_types),
        mechanism_dim=128,
        gene_symbols=None
    )
    model.load_state_dict(torch.load('nafs2_best_v6.pt', map_location='cpu', weights_only=True))
    model.eval()

    unimol_embs = torch.load('unimol_embeddings_l1000_10k.pt', map_location='cpu', weights_only=True)

    controls = {
        'Rapamycin': {
            'smiles': r'COC1CC(CC(C)C2CC(=O)C(C)\C=C(C)\C(O)C(OC)C(=O)C(C)CC(C)\C=C\C=C\C=C(C)\C(CC3CCC(C)C(O)(O3)C(=O)C(=O)N3CCCCC3C(=O)O2)OC)CCC1O',
            'cell': 'A549'
        },
        'Wortmannin': {
            'smiles': r'COC[C@H]1OC(=O)c2coc3C(=O)C4=C([C@@H](C[C@@]5(C)C4CCC5=O)OC(=O)C)[C@]1(C)c23',
            'cell': 'A549'
        },
    }

    predictions = {}
    with torch.no_grad():
        for name, info in controls.items():
            emb = unimol_embs[info['smiles']]
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            cell_idx = cell_type_map[info['cell']]
            pred, _, _ = model(emb_tensor, torch.tensor([cell_idx]), mask_cell_types=False)
            predictions[name] = pred.squeeze(0).numpy()

    dm.meta['pert_iname_lower'] = dm.meta['pert_iname'].str.lower()
    gt_profiles = {}
    for name, info in controls.items():
        aliases = [name.lower()]
        mask = (dm.meta['pert_iname_lower'].isin(aliases)) & (dm.meta['pert_time'] == 6) & (dm.meta['cell_id'] == info['cell'])
        matching = dm.meta[mask].sort_values(by='pert_dose', ascending=False)
        if len(matching) > 0:
            sig_id = matching.iloc[0].name
            gctoo = parse(str(dm.gctx_path), cid=[sig_id])
            actual = gctoo.data_df.values.squeeze()
            gt_profiles[name] = actual - baselines[info['cell']]
        else:
            gt_profiles[name] = None

    rho, _ = spearmanr(predictions['Rapamycin'], predictions['Wortmannin'])
    print(f'V6 Spearman(Rapamycin, Wortmannin) = {rho:+.4f}')
    return predictions, gt_profiles


def create_v6_figure(v6_pred, output_path):
    setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)

    x = v6_pred['Rapamycin']
    y = v6_pred['Wortmannin']
    n_genes = len(x)

    color_fail = '#c0392b'
    color_ref = '#2c3e50'

    ax.scatter(x, y, s=1, c=color_fail, alpha=0.18, edgecolors='none', rasterized=True)

    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color=color_ref,
            linewidth=1.5, label='y = x (collapse)')

    ax.set_xlabel('Predicted logFC: Rapamycin\n(mTOR inhibitor)', fontsize=12)
    ax.set_ylabel('Predicted logFC: Wortmannin\n(PI3K inhibitor)', fontsize=12)

    rho, _ = spearmanr(x, y)
    ax.text(0.05, 0.97, f'Spearman ρ = {rho:+.2f}',
            transform=ax.transAxes, fontsize=13, fontweight='bold', color=color_fail,
            verticalalignment='top')
    ax.text(0.05, 0.84, f'Each point = one gene\n(n = {n_genes:,})',
            transform=ax.transAxes, fontsize=10, color='#333333',
            verticalalignment='top')

    ax.tick_params(labelsize=10)
    sns.despine(ax=ax)
    ax.legend(loc='lower right', fontsize=10, frameon=True, edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.08)
    plt.close()
    print(f'Saved {output_path}')


def load_roc_curves():
    curves = {}
    aurocs = {}
    for path in [ROC_CURVES_PATH, STRUCTURE_ONLY_ROC_PATH]:
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            if key == 'aurocs':
                continue
            arr = data[key]
            fpr, tpr, _ = arr[0], arr[1], arr[2]
            curves[key] = (fpr, tpr)
        auroc_arr = data['aurocs']
        for row in auroc_arr:
            name, score = row
            aurocs[name] = float(score)
    return curves, aurocs


def create_binary_roc_figure(output_path):
    setup_style()
    fig, ax = plt.subplots(figsize=(6.0, 5.5), dpi=150)

    roc_curves, aurocs = load_roc_curves()

    # Emphasized: structure-only models (no L1000 context)
    color_clamp = '#2980b9'
    color_win = '#27ae60'
    # Muted: L1000-context models
    color_context = '#95a5a6'
    color_ridge = '#bdc3c7'
    color_ref = '#2c3e50'

    curve_styles = {
        # Structure-only (pronounced)
        'XGB Morgan + CLAMP': {'color': color_clamp, 'linewidth': 3.0, 'linestyle': '-', 'zorder': 5},
        'XGB Morgan only': {'color': color_win, 'linewidth': 3.0, 'linestyle': '-', 'zorder': 5},
        # L1000-context (muted)
        'Morgan+context': {'color': color_context, 'linewidth': 1.5, 'linestyle': '--', 'zorder': 3},
        'CLAMP+context': {'color': color_context, 'linewidth': 1.5, 'linestyle': ':', 'zorder': 3},
        'Ridge+context (Morgan)': {'color': color_ridge, 'linewidth': 1.5, 'linestyle': '-.', 'zorder': 3},
        'Ridge+context (CLAMP)': {'color': color_ridge, 'linewidth': 1.5, 'linestyle': (0, (3, 1, 1, 1)), 'zorder': 3},
    }

    # Order: muted context models first, then emphasized structure-only models on top
    plot_order = [
        'Ridge+context (CLAMP)',
        'Ridge+context (Morgan)',
        'CLAMP+context',
        'Morgan+context',
        'XGB Morgan only',
        'XGB Morgan + CLAMP',
    ]

    for name in plot_order:
        if name not in roc_curves:
            continue
        fpr, tpr = roc_curves[name]
        auroc = aurocs.get(name, np.nan)
        style = curve_styles[name]
        label = f'{name} (AUROC = {auroc:.3f})'
        ax.plot(fpr, tpr, label=label, **style)

    ax.plot([0, 1], [0, 1], ':', color=color_ref, linewidth=1.5,
            label='Random classifier (AUROC = 0.500)', zorder=2)

    ax.set_xlabel('False positive rate', fontsize=12)
    ax.set_ylabel('True positive rate', fontsize=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    ax.tick_params(labelsize=10)
    sns.despine(ax=ax)
    ax.legend(loc='lower right', fontsize=9, frameon=True, edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.08)
    plt.close()
    print(f'Saved {output_path}')


def create_enrichment_figure(output_path):
    setup_style()
    df = pd.read_csv(ENRICHMENT_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), dpi=150)

    color_clamp = '#2980b9'
    color_win = '#27ae60'

    for ax, metric, title in zip(
        axes,
        ['autophagy_active_pct', 'mtor_pi3k_pct'],
        ['ChEMBL autophagy-active annotations', 'ChEMBL mTOR/PI3K target annotations']
    ):
        for model, color in [('Morgan + CLAMP', color_clamp), ('Morgan only', color_win)]:
            sub = df[df['model'] == model]
            ax.plot(sub['threshold'], sub[metric], marker='o', markersize=6,
                    linewidth=2.5, label=model, color=color)

        ax.set_xlabel('Modulator score threshold', fontsize=12)
        ax.set_ylabel(f'Annotation rate (%)', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        ax.tick_params(labelsize=10)
        sns.despine(ax=ax)
        ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='#cccccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.08)
    plt.close()
    print(f'Saved {output_path}')


def create_multiclass_figure(output_path):
    setup_style()

    data = {
        'Morgan only': {'Accuracy': 0.900, 'Macro F1': 0.532},
        'Morgan + CLAMP': {'Accuracy': 0.900, 'Macro F1': 0.455},
    }

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=150)

    models = list(data.keys())
    x = np.arange(len(models))
    width = 0.35

    color_win = '#27ae60'
    color_clamp = '#2980b9'

    acc_vals = [data[m]['Accuracy'] for m in models]
    f1_vals = [data[m]['Macro F1'] for m in models]

    bars1 = ax.bar(x - width/2, acc_vals, width, label='Accuracy', color=color_win, edgecolor='#333333')
    bars2 = ax.bar(x + width/2, f1_vals, width, label='Macro F1', color=color_clamp, edgecolor='#333333')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('3-class autophagy classifier performance\n(activator / inhibitor / neutral)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=10)
    ax.legend(loc='upper right', fontsize=10, frameon=True, edgecolor='#cccccc')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.08)
    plt.close()
    print(f'Saved {output_path}')


def create_linkedin_infographic(v6_pred, output_path):
    setup_style()

    fig = plt.figure(figsize=(8.0, 4.18), dpi=150)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.40,
                  left=0.10, right=0.92, top=0.82, bottom=0.30)

    color_fail = '#c0392b'
    # Emphasized structure-only models
    color_win = '#27ae60'
    color_clamp = '#2980b9'
    # Muted L1000-context models
    color_context = '#95a5a6'
    color_ridge = '#bdc3c7'
    color_ref = '#2c3e50'

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    x = v6_pred['Rapamycin']
    y = v6_pred['Wortmannin']
    n_genes = len(x)

    ax1.scatter(x, y, s=1, c=color_fail, alpha=0.18, edgecolors='none', rasterized=True)

    lim_min = min(x.min(), y.min())
    lim_max = max(x.max(), y.max())
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], '--', color=color_ref,
             linewidth=1.5, label='y = x (collapse)')

    ax1.set_xlabel('Predicted logFC: Rapamycin\n(mTOR inhibitor)', fontsize=11)
    ax1.set_ylabel('Predicted logFC: Wortmannin\n(PI3K inhibitor)', fontsize=11)

    rho, _ = spearmanr(x, y)
    ax1.text(0.05, 0.97, f'Spearman ρ = {rho:+.2f}',
             transform=ax1.transAxes, fontsize=12, fontweight='bold', color=color_fail,
             verticalalignment='top')
    ax1.text(0.05, 0.84, f'Each point = one gene\n(n = {n_genes:,})',
             transform=ax1.transAxes, fontsize=9, color='#333333',
             verticalalignment='top')

    ax1.tick_params(labelsize=9)
    sns.despine(ax=ax1)
    ax1.legend(loc='lower right', fontsize=9, frameon=True, edgecolor='#cccccc')
    ax1.text(-0.18, 1.05, 'a', transform=ax1.transAxes, fontsize=18,
             fontweight='bold', va='top', ha='right', color='#333333')

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    roc_curves, aurocs = load_roc_curves()

    curve_styles = {
        # Structure-only (pronounced)
        'XGB Morgan + CLAMP': {'color': color_clamp, 'linewidth': 2.5, 'linestyle': '-', 'zorder': 5},
        'XGB Morgan only': {'color': color_win, 'linewidth': 2.5, 'linestyle': '-', 'zorder': 5},
        # L1000-context (muted)
        'Morgan+context': {'color': color_context, 'linewidth': 1.5, 'linestyle': '--', 'zorder': 3},
        'CLAMP+context': {'color': color_context, 'linewidth': 1.5, 'linestyle': ':', 'zorder': 3},
        'Ridge+context (Morgan)': {'color': color_ridge, 'linewidth': 1.5, 'linestyle': '-.', 'zorder': 3},
        'Ridge+context (CLAMP)': {'color': color_ridge, 'linewidth': 1.5, 'linestyle': (0, (3, 1, 1, 1)), 'zorder': 3},
    }

    plot_order = [
        'Ridge+context (CLAMP)',
        'Ridge+context (Morgan)',
        'CLAMP+context',
        'Morgan+context',
        'XGB Morgan only',
        'XGB Morgan + CLAMP',
    ]
    for name in plot_order:
        if name not in roc_curves:
            continue
        fpr, tpr = roc_curves[name]
        auroc = aurocs.get(name, np.nan)
        style = curve_styles[name]
        label = f'{name} (AUROC = {auroc:.3f})'
        ax2.plot(fpr, tpr, label=label, **style)

    ax2.plot([0, 1], [0, 1], ':', color=color_ref, linewidth=1.5,
             label='Random classifier (AUROC = 0.500)', zorder=2)

    ax2.set_xlabel('False positive rate', fontsize=11)
    ax2.set_ylabel('True positive rate', fontsize=11)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    ax2.tick_params(labelsize=9)
    sns.despine(ax=ax2)
    ax2.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=7,
        frameon=True,
        edgecolor='#cccccc',
        columnspacing=0.8,
        handletextpad=0.4
    )
    ax2.text(-0.18, 1.05, 'b', transform=ax2.transAxes, fontsize=18,
             fontweight='bold', va='top', ha='right', color='#333333')

    fig.suptitle('Black-box transcriptome AI collapses; Morgan fingerprints rescue autophagy prediction',
                 fontsize=14, fontweight='bold', y=0.96, color='#333333')
    fig.text(0.5, 0.02,
             'Data: LINCS L1000 and HAMDB.  V6 = Uni-Mol neural network.  ROC curves = binary modulator classifiers on scaffold-held-out test set; Ridge is a linear baseline.',
             ha='center', fontsize=8, color='#555555')

    plt.savefig(output_path, dpi=150, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.08)
    plt.close()
    print(f'Saved {output_path}')


def main():
    print('Extracting V6 predictions...')
    v6_pred, _ = extract_v6_predictions()

    print('\nGenerating figures...')
    create_v6_figure(v6_pred, OUTPUT_DIR / 'fig1_v6_correlation.png')
    create_binary_roc_figure(OUTPUT_DIR / 'fig2_binary_classifier_roc.png')
    create_enrichment_figure(OUTPUT_DIR / 'fig3_chembl_enrichment.png')
    create_multiclass_figure(OUTPUT_DIR / 'fig4_multiclass_comparison.png')
    create_linkedin_infographic(v6_pred, OUTPUT_DIR / 'linkedin_infographic.png')

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
