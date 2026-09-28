#!/usr/bin/env python3
"""
Extract V6 model predictions and L1000 ground-truth profiles for
Rapamycin / Wortmannin (A549, 6 h) and cache them to paper/fig1_data.npz
so figures can be replotted without torch/cmapPy.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK_DIR))

from models_12k import NAFS2SingleHeadModel          # noqa: E402
from dataset_l1000_fixed import L1000DataManager     # noqa: E402
from batch_correction import load_baselines          # noqa: E402
from cmapPy.pandasGEXpress.parse import parse        # noqa: E402


def main():
    dm = L1000DataManager(
        gctx_path="data/l1000/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        sig_info_path="data/l1000/GSE92742_Broad_LINCS_sig_info.txt.gz",
        pert_info_path="data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz",
    )
    dm.load_metadata()

    cell_types = dm.meta['cell_id'].dropna().unique().tolist()
    cell_type_map = {ct: i for i, ct in enumerate(cell_types)}
    baselines = load_baselines('cell_type_baselines.npz')

    model = NAFS2SingleHeadModel(
        unimol_dim=512, n_genes=12328,
        n_cell_types=len(cell_types), mechanism_dim=128,
        gene_symbols=None,
    )
    model.load_state_dict(torch.load('nafs2_best_v6.pt', map_location='cpu',
                                     weights_only=True))
    model.eval()

    unimol_embs = torch.load('unimol_embeddings_l1000_10k.pt',
                             map_location='cpu', weights_only=True)

    controls = {
        'Rapamycin': {
            'smiles': r'COC1CC(CC(C)C2CC(=O)C(C)\C=C(C)\C(O)C(OC)C(=O)C(C)CC(C)\C=C\C=C\C=C(C)\C(CC3CCC(C)C(O)(O3)C(=O)C(=O)N3CCCCC3C(=O)O2)OC)CCC1O',
            'cell': 'A549',
        },
        'Wortmannin': {
            'smiles': r'COC[C@H]1OC(=O)c2coc3C(=O)C4=C([C@@H](C[C@@]5(C)C4CCC5=O)OC(=O)C)[C@]1(C)c23',
            'cell': 'A549',
        },
    }

    predictions = {}
    with torch.no_grad():
        for name, info in controls.items():
            emb = unimol_embs[info['smiles']]
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            cell_idx = cell_type_map[info['cell']]
            pred, _, _ = model(emb_tensor, torch.tensor([cell_idx]),
                               mask_cell_types=False)
            predictions[name] = pred.squeeze(0).numpy()

    dm.meta['pert_iname_lower'] = dm.meta['pert_iname'].str.lower()
    gt_profiles = {}
    for name, info in controls.items():
        # LINCS uses INN names: rapamycin is 'sirolimus'
        aliases = {'Rapamycin': ['rapamycin', 'sirolimus'],
                   'Wortmannin': ['wortmannin']}[name]
        mask = (dm.meta['pert_iname_lower'].isin(aliases)
                & (dm.meta['pert_time'] == 6)
                & (dm.meta['cell_id'] == info['cell']))
        matching = dm.meta[mask].sort_values(by='pert_dose', ascending=False)
        if len(matching) > 0:
            sig_id = matching.iloc[0].name
            print(f'{name}: GT signature {sig_id} '
                  f'(dose={matching.iloc[0]["pert_dose"]}, '
                  f'{len(matching)} signatures)')
            gctoo = parse(str(dm.gctx_path), cid=[sig_id])
            actual = gctoo.data_df.values.squeeze()
            gt_profiles[name] = actual - baselines[info['cell']]
        else:
            print(f'{name}: NO ground-truth signature found')
            gt_profiles[name] = None

    rho_m, _ = spearmanr(predictions['Rapamycin'], predictions['Wortmannin'])
    print(f'Model Spearman(Rapamycin, Wortmannin) = {rho_m:+.4f} '
          f'(n={len(predictions["Rapamycin"])})')

    out = {
        'pred_rapamycin': predictions['Rapamycin'],
        'pred_wortmannin': predictions['Wortmannin'],
    }
    if all(v is not None for v in gt_profiles.values()):
        gt_r, gt_w = gt_profiles['Rapamycin'], gt_profiles['Wortmannin']
        n = min(len(gt_r), len(gt_w))
        gt_r, gt_w = gt_r[:n], gt_w[:n]
        rho_gt, _ = spearmanr(gt_r, gt_w)
        print(f'GT Spearman(Rapamycin, Wortmannin) = {rho_gt:+.4f} (n={n})')
        out['gt_rapamycin'] = gt_r
        out['gt_wortmannin'] = gt_w

    np.savez(WORK_DIR / 'paper' / 'fig1_data.npz', **out)
    print('Saved paper/fig1_data.npz')


if __name__ == '__main__':
    main()
