from pathlib import Path
import os
import sys
import numpy as np
import torch
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(str(Path(__file__).resolve().parent))
from models_12k import NAFS2SingleHeadModel
from dataset_l1000_fixed import L1000DataManager, L1000ChunkedDataset
from batch_correction import load_baselines

def main():
    checkpoint_path = "nafs2_best_v6.pt"
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        
    print("="*80)
    print("NAFS2 V6 POST-TRAINING BIOLOGICAL AUDIT & COLLAPSE SCORECARD")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*80)
    
    # 1. Load DataManager
    print("\nLoading LINCS metadata...")
    dm = L1000DataManager(
        gctx_path="data/l1000/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        sig_info_path="data/l1000/GSE92742_Broad_LINCS_sig_info.txt.gz",
        pert_info_path="data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz"
    )
    dm.load_metadata()
    
    # Get symbols mapping
    from cmapPy.pandasGEXpress.parse import parse
    row_df = parse(str(dm.gctx_path), row_meta_only=True)
    gene_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_gene_info.txt.gz', sep='\t')
    gene_map = dict(zip(gene_info['pr_gene_id'].astype(str), gene_info['pr_gene_symbol']))
    gene_symbols = [gene_map.get(str(i), str(i)) for i in row_df.index]
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    
    cell_types = dm.meta['cell_id'].dropna().unique().tolist()
    cell_type_map = {ct: i for i, ct in enumerate(cell_types)}
    baselines = load_baselines('cell_type_baselines.npz')
    
    # 2. Load Model
    print("\nLoading model...")
    model = NAFS2SingleHeadModel(
        unimol_dim=512,
        n_genes=len(gene_symbols),
        n_cell_types=len(cell_types),
        mechanism_dim=128,
        gene_symbols=gene_symbols
    )
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint {checkpoint_path} not found!")
        sys.exit(1)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.eval()
    
    # 3. Load Uni-Mol embeddings
    unimol_embs = torch.load("unimol_embeddings_l1000_10k.pt", map_location='cpu', weights_only=True)
    
    control_compounds = {
        'Rapamycin': {
            'aliases': ['rapamycin', 'sirolimus'],
            'smiles': "COC1CC(CC(C)C2CC(=O)C(C)\C=C(C)\C(O)C(OC)C(=O)C(C)CC(C)\C=C\C=C\C=C(C)\C(CC3CCC(C)C(O)(O3)C(=O)C(=O)N3CCCCC3C(=O)O2)OC)CCC1O",
            'eval_cell': 'A549'
        },
        'Staurosporine': {
            'aliases': ['staurosporine'],
            'smiles': "CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n3c4ccccc4c5c6CNC(=O)c6c7c8ccccc8n2c7c35",
            'eval_cell': 'A549'
        },
        'Chloroquine': {
            'aliases': ['chloroquine'],
            'smiles': "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
            'eval_cell': 'A549'
        },
        'Wortmannin': {
            'aliases': ['wortmannin'],
            'smiles': "COC[C@H]1OC(=O)c2coc3C(=O)C4=C([C@@H](C[C@@]5(C)C4CCC5=O)OC(=O)C)[C@]1(C)c23",
            'eval_cell': 'A549'
        },
        'Tamoxifen': {
            'aliases': ['tamoxifen'],
            'smiles': "CCC(=C(c1ccccc1)c2ccc(OCCN(C)C)cc2)c3ccccc3",
            'eval_cell': 'PC3'
        },
        'Metformin': {
            'aliases': ['metformin'],
            'smiles': "CN(C)C(=N)NC(N)=N",
            'eval_cell': 'A375'
        },
        'Random Control': {
            'aliases': [],
            'smiles': "CC(C)CC(C)(C)C",
            'eval_cell': 'A549'
        }
    }
    
    # 4. Extract ground-truth profiles and run inference
    predictions = {}
    ground_truths = {}
    
    print("\n--- Comparing Predictions against 6h LINCS Ground-Truth ---")
    dm.meta['pert_iname_lower'] = dm.meta['pert_iname'].str.lower()
    
    for name, info in control_compounds.items():
        smiles = info['smiles']
        cell = info['eval_cell']
        
        emb = unimol_embs[smiles]
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        cell_idx = cell_type_map[cell]
        
        with torch.no_grad():
            pred, _, _ = model(emb_tensor, torch.tensor([cell_idx]), mask_cell_types=False)
            pred_logfc = pred.squeeze(0).numpy()
            
        predictions[name] = pred_logfc
        
        if name == 'Random Control':
            ground_truths[name] = np.zeros_like(pred_logfc)
            continue
            
        aliases = info['aliases']
        mask = (dm.meta['pert_iname_lower'].isin(aliases)) & (dm.meta['pert_time'] == 6) & (dm.meta['cell_id'] == cell)
        matching = dm.meta[mask]
        
        matching = matching.sort_values(by='pert_dose', ascending=False)
        best_sig = matching.iloc[0]
        sig_id = best_sig.name
        
        gctoo = parse(str(dm.gctx_path), cid=[sig_id])
        actual_profile = gctoo.data_df.values.squeeze()
        actual_logfc = actual_profile - baselines[cell]
        ground_truths[name] = actual_logfc
        
        rho, _ = spearmanr(pred_logfc, actual_logfc)
        print(f"  {name:<13} | Cell: {cell:<6} | Spearman(Pred, GT) = {rho:+.4f}")
        
    # 5. Extract A549 profiles to build mean perturbed baseline
    print("\nLoading dataset chunks to compute mean A549 perturbed baseline...")
    dataset = L1000ChunkedDataset(
        dm, unimol_embs, cell_type_map,
        chunk_size=10000, cache_dir="chunk_cache_12k_v3",
        subset_limit=20000,
        baselines=baselines
    )
    
    a549_profiles = []
    for idx, item in enumerate(dataset.valid_indices):
        sig_id = item[2]
        info = dm.meta.loc[sig_id] if sig_id in dm.meta.index else {}
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        if info.get('cell_id') == 'A549':
            prof = dataset.profiles[idx].copy() - baselines['A549']
            a549_profiles.append(prof)
            
    mean_a549_perturbed = np.mean(a549_profiles, axis=0)
    print(f"Computed mean A549 perturbed profile from {len(a549_profiles)} signatures.")
    
    # 6. Evaluate Collapse scorecard
    print("\n" + "="*80)
    print("COLLAPSE SCORECARD EVALUATION")
    print("="*80)
    
    checks_passed = 0
    checks_total = 0
    
    # Check 1: Rapa vs Wortmannin predicted correlation < 0.50
    checks_total += 1
    rho_rapa_wort, _ = spearmanr(predictions['Rapamycin'], predictions['Wortmannin'])
    if rho_rapa_wort < 0.50:
        print(f"  ✅ [PASS] Spearman(Rapamycin, Wortmannin) = {rho_rapa_wort:+.4f} < 0.50 (mTOR vs PI3K resolved)")
        checks_passed += 1
    else:
        print(f"  ❌ [FAIL] Spearman(Rapamycin, Wortmannin) = {rho_rapa_wort:+.4f} >= 0.50 (collapsed active predictions)")
        
    # Check 2: Rapa vs Mean A549 profile < 0.50
    checks_total += 1
    rho_rapa_mean, _ = spearmanr(predictions['Rapamycin'], mean_a549_perturbed)
    if rho_rapa_mean < 0.50:
        print(f"  ✅ [PASS] Spearman(Rapamycin, Mean A549) = {rho_rapa_mean:+.4f} < 0.50 (differentiated from cell mean)")
        checks_passed += 1
    else:
        print(f"  ❌ [FAIL] Spearman(Rapamycin, Mean A549) = {rho_rapa_mean:+.4f} >= 0.50 (collapsed to cell mean template)")
        
    # Check 3: Predicted DDIT4 value within \pm 1.0 of actual (-0.761)
    checks_total += 1
    pred_ddit4 = predictions['Rapamycin'][gene_to_idx['DDIT4']]
    gt_ddit4 = ground_truths['Rapamycin'][gene_to_idx['DDIT4']] # is -0.761
    if abs(pred_ddit4 - gt_ddit4) <= 1.0:
        print(f"  ✅ [PASS] Rapamycin DDIT4 predicted: {pred_ddit4:+.3f} (GT: {gt_ddit4:+.3f}, diff={abs(pred_ddit4 - gt_ddit4):.3f} <= 1.0)")
        checks_passed += 1
    else:
        print(f"  ❌ [FAIL] Rapamycin DDIT4 predicted: {pred_ddit4:+.3f} (GT: {gt_ddit4:+.3f}, diff={abs(pred_ddit4 - gt_ddit4):.3f} > 1.0)")
        
    # Check 4: Predicted FOXO3 value within \pm 0.5 of actual (+1.502)
    checks_total += 1
    pred_foxo3 = predictions['Rapamycin'][gene_to_idx['FOXO3']]
    gt_foxo3 = ground_truths['Rapamycin'][gene_to_idx['FOXO3']] # is +1.502
    if abs(pred_foxo3 - gt_foxo3) <= 0.5:
        print(f"  ✅ [PASS] Rapamycin FOXO3 predicted: {pred_foxo3:+.3f} (GT: {gt_foxo3:+.3f}, diff={abs(pred_foxo3 - gt_foxo3):.3f} <= 0.5)")
        checks_passed += 1
    else:
        print(f"  ❌ [FAIL] Rapamycin FOXO3 predicted: {pred_foxo3:+.3f} (GT: {gt_foxo3:+.3f}, diff={abs(pred_foxo3 - gt_foxo3):.3f} > 0.5)")
        
    # Check 5: Random Control is inactive (mean |val| < 0.10)
    checks_total += 1
    rand_mean_abs = np.mean(np.abs(predictions['Random Control']))
    if rand_mean_abs < 0.10:
        print(f"  ✅ [PASS] Random Control is inactive (mean |val| = {rand_mean_abs:.4f} < 0.10)")
        checks_passed += 1
    else:
        print(f"  ❌ [FAIL] Random Control shows high activity (mean |val| = {rand_mean_abs:.4f} >= 0.10)")
        
    print(f"\nAudit Result: {checks_passed}/{checks_total} checks passed.")
    print("="*80)

if __name__ == "__main__":
    main()
