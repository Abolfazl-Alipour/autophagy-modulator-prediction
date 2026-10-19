"""
Diagnostic 1: Linear Baseline Upper Bounds

Trains Ridge regression models on the same training data as the NN,
using scaffold-split validation, and reports upper bounds for prediction.
"""
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from cmapPy.pandasGEXpress.parse import parse

sys.path.append(str(Path(__file__).resolve().parent))
from dataset_l1000_fixed import L1000DataManager, get_murcko_scaffold
from batch_correction import load_baselines


def scaffold_split(scaffolds, train_frac=0.8, seed=42):
    unique = list(set(scaffolds))
    np.random.seed(seed)
    np.random.shuffle(unique)
    n_train = int(len(unique) * train_frac)
    train_scaffolds = set(unique[:n_train])
    val_scaffolds = set(unique[n_train:])
    train_idx = [i for i, s in enumerate(scaffolds) if s in train_scaffolds]
    val_idx = [i for i, s in enumerate(scaffolds) if s in val_scaffolds]
    return np.array(train_idx), np.array(val_idx)


def evaluate_ridge(X_train, X_val, Y_train, Y_val, name):
    print(f"\n--- {name} ---")
    model = Ridge(alpha=10.0)
    model.fit(X_train, Y_train)
    pred_val = model.predict(X_val)

    # R2 per gene
    ss_res = np.sum((Y_val - pred_val) ** 2, axis=0)
    ss_tot = np.sum((Y_val - Y_val.mean(axis=0)) ** 2, axis=0)
    r2_per_gene = 1 - ss_res / (ss_tot + 1e-12)
    print(f"  Mean R2 across genes: {np.mean(r2_per_gene):.4f}")
    print(f"  Median R2: {np.median(r2_per_gene):.4f}")
    print(f"  Genes with R2 > 0.1: {np.sum(r2_per_gene > 0.1)} / {len(r2_per_gene)}")
    print(f"  Genes with R2 > 0.5: {np.sum(r2_per_gene > 0.5)} / {len(r2_per_gene)}")

    # Spearman per gene on val set (sample if too large)
    n_val = len(Y_val)
    sample_size = min(2000, n_val)
    if n_val > sample_size:
        idx = np.random.choice(n_val, sample_size, replace=False)
    else:
        idx = np.arange(n_val)
    spearmans = []
    for g in range(Y_val.shape[1]):
        rho, _ = spearmanr(Y_val[idx, g], pred_val[idx, g])
        if not np.isnan(rho):
            spearmans.append(rho)
    spearmans = np.array(spearmans)
    print(f"  Mean Spearman (sampled val): {np.mean(spearmans):.4f}")
    print(f"  Median Spearman: {np.median(spearmans):.4f}")
    print(f"  Genes with |Spearman| > 0.2: {np.sum(np.abs(spearmans) > 0.2)} / {len(spearmans)}")

    return model, pred_val, r2_per_gene, spearmans


def main():
    print("=" * 80)
    print("DIAGNOSTIC 1: Linear Baseline Upper Bounds")
    print("=" * 80)

    print("\nLoading cached training data...")
    cache = np.load('diag_cache.npz', allow_pickle=True)
    X_unimol = cache['X_unimol']
    X_morgan = cache['X_morgan']
    Y = cache['Y']
    cell_ids = cache['cell_ids']
    scaffolds = cache['scaffolds']
    gene_symbols = cache['gene_symbols']
    cell_types = cache['cell_types']
    smiles_list = cache['smiles']
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

    n_compounds = len(set(smiles_list))
    print(f"Total signatures: {len(Y)}, unique compounds: {n_compounds}")

    # Scaffold split
    train_idx, val_idx = scaffold_split(scaffolds.tolist(), train_frac=0.8, seed=42)
    print(f"\nScaffold split: {len(train_idx)} train, {len(val_idx)} val")
    print(f"Train scaffolds: {len(set(scaffolds[train_idx]))}, Val scaffolds: {len(set(scaffolds[val_idx]))}")

    # Cell type one-hot
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cell_oh = encoder.fit_transform(cell_ids.reshape(-1, 1))
    print(f"Cell type one-hot dimension: {cell_oh.shape[1]}")

    # Feature sets
    feature_sets = {
        'cell_type_only': cell_oh,
        'morgan': X_morgan,
        'unimol': X_unimol,
        'morgan+cell_type': np.hstack([X_morgan, cell_oh]),
        'unimol+cell_type': np.hstack([X_unimol, cell_oh]),
    }

    results = {}
    for name, X in feature_sets.items():
        model, pred_val, r2, spear = evaluate_ridge(
            X[train_idx], X[val_idx], Y[train_idx], Y[val_idx], name
        )
        results[name] = {'model': model, 'r2': r2, 'spearman': spear}

    # Key gene summary
    print("\n" + "=" * 80)
    print("KEY GENE R2 COMPARISON")
    print("=" * 80)
    key_genes = ['DDIT4', 'FOXO3', 'CASP3', 'LAMP1', 'SQSTM1', 'CDKN2A', 'CDKN1A', 'CTSD', 'TFEB', 'XIST', 'COL11A1']
    print(f"{'Gene':<12}", end='')
    for name in feature_sets:
        print(f"{name:>20}", end='')
    print()
    for gene in key_genes:
        idx = gene_to_idx.get(gene)
        if idx is None:
            continue
        print(f"{gene:<12}", end='')
        for name in feature_sets:
            r2 = results[name]['r2'][idx]
            print(f"{r2:>20.4f}", end='')
        print()

    # Control compound evaluation
    print("\n" + "=" * 80)
    print("CONTROL COMPOUND EVALUATION (held-out scaffolds)")
    print("=" * 80)

    control_compounds = {
        'Rapamycin': "COC1CC(CC(C)C2CC(=O)C(C)\C=C(C)\C(O)C(OC)C(=O)C(C)CC(C)\C=C\C=C\C=C(C)\C(CC3CCC(C)C(O)(O3)C(=O)C(=O)N3CCCCC3C(=O)O2)OC)CCC1O",
        'Staurosporine': "CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n3c4ccccc4c5c6CNC(=O)c6c7c8ccccc8n2c7c35",
        'Chloroquine': "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12",
        'Wortmannin': "COC[C@H]1OC(=O)c2coc3C(=O)C4=C([C@@H](C[C@@]5(C)C4CCC5=O)OC(=O)C)[C@]1(C)c23",
        'Tamoxifen': "CCC(=C(c1ccccc1)c2ccc(OCCN(C)C)cc2)c3ccccc3",
        'Metformin': "CN(C)C(=N)NC(N)=N",
        'Random Control': "CC(C)CC(C)(C)C"
    }

    # Load control profiles from GCTX
    dm = L1000DataManager(
        gctx_path="data/l1000/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        sig_info_path="data/l1000/GSE92742_Broad_LINCS_sig_info.txt.gz",
        pert_info_path="data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz"
    )
    dm.load_metadata()
    baselines = load_baselines('cell_type_baselines.npz')
    dm.meta['pert_iname_lower'] = dm.meta['pert_iname'].str.lower()

    # Load unimol embeddings for controls
    unimol_embs = torch.load('unimol_embeddings_l1000_10k.pt', map_location='cpu', weights_only=True)

    control_profiles = {}
    for name, smiles in control_compounds.items():
        if name == 'Random Control':
            control_profiles[name] = np.zeros(len(gene_symbols))
            continue
        aliases = [name.lower()]
        if name == 'Rapamycin':
            aliases = ['rapamycin', 'sirolimus']
        mask = (dm.meta['pert_iname_lower'].isin(aliases)) & (dm.meta['pert_time'] == 6) & (dm.meta['cell_id'] == 'A549')
        matching = dm.meta[mask].sort_values('pert_dose', ascending=False)
        if len(matching) == 0:
            print(f"  {name}: no A549 6h signature found")
            continue
        sig_id = matching.iloc[0].name
        gctoo = parse(str(dm.gctx_path), cid=[sig_id])
        profile = gctoo.data_df.values.squeeze() - baselines['A549']
        control_profiles[name] = profile

    # Predict with Ridge-UniMol+cell and Ridge-Morgan+cell
    from rdkit import Chem
    from rdkit.Chem import AllChem
    for feat_name in ['unimol+cell_type', 'morgan+cell_type']:
        print(f"\n--- {feat_name} ---")
        model = results[feat_name]['model']
        for name, smiles in control_compounds.items():
            if name not in control_profiles:
                continue
            if smiles not in unimol_embs:
                print(f"  {name}: SMILES not in embeddings")
                continue
            emb = unimol_embs[smiles]
            emb = emb.numpy() if isinstance(emb, torch.Tensor) else emb
            # Morgan
            mol = Chem.MolFromSmiles(smiles)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            morgan_fp = arr.astype(np.float32)
            # Cell type: A549
            a549_idx = list(cell_types).index('A549')
            cell_vec = np.zeros(cell_oh.shape[1])
            cell_vec[a549_idx] = 1.0

            if feat_name == 'unimol+cell_type':
                x = np.hstack([emb, cell_vec])
            else:
                x = np.hstack([morgan_fp, cell_vec])
            pred = model.predict(x.reshape(1, -1)).squeeze()
            gt = control_profiles[name]
            rho, _ = spearmanr(pred, gt)
            print(f"  {name:<16} | Spearman(Pred, GT) = {rho:+.4f}")

    print("=" * 80)


if __name__ == '__main__':
    main()
