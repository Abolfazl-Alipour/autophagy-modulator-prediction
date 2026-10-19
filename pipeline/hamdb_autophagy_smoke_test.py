"""
Build HAMDB-only autophagy cache and run scaffold-split smoke test.
Uses only the 222 HAMDB compounds that exist in L1000.
"""
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('hamdb_autophagy_smoke')


def get_numeric_pcids(series):
    ids = set()
    for x in series.dropna():
        s = str(x).strip()
        try:
            ids.add(str(int(float(s))))
        except:
            pass
    return ids


def get_scaffold_split(scaffolds_arr, test_frac=0.2, random_state=42):
    unique_scaffolds = list(set(scaffolds_arr))
    np.random.seed(random_state)
    np.random.shuffle(unique_scaffolds)
    n_test = max(1, int(len(unique_scaffolds) * test_frac))
    test_scafs = set(unique_scaffolds[-n_test:])
    train_mask = np.array([s not in test_scafs for s in scaffolds_arr])
    test_mask = np.array([s in test_scafs for s in scaffolds_arr])
    return train_mask, test_mask


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, Y_train, X_test, Y_test, n_epochs=50, batch_size=256, lr=1e-3, device='cpu'):
    model = MLPRegressor(X_train.shape[1], Y_train.shape[1], [512, 256]).to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    n = X_train_t.shape[0]
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(X_train_t[idx])
            loss = criterion(pred, Y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_test_t), Y_test_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            logger.info(f"    MLP epoch {epoch}: val_loss={val_loss:.4f}")
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(X_test_t).cpu().numpy()


def evaluate_multioutput(Y_true, Y_pred):
    n_genes = Y_true.shape[1]
    per_gene_r2 = []
    per_gene_sp = []
    for j in range(n_genes):
        y_t = Y_true[:, j]
        y_p = Y_pred[:, j]
        mask = ~np.isnan(y_t) & ~np.isnan(y_p)
        if mask.sum() < 10:
            continue
        r2 = r2_score(y_t[mask], y_p[mask])
        sp, _ = spearmanr(y_t[mask], y_p[mask])
        per_gene_r2.append(r2)
        per_gene_sp.append(sp)
    per_gene_r2 = np.array(per_gene_r2)
    per_gene_sp = np.array(per_gene_sp)
    overall_r2 = r2_score(Y_true.flatten(), Y_pred.flatten())
    return {
        'overall_r2': overall_r2,
        'mean_gene_r2': per_gene_r2.mean(),
        'median_gene_r2': np.median(per_gene_r2),
        'mean_gene_spearman': per_gene_sp.mean(),
        'median_gene_spearman': np.median(per_gene_sp),
        'n_positive_r2': (per_gene_r2 > 0).sum(),
        'n_strong_r2': (per_gene_r2 > 0.1).sum(),
        'n_strong_sp': (np.abs(per_gene_sp) > 0.2).sum(),
        'n_genes': len(per_gene_r2),
    }


def main():
    logger.info("=== HAMDB-only autophagy smoke test ===")

    # Load autophagy gene list
    with open('autophagy_genes.txt', 'r') as f:
        auto_genes = [line.strip() for line in f if line.strip()]

    # Load HAMDB
    hamdb = pd.read_csv('data/hamdb_chemicals.csv', encoding='latin-1')
    hamdb = hamdb.dropna(subset=['Canonical_SMILES'])
    hamdb_pcids = get_numeric_pcids(hamdb['Pubchem_CID'])
    logger.info(f"HAMDB compounds with SMILES: {len(hamdb)}")
    logger.info(f"HAMDB compounds with numeric PubChem CID: {len(hamdb_pcids)}")

    # Load L1000 pert_info
    pert_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz', sep='\t',
                            usecols=['pert_id', 'canonical_smiles', 'pubchem_cid'])

    # Find HAMDB compounds in L1000
    pert_info['numeric_pcid'] = pert_info['pubchem_cid'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else None
    )
    hamdb_pert_info = pert_info[pert_info['numeric_pcid'].isin(hamdb_pcids)]
    hamdb_pert_info = hamdb_pert_info.drop_duplicates(subset=['canonical_smiles'])
    hamdb_smiles = set(hamdb_pert_info['canonical_smiles'].tolist())
    logger.info(f"HAMDB compounds found in L1000: {len(hamdb_smiles)}")

    # Load full cache
    logger.info("Loading full L1000 cache...")
    d = np.load('diag_cache_v9_morgan_full.npz', allow_pickle=True)
    full_smiles = d['smiles']
    full_Y = d['Y']
    full_scaffolds = d['scaffolds']
    full_cell_ids = d['cell_ids']
    full_doses = d['doses']
    full_times = d['times']
    gene_symbols = list(d['gene_symbols'])
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    auto_indices = [gene_to_idx[g] for g in auto_genes if g in gene_to_idx]

    # Filter to HAMDB compounds
    mask = np.array([sm in hamdb_smiles for sm in full_smiles])
    n_hamdb_sigs = mask.sum()
    logger.info(f"HAMDB signatures in full cache: {n_hamdb_sigs}")

    if n_hamdb_sigs == 0:
        logger.error("No HAMDB signatures found!")
        return

    smiles = full_smiles[mask]
    scaffolds = full_scaffolds[mask]
    Y_auto = full_Y[np.ix_(mask, auto_indices)].astype(np.float32)
    Y_auto = np.nan_to_num(Y_auto, nan=0.0)
    cell_ids = full_cell_ids[mask]
    doses = full_doses[mask]
    times = full_times[mask]

    logger.info(f"  Y_auto shape: {Y_auto.shape}")
    logger.info(f"  Unique compounds: {len(set(smiles))}")
    logger.info(f"  Unique scaffolds: {len(set(scaffolds))}")
    logger.info(f"  Time dist: {sorted(Counter(times.tolist()).items())}")

    # Load CLAMP embeddings
    logger.info("Loading CLAMP embeddings...")
    clamp_embs = torch.load('clamp_embeddings_l1000_full.pt', map_location='cpu', weights_only=True)
    X_clamp = np.array([clamp_embs[sm].numpy() for sm in smiles], dtype=np.float32)
    logger.info(f"  X_clamp shape: {X_clamp.shape}")

    # Save cache
    np.savez_compressed('diag_cache_hamdb_autophagy.npz',
        X_clamp=X_clamp, Y_auto=Y_auto, smiles=smiles, scaffolds=scaffolds,
        cell_ids=cell_ids, doses=doses, times=times,
        gene_symbols=np.array(auto_genes, dtype=object)
    )
    logger.info("Saved diag_cache_hamdb_autophagy.npz")

    # Scaffold split
    train_mask, test_mask = get_scaffold_split(scaffolds)
    logger.info(f"Scaffold split: train={train_mask.sum()}, test={test_mask.sum()}")

    X_train, X_test = X_clamp[train_mask], X_clamp[test_mask]
    Y_train, Y_test = Y_auto[train_mask], Y_auto[test_mask]

    results = []

    # Ridge
    logger.info("Training Ridge...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train)
    Y_pred = ridge.predict(X_test)
    res = evaluate_multioutput(Y_test, Y_pred)
    res['model'] = 'Ridge'
    results.append(res)
    logger.info(f"  Ridge: {res}")

    # XGB
    logger.info("Training XGBoost...")
    try:
        import xgboost as xgb
        xgb_base = xgb.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            tree_method='hist', objective='reg:squarederror',
            n_jobs=1, random_state=42, verbosity=0
        )
        xgb_multi = MultiOutputRegressor(xgb_base, n_jobs=12)
        xgb_multi.fit(X_train, Y_train)
        Y_pred = xgb_multi.predict(X_test)
        res = evaluate_multioutput(Y_test, Y_pred)
        res['model'] = 'XGB'
        results.append(res)
        logger.info(f"  XGB: {res}")
    except Exception as e:
        logger.error(f"  XGB failed: {e}")

    # MLP
    logger.info("Training MLP...")
    Y_pred = train_mlp(X_train, Y_train, X_test, Y_test, n_epochs=50)
    res = evaluate_multioutput(Y_test, Y_pred)
    res['model'] = 'MLP'
    results.append(res)
    logger.info(f"  MLP: {res}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/hamdb_autophagy_smoke_results.csv', index=False)
    logger.info(f"\nSaved hamdb_autophagy_smoke_results.csv")
    logger.info("\nSummary:")
    logger.info(df.to_string(index=False))


if __name__ == '__main__':
    main()