"""
Diagnostic 2: Linear Probe on Key Genes

For each key gene, train a Ridge regression to predict its logFC from
various feature sets. Compare chemical signal vs. cell-type/dose signal.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed


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


def evaluate_probe(X_train, X_val, y_train, y_val, alpha=10.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)

    ss_res = np.sum((y_val - pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)

    rho, _ = spearmanr(y_val, pred)
    if np.isnan(rho):
        rho = 0.0
    return r2, rho


def _fit_one_gene(g, X_train, X_val, Y_train, Y_val, alpha=10.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, Y_train[:, g])
    pred = model.predict(X_val)
    ss_res = np.sum((Y_val[:, g] - pred) ** 2)
    ss_tot = np.sum((Y_val[:, g] - np.mean(Y_val[:, g])) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    rho, _ = spearmanr(Y_val[:, g], pred)
    if np.isnan(rho):
        rho = 0.0
    return g, r2, rho


def main():
    print("=" * 80)
    print("DIAGNOSTIC 2: Linear Probe on Key Genes")
    print("=" * 80)

    print("\nLoading cached training data...")
    cache = np.load('diag_cache.npz', allow_pickle=True)
    X_unimol = cache['X_unimol']
    X_morgan = cache['X_morgan']
    Y = cache['Y']
    cell_ids = cache['cell_ids']
    doses = cache['doses']
    scaffolds = cache['scaffolds']
    gene_symbols = cache['gene_symbols']
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}

    # Feature sets
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cell_oh = encoder.fit_transform(cell_ids.reshape(-1, 1))

    dose_norm = (doses - np.mean(doses)) / (np.std(doses) + 1e-12)
    dose_feat = dose_norm.reshape(-1, 1)

    feature_sets = {
        'cell_type_only': cell_oh,
        'dose_only': dose_feat,
        'cell_type+dose': np.hstack([cell_oh, dose_feat]),
        'morgan': X_morgan,
        'unimol': X_unimol,
        'morgan+cell_type+dose': np.hstack([X_morgan, cell_oh, dose_feat]),
        'unimol+cell_type+dose': np.hstack([X_unimol, cell_oh, dose_feat]),
    }

    train_idx, val_idx = scaffold_split(scaffolds.tolist(), train_frac=0.8, seed=42)

    key_genes = ['DDIT4', 'FOXO3', 'CASP3', 'LAMP1', 'SQSTM1', 'CDKN2A', 'CDKN1A',
                 'CTSD', 'TFEB', 'XIST', 'COL11A1', 'BAX', 'PARP1', 'GAA', 'ATG5',
                 'IL1B', 'CCL2', 'CASP7']

    print(f"\nTraining per-gene probes on {len(train_idx)} signatures, validating on {len(val_idx)}")
    print(f"{'Gene':<12}", end='')
    for name in feature_sets:
        print(f"{name:>22}", end='')
    print()
    print(" " * 12, end='')
    for _ in feature_sets:
        print(f"{'R2 / Spearman':>22}", end='')
    print()

    for gene in key_genes:
        gidx = gene_to_idx.get(gene)
        if gidx is None:
            continue
        y = Y[:, gidx]
        print(f"{gene:<12}", end='')
        for name, X in feature_sets.items():
            r2, rho = evaluate_probe(X[train_idx], X[val_idx], y[train_idx], y[val_idx])
            print(f"{r2:>6.3f} / {rho:>+6.3f}     ", end='')
        print()

    # Genome-wide scan using all genes in parallel
    print("\n" + "=" * 80)
    print("GENOME-WIDE SIGNAL PREVALENCE (Uni-Mol + cell + dose, all genes, parallel)")
    print("=" * 80)
    X = feature_sets['unimol+cell_type+dose']
    n_jobs = -1  # use all cores
    print(f"Fitting {Y.shape[1]} Ridge probes in parallel with joblib (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_one_gene)(g, X[train_idx], X[val_idx], Y[train_idx], Y[val_idx])
        for g in range(Y.shape[1])
    )
    r2s = np.array([r for (_, r, _) in results])
    spearmans = np.array([rho for (_, _, rho) in results])

    top_idx = np.argsort(np.abs(spearmans))[::-1][:20]
    print(f"\n{'Rank':<6}{'Gene':<15}{'Spearman':>12}{'R2':>12}")
    for rank, g in enumerate(top_idx, 1):
        print(f"{rank:<6}{gene_symbols[g]:<15}{spearmans[g]:>+12.4f}{r2s[g]:>12.4f}")

    print(f"\nGenes with |Spearman| > 0.2: {np.sum(np.abs(spearmans) > 0.2)} / {len(spearmans)}")
    print(f"Genes with R2 > 0: {np.sum(r2s > 0)} / {len(r2s)}")
    print(f"Genes with R2 > 0.1: {np.sum(r2s > 0.1)} / {len(r2s)}")

    print("=" * 80)


if __name__ == '__main__':
    main()
