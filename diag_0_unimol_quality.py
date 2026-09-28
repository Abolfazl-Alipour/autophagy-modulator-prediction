"""
Diagnostic 0: Uni-Mol Embedding Quality Check

Tests whether precomputed Uni-Mol embeddings encode chemical similarity
and whether chemical similarity correlates with transcriptomic similarity.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_distances


def tanimoto_distance(fp1, fp2):
    # Tanimoto = 1 - Tanimoto similarity
    inter = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    if union == 0:
        return 0.0
    return 1.0 - inter / union


def main():
    print("=" * 80)
    print("DIAGNOSTIC 0: Uni-Mol Embedding Quality Check")
    print("=" * 80)

    print("\nLoading cached training data...")
    cache = np.load('diag_cache.npz', allow_pickle=True)
    X_unimol = cache['X_unimol']
    X_morgan = cache['X_morgan']
    Y = cache['Y']
    smiles = cache['smiles']

    n_total = len(smiles)
    n_sample = min(200, n_total)
    np.random.seed(42)
    idx = np.random.choice(n_total, n_sample, replace=False)

    X_u = X_unimol[idx]
    X_m = X_morgan[idx].astype(bool)
    Y_s = Y[idx]

    print(f"Sampled {n_sample} compounds for pairwise analysis.")

    # Pairwise distances
    print("\nComputing pairwise distance matrices...")
    n = n_sample
    k = n * (n - 1) // 2

    unimol_dist = np.zeros(k)
    morgan_dist = np.zeros(k)
    profile_dist = np.zeros(k)

    # Compute all pairwise distances efficiently
    # Uni-Mol cosine distance
    u_dist_mat = cosine_distances(X_u)
    # Morgan Tanimoto distance
    m_sim_mat = np.zeros((n, n))
    for i in range(n):
        inter = np.sum(X_m[i] & X_m, axis=1)
        union = np.sum(X_m[i] | X_m, axis=1)
        m_sim_mat[i, :] = np.where(union > 0, inter / union, 0.0)
    m_dist_mat = 1.0 - m_sim_mat

    # Profile Spearman distance = 1 - |Spearman| or 1 - Spearman? Use 1 - Spearman.
    p_dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                p_dist_mat[i, j] = 0.0
            else:
                rho, _ = spearmanr(Y_s[i], Y_s[j])
                p_dist_mat[i, j] = 1.0 - rho
                p_dist_mat[j, i] = 1.0 - rho

    # Extract upper triangle
    triu_idx = np.triu_indices(n, k=1)
    unimol_dist = u_dist_mat[triu_idx]
    morgan_dist = m_dist_mat[triu_idx]
    profile_dist = p_dist_mat[triu_idx]

    # Correlations
    print("\n--- Correlation between distance metrics ---")
    corr_um_morgan, _ = pearsonr(unimol_dist, morgan_dist)
    corr_um_profile, _ = pearsonr(unimol_dist, profile_dist)
    corr_morgan_profile, _ = pearsonr(morgan_dist, profile_dist)

    print(f"Pearson(Uni-Mol distance, Morgan distance)      = {corr_um_morgan:+.4f}")
    print(f"Pearson(Uni-Mol distance, Profile distance)     = {corr_um_profile:+.4f}")
    print(f"Pearson(Morgan distance, Profile distance)      = {corr_morgan_profile:+.4f}")

    # Rank correlations
    print("\n--- Rank correlations (Spearman) ---")
    s_um_morgan, _ = spearmanr(unimol_dist, morgan_dist)
    s_um_profile, _ = spearmanr(unimol_dist, profile_dist)
    s_morgan_profile, _ = spearmanr(morgan_dist, profile_dist)
    print(f"Spearman(Uni-Mol distance, Morgan distance)     = {s_um_morgan:+.4f}")
    print(f"Spearman(Uni-Mol distance, Profile distance)    = {s_um_profile:+.4f}")
    print(f"Spearman(Morgan distance, Profile distance)     = {s_morgan_profile:+.4f}")

    # Distribution stats
    print("\n--- Distance distributions ---")
    print(f"Uni-Mol cosine distance:  mean={np.mean(unimol_dist):.4f}, std={np.std(unimol_dist):.4f}")
    print(f"Morgan Tanimoto distance: mean={np.mean(morgan_dist):.4f}, std={np.std(morgan_dist):.4f}")
    print(f"Profile Spearman distance: mean={np.mean(profile_dist):.4f}, std={np.std(profile_dist):.4f}")

    # Interpretation
    print("\n--- Interpretation ---")
    if abs(corr_um_profile) < 0.05:
        print("Uni-Mol distance does NOT correlate with transcriptomic profile distance.")
        print("The embeddings are not informative for predicting L1000 profiles.")
    elif abs(corr_um_profile) < 0.20:
        print("Uni-Mol distance has a WEAK correlation with transcriptomic profile distance.")
        print("Chemistry explains only a small fraction of transcriptomic variation.")
    else:
        print("Uni-Mol distance has a MODERATE/STRONG correlation with transcriptomic profile distance.")

    if abs(corr_um_morgan) < 0.30:
        print("Uni-Mol distance does NOT correlate well with Morgan chemical similarity.")
        print("Uni-Mol may be encoding different chemical features than standard fingerprints.")
    else:
        print("Uni-Mol distance correlates reasonably with Morgan chemical similarity.")

    print("=" * 80)


if __name__ == '__main__':
    main()
