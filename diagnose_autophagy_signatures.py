"""
Diagnostic tests for the autophagy signature inverse-search pipeline.

Tests:
1. Split-half reproducibility of activator/inhibitor signatures.
2. Null distribution: mean signature of random compound samples.
3. Cell-type composition of activator/inhibitor 24h signatures.
4. Enrichment of HAMDB labels in top inverse-search hits.
5. Correlation of individual activator/inhibitor signatures to their class means.
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('diagnose_signatures')


def main():
    # Load curated directions
    logger.info("Loading HAMDB directions...")
    hamdb = pd.read_csv('hamdb_autophagy_directions.csv')
    activator_names = set(hamdb[hamdb['autophagy_direction'] == 'activator']['Chemcial_Name'].str.lower())
    inhibitor_names = set(hamdb[hamdb['autophagy_direction'] == 'inhibitor']['Chemcial_Name'].str.lower())

    # Build name -> pcid mapping
    name_to_pcid = {}
    for _, row in hamdb.iterrows():
        name = str(row['Chemcial_Name']).strip().lower()
        if pd.notna(row['Pubchem_CID']):
            name_to_pcid[name] = str(int(float(row['Pubchem_CID'])))

    # Load pert_info
    logger.info("Loading L1000 pert_info...")
    pert_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz', sep='\t',
                            usecols=['pert_id', 'pert_iname', 'canonical_smiles', 'pubchem_cid'])
    pert_info['numeric_pcid'] = pert_info['pubchem_cid'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else None
    )

    # Map directions to L1000 SMILES
    activator_pcids = set(name_to_pcid.get(n) for n in activator_names)
    inhibitor_pcids = set(name_to_pcid.get(n) for n in inhibitor_names)
    activator_pcids.discard(None)
    inhibitor_pcids.discard(None)

    activator_smiles = set(pert_info[pert_info['numeric_pcid'].isin(activator_pcids)]['canonical_smiles'].dropna().unique())
    inhibitor_smiles = set(pert_info[pert_info['numeric_pcid'].isin(inhibitor_pcids)]['canonical_smiles'].dropna().unique())
    activator_smiles = {s for s in activator_smiles if s != '-666'}
    inhibitor_smiles = {s for s in inhibitor_smiles if s != '-666'}

    # Load cache
    logger.info("Loading full cache...")
    d = np.load('diag_cache_v9_morgan_full.npz', allow_pickle=True)
    Y = d['Y'].astype(np.float32)
    smiles = d['smiles']
    times = d['times']
    cell_ids = d['cell_ids']
    gene_symbols = list(d['gene_symbols'])
    cell_types = list(d['cell_types'])

    # Autophagy genes
    with open('autophagy_genes.txt', 'r') as f:
        auto_genes = [line.strip() for line in f if line.strip()]
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    auto_indices = [gene_to_idx[g] for g in auto_genes if g in gene_to_idx]

    # 24h subset
    mask24 = times == 24
    Y24 = Y[np.ix_(mask24, auto_indices)]
    smiles24 = smiles[mask24]
    cell_ids24 = cell_ids[mask24]

    act_mask = np.array([s in activator_smiles for s in smiles24])
    inh_mask = np.array([s in inhibitor_smiles for s in smiles24])

    act_indices = np.where(act_mask)[0]
    inh_indices = np.where(inh_mask)[0]
    n_act = len(act_indices)
    n_inh = len(inh_indices)

    logger.info(f"Activator 24h sigs: {n_act}, Inhibitor 24h sigs: {n_inh}")

    # 1. Split-half reproducibility
    logger.info("\n=== 1. Split-half reproducibility ===")
    rng = np.random.default_rng(42)
    n_splits = 100
    act_repro = []
    inh_repro = []
    for _ in range(n_splits):
        half1 = rng.choice(act_indices, size=n_act // 2, replace=False)
        half2 = rng.choice(np.setdiff1d(act_indices, half1), size=n_act // 2, replace=False)
        m1 = np.nanmean(Y24[half1], axis=0)
        m2 = np.nanmean(Y24[half2], axis=0)
        act_repro.append(spearmanr(m1, m2, nan_policy='omit')[0])

        half1 = rng.choice(inh_indices, size=n_inh // 2, replace=False)
        half2 = rng.choice(np.setdiff1d(inh_indices, half1), size=n_inh // 2, replace=False)
        m1 = np.nanmean(Y24[half1], axis=0)
        m2 = np.nanmean(Y24[half2], axis=0)
        inh_repro.append(spearmanr(m1, m2, nan_policy='omit')[0])

    logger.info(f"Activator split-half Spearman: mean={np.mean(act_repro):.3f}, std={np.std(act_repro):.3f}, min={np.min(act_repro):.3f}, max={np.max(act_repro):.3f}")
    logger.info(f"Inhibitor split-half Spearman: mean={np.mean(inh_repro):.3f}, std={np.std(inh_repro):.3f}, min={np.min(inh_repro):.3f}, max={np.max(inh_repro):.3f}")

    # 2. Null distribution
    logger.info("\n=== 2. Null distribution (random compound samples) ===")
    non_hamdb_mask = ~(act_mask | inh_mask)
    non_hamdb_indices = np.where(non_hamdb_mask)[0]
    act_mean = np.nanmean(Y24[act_indices], axis=0)
    inh_mean = np.nanmean(Y24[inh_indices], axis=0)

    null_act = []
    null_inh = []
    for _ in range(100):
        rand_act = rng.choice(non_hamdb_indices, size=n_act, replace=False)
        rand_inh = rng.choice(non_hamdb_indices, size=n_inh, replace=False)
        rand_act_mean = np.nanmean(Y24[rand_act], axis=0)
        rand_inh_mean = np.nanmean(Y24[rand_inh], axis=0)
        null_act.append(spearmanr(act_mean, rand_act_mean, nan_policy='omit')[0])
        null_inh.append(spearmanr(inh_mean, rand_inh_mean, nan_policy='omit')[0])

    logger.info(f"Random-vs-activator signature Spearman: mean={np.mean(null_act):.3f}, std={np.std(null_act):.3f}, max={np.max(null_act):.3f}")
    logger.info(f"Random-vs-inhibitor signature Spearman: mean={np.mean(null_inh):.3f}, std={np.std(null_inh):.3f}, max={np.max(null_inh):.3f}")

    # 3. Cell-type composition
    logger.info("\n=== 3. Cell-type composition ===")
    act_cell_counts = Counter([cell_types[i] for i in cell_ids24[act_indices]])
    inh_cell_counts = Counter([cell_types[i] for i in cell_ids24[inh_indices]])
    logger.info("Top cell types in activator signatures:")
    for cell, cnt in act_cell_counts.most_common(10):
        logger.info(f"  {cell}: {cnt}")
    logger.info("Top cell types in inhibitor signatures:")
    for cell, cnt in inh_cell_counts.most_common(10):
        logger.info(f"  {cell}: {cnt}")

    # 4. Enrichment of HAMDB labels in top hits
    logger.info("\n=== 4. Enrichment of HAMDB labels in top activator hits ===")
    # Compute correlations for all non-HAMDB and HAMDB activator signatures
    all_unique_smiles = sorted(set(smiles24))
    smile_to_mean_corr = {}
    for sm in all_unique_smiles:
        mask = smiles24 == sm
        corrs = []
        for idx in np.where(mask)[0]:
            c, _ = spearmanr(Y24[idx], act_mean, nan_policy='omit')
            corrs.append(c)
        smile_to_mean_corr[sm] = np.nanmean(corrs)

    df_hits = pd.DataFrame([
        {'smiles': sm, 'corr': smile_to_mean_corr[sm],
         'is_activator': sm in activator_smiles,
         'is_inhibitor': sm in inhibitor_smiles}
        for sm in all_unique_smiles
    ])
    df_hits = df_hits.sort_values('corr', ascending=False)

    for top_n in [50, 100, 200, 500]:
        top = df_hits.head(top_n)
        n_act = top['is_activator'].sum()
        n_inh = top['is_inhibitor'].sum()
        frac_act = n_act / top_n
        logger.info(f"  Top {top_n}: {n_act} HAMDB activators ({frac_act:.1%}), {n_inh} HAMDB inhibitors")

    # Compare to baseline prevalence
    baseline_act = df_hits['is_activator'].mean()
    baseline_inh = df_hits['is_inhibitor'].mean()
    logger.info(f"  Baseline: {baseline_act:.1%} HAMDB activators, {baseline_inh:.1%} HAMDB inhibitors in all {len(df_hits)} compounds")

    # 5. Within-class correlation distribution
    logger.info("\n=== 5. Within-class correlation to class mean ===")
    act_self_corrs = []
    for idx in act_indices:
        c, _ = spearmanr(Y24[idx], act_mean, nan_policy='omit')
        act_self_corrs.append(c)
    inh_self_corrs = []
    for idx in inh_indices:
        c, _ = spearmanr(Y24[idx], inh_mean, nan_policy='omit')
        inh_self_corrs.append(c)

    logger.info(f"Activator signatures vs activator mean: mean={np.nanmean(act_self_corrs):.3f}, std={np.nanstd(act_self_corrs):.3f}, median={np.nanmedian(act_self_corrs):.3f}")
    logger.info(f"Inhibitor signatures vs inhibitor mean: mean={np.nanmean(inh_self_corrs):.3f}, std={np.nanstd(inh_self_corrs):.3f}, median={np.nanmedian(inh_self_corrs):.3f}")

    # Save diagnostics
    diag = {
        'activator_split_half_mean': float(np.mean(act_repro)),
        'activator_split_half_std': float(np.std(act_repro)),
        'inhibitor_split_half_mean': float(np.mean(inh_repro)),
        'inhibitor_split_half_std': float(np.std(inh_repro)),
        'null_vs_activator_mean': float(np.mean(null_act)),
        'null_vs_activator_max': float(np.max(null_act)),
        'null_vs_inhibitor_mean': float(np.mean(null_inh)),
        'null_vs_inhibitor_max': float(np.max(null_inh)),
        'activator_self_corr_mean': float(np.nanmean(act_self_corrs)),
        'activator_self_corr_median': float(np.nanmedian(act_self_corrs)),
        'inhibitor_self_corr_mean': float(np.nanmean(inh_self_corrs)),
        'inhibitor_self_corr_median': float(np.nanmedian(inh_self_corrs)),
    }
    pd.DataFrame([diag]).to_csv('autophagy_signature_diagnostics.csv', index=False)
    logger.info("\nSaved autophagy_signature_diagnostics.csv")


if __name__ == '__main__':
    main()
