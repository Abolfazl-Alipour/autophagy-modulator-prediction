"""
Extract 24h transcriptomic signatures for HAMDB-curated autophagy activators and inhibitors,
then perform inverse search: find all L1000 compounds whose 24h signature matches each reference.

Inputs:
  - hamdb_autophagy_directions.csv (curated directions)
  - diag_cache_v9_morgan_full.npz (L1000 signatures)
  - autophagy_genes.txt (199 autophagy genes)

Outputs:
  - autophagy_activator_signature.npz (mean/median 24h activator signature)
  - autophagy_inhibitor_signature.npz (mean/median 24h inhibitor signature)
  - activator_signature_hits.csv (top matching compounds)
  - inhibitor_signature_hits.csv (top matching compounds)
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('extract_autophagy_signatures')


def get_numeric_pcids(series):
    ids = set()
    for x in series.dropna():
        s = str(x).strip()
        try:
            ids.add(str(int(float(s))))
        except Exception:
            pass
    return ids


def main():
    # Load autophagy gene list
    logger.info("Loading autophagy gene list...")
    with open('autophagy_genes.txt', 'r') as f:
        auto_genes = [line.strip() for line in f if line.strip()]

    # Load HAMDB curated directions
    logger.info("Loading curated HAMDB directions...")
    hamdb = pd.read_csv('hamdb_autophagy_directions.csv')
    activator_pcids = get_numeric_pcids(hamdb[hamdb['autophagy_direction'] == 'activator']['Pubchem_CID'])
    inhibitor_pcids = get_numeric_pcids(hamdb[hamdb['autophagy_direction'] == 'inhibitor']['Pubchem_CID'])
    logger.info(f"  Activators: {len(activator_pcids)} compounds, Inhibitors: {len(inhibitor_pcids)} compounds")

    # Load L1000 pert_info
    logger.info("Loading L1000 pert_info...")
    pert_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz', sep='\t',
                            usecols=['pert_id', 'pert_iname', 'canonical_smiles', 'pubchem_cid'])
    pert_info['numeric_pcid'] = pert_info['pubchem_cid'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else None
    )

    # Map PCIDs to SMILES
    activator_smiles = set(pert_info[pert_info['numeric_pcid'].isin(activator_pcids)]['canonical_smiles'].dropna().unique())
    inhibitor_smiles = set(pert_info[pert_info['numeric_pcid'].isin(inhibitor_pcids)]['canonical_smiles'].dropna().unique())
    activator_smiles = {s for s in activator_smiles if s != '-666'}
    inhibitor_smiles = {s for s in inhibitor_smiles if s != '-666'}
    logger.info(f"  Activator SMILES in L1000: {len(activator_smiles)}")
    logger.info(f"  Inhibitor SMILES in L1000: {len(inhibitor_smiles)}")

    # Load full cache
    logger.info("Loading full L1000 cache...")
    d = np.load('diag_cache_v9_morgan_full.npz', allow_pickle=True)
    Y_full = d['Y']
    smiles_full = d['smiles']
    times_full = d['times']
    gene_symbols = list(d['gene_symbols'])
    logger.info(f"  Full cache: {Y_full.shape}")

    # Map to autophagy genes
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    auto_indices = [gene_to_idx[g] for g in auto_genes if g in gene_to_idx]
    missing = [g for g in auto_genes if g not in gene_to_idx]
    if missing:
        logger.warning(f"  {len(missing)} autophagy genes not in L1000: {missing[:10]}")
    logger.info(f"  Using {len(auto_indices)} autophagy genes")

    # Filter to 24h signatures
    mask_24h = times_full == 24
    Y_24h = Y_full[np.ix_(mask_24h, auto_indices)].astype(np.float32)
    smiles_24h = smiles_full[mask_24h]
    logger.info(f"  24h signatures: {Y_24h.shape[0]}")

    # Identify activator / inhibitor 24h signatures
    activator_mask = np.array([sm in activator_smiles for sm in smiles_24h])
    inhibitor_mask = np.array([sm in inhibitor_smiles for sm in smiles_24h])
    logger.info(f"  Activator 24h signatures: {activator_mask.sum()}")
    logger.info(f"  Inhibitor 24h signatures: {inhibitor_mask.sum()}")

    if activator_mask.sum() == 0 or inhibitor_mask.sum() == 0:
        logger.error("No 24h signatures found for activators or inhibitors!")
        return

    # Compute mean/median signatures
    Y_activator = Y_24h[activator_mask]
    Y_inhibitor = Y_24h[inhibitor_mask]

    activator_mean = np.nanmean(Y_activator, axis=0)
    activator_median = np.nanmedian(Y_activator, axis=0)
    inhibitor_mean = np.nanmean(Y_inhibitor, axis=0)
    inhibitor_median = np.nanmedian(Y_inhibitor, axis=0)

    np.savez_compressed('autophagy_activator_signature.npz',
                        mean=activator_mean,
                        median=activator_median,
                        gene_symbols=np.array(auto_genes, dtype=object),
                        n_signatures=int(activator_mask.sum()),
                        n_compounds=len(activator_smiles & set(smiles_24h)))
    logger.info("Saved autophagy_activator_signature.npz")

    np.savez_compressed('autophagy_inhibitor_signature.npz',
                        mean=inhibitor_mean,
                        median=inhibitor_median,
                        gene_symbols=np.array(auto_genes, dtype=object),
                        n_signatures=int(inhibitor_mask.sum()),
                        n_compounds=len(inhibitor_smiles & set(smiles_24h)))
    logger.info("Saved autophagy_inhibitor_signature.npz")

    # Inverse search: compute Spearman correlation of each 24h signature to each reference
    logger.info("Computing inverse search (Spearman vs activator signature)...")
    n = Y_24h.shape[0]
    activator_corr = np.zeros(n, dtype=np.float32)
    inhibitor_corr = np.zeros(n, dtype=np.float32)

    for i in range(n):
        y = Y_24h[i]
        if np.nanstd(y) < 1e-6:
            continue
        activator_corr[i], _ = spearmanr(y, activator_mean, nan_policy='omit')
        inhibitor_corr[i], _ = spearmanr(y, inhibitor_mean, nan_policy='omit')

    # Aggregate per compound (best signature and mean signature correlation)
    unique_smiles = sorted(set(smiles_24h))
    records = []
    for sm in unique_smiles:
        mask = smiles_24h == sm
        records.append({
            'canonical_smiles': sm,
            'n_24h_sigs': int(mask.sum()),
            'activator_corr_max': float(np.nanmax(activator_corr[mask])),
            'activator_corr_mean': float(np.nanmean(activator_corr[mask])),
            'inhibitor_corr_max': float(np.nanmax(inhibitor_corr[mask])),
            'inhibitor_corr_mean': float(np.nanmean(inhibitor_corr[mask])),
            'is_hamdb_activator': sm in activator_smiles,
            'is_hamdb_inhibitor': sm in inhibitor_smiles,
        })

    df = pd.DataFrame(records)

    # Save full ranking
    df.to_csv('autophagy_signature_hits_all.csv', index=False)
    logger.info("Saved autophagy_signature_hits_all.csv")

    # Save top activator hits (excluding known HAMDB activators)
    top_activator_hits = df[~df['is_hamdb_activator']].sort_values('activator_corr_mean', ascending=False).head(200)
    top_activator_hits.to_csv('activator_signature_hits.csv', index=False)
    logger.info(f"Saved activator_signature_hits.csv with {len(top_activator_hits)} top hits")

    # Save top inhibitor hits (excluding known HAMDB inhibitors)
    top_inhibitor_hits = df[~df['is_hamdb_inhibitor']].sort_values('inhibitor_corr_mean', ascending=False).head(200)
    top_inhibitor_hits.to_csv('inhibitor_signature_hits.csv', index=False)
    logger.info(f"Saved inhibitor_signature_hits.csv with {len(top_inhibitor_hits)} top hits")

    # Summary
    n_activator_compounds = len(set(smiles_24h[activator_mask]))
    n_inhibitor_compounds = len(set(smiles_24h[inhibitor_mask]))
    logger.info("\nSummary:")
    logger.info(f"  Activator 24h sigs: {activator_mask.sum()}, compounds: {n_activator_compounds}")
    logger.info(f"  Inhibitor 24h sigs: {inhibitor_mask.sum()}, compounds: {n_inhibitor_compounds}")
    logger.info(f"  Top non-HAMDB activator hit mean corr: {top_activator_hits['activator_corr_mean'].iloc[0]:.4f}")
    logger.info(f"  Top non-HAMDB inhibitor hit mean corr: {top_inhibitor_hits['inhibitor_corr_mean'].iloc[0]:.4f}")


if __name__ == '__main__':
    main()
