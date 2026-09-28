"""
Improved autophagy signature extraction and inverse search.

Fixes based on 10-round review:
  1. Conservative HAMDB curation (only unambiguous activators/inhibitors).
  2. Strict core autophagy gene set (~75 genes, excluding generic stress genes).
  3. Per-cell-line signatures.
  4. Contrastive signatures: subtract random cell-matched background.
  5. Dose filtering (1-10 uM).
  6. Scaffold-aware cross-validation.
  7. Contrastive inverse-search score.
  8. Permutation-based null distribution and enrichment.
  9. Consistency filter (>=2 signatures per compound).

Outputs:
  - autophagy_activator_signature_v2.npz
  - autophagy_inhibitor_signature_v2.npz
  - activator_signature_hits_v2.csv
  - inhibitor_signature_hits_v2.csv
  - autophagy_signature_diagnostics_v2.csv
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from collections import defaultdict, Counter
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('extract_autophagy_signatures_v2')

# Strict core autophagy gene set (~75 genes)
STRICT_AUTOPHAGY_GENES = [
    # Core ATG machinery
    'ULK1', 'ULK2', 'ATG13', 'RB1CC1',
    'PIK3C3', 'PIK3R4', 'BECN1', 'ATG14', 'NRBF2', 'UVRAG',
    'ATG2A', 'ATG2B', 'ATG9A', 'ATG9B',
    'ATG3', 'ATG4A', 'ATG4B', 'ATG4C', 'ATG4D', 'ATG5', 'ATG7', 'ATG10', 'ATG12', 'ATG16L1', 'ATG101',
    # LC3/GABARAP family
    'MAP1LC3B', 'MAP1LC3C', 'GABARAPL1', 'GABARAPL2',
    'WIPI1', 'WIPI2',
    # Autophagy adaptors / receptors
    'SQSTM1', 'NBR1', 'TAX1BP1', 'OPTN', 'CALCOCO2',
    # Lysosomal / TFEB
    'LAMP1', 'LAMP2', 'CTSD', 'CTSL', 'CTSB', 'TPP1', 'GNS', 'NPC1', 'NPC2', 'TFEB', 'TFE3',
    # mTOR pathway readouts
    'DDIT4', 'EIF4EBP1', 'RPS6KB1', 'TSC1', 'TSC2', 'MTOR', 'RHEB',
    # AMPK
    'PRKAA1', 'PRKAA2', 'PRKAB1', 'PRKAB2', 'PRKAG1', 'PRKAG2', 'STK11',
    # Mitophagy / ER-phagy
    'PINK1', 'BNIP3', 'BNIP3L', 'RTN3', 'FAM134B',
    # Other core regulators
    'AMBRA1', 'DEPTOR', 'PTEN', 'FOXO3',
]


def get_numeric_pcids(series):
    ids = set()
    for x in series.dropna():
        s = str(x).strip()
        try:
            ids.add(str(int(float(s))))
        except Exception:
            pass
    return ids


def conservative_direction(row):
    """Conservative direction curation. Returns 'activator', 'inhibitor', or None (exclude)."""
    name = str(row.get('Chemcial_Name', '')).strip().lower()
    category = str(row.get('Category', '')).strip().lower()
    target = str(row.get('Target', '')).strip().lower()
    text = f"{category} {target}"

    # Explicit autophagy labels
    if 'autophagy activator' in category:
        return 'activator'
    if 'autophagy inhibitor' in category:
        return 'inhibitor'

    # Plain Activator / Inhibitor in HAMDB context
    if category == 'activator':
        return 'activator'
    if category == 'inhibitor':
        return 'inhibitor'

    # Well-established activator mechanisms
    activator_patterns = [
        r'\bmtor inhibitor\b',
        r'\bampk activator\b',
        r'\bproteasome inhibitor\b',
        r'\bhsp90 inhibitor\b',
        r'\bmek inhibitor\b',
        r'\bbraf inhibitor\b',
        r'\begfr inhibitor\b',
        r'\bher2 inhibitor\b',
    ]
    for p in activator_patterns:
        if re.search(p, text):
            return 'activator'

    # Well-established inhibitor mechanisms
    inhibitor_patterns = [
        r'\bulk1 inhibitor\b',
        r'\bvps34 inhibitor\b',
        r'\bvps34\b',
        r'\bp97 inhibitor\b',
        r'\blysosomal inhibitor\b',
        r'\bautophagic flux inhibitor\b',
    ]
    for p in inhibitor_patterns:
        if re.search(p, text):
            return 'inhibitor'

    # Manual overrides for famous compounds
    manual = {
        'rapamycin (sirolimus)': 'activator',
        'sirolimus': 'activator',
        'everolimus': 'activator',
        'temsirolimus': 'activator',
        'metformin': 'activator',
        'lithium': 'activator',
        'spermidine': 'activator',
        'chloroquine': 'inhibitor',
        'hydroxychloroquine': 'inhibitor',
        'bafilomycin a1': 'inhibitor',
        'vinblastine': 'inhibitor',
        'vincristine': 'inhibitor',
        'paclitaxel': 'inhibitor',
        'epothilone b': 'inhibitor',
    }
    if name in manual:
        return manual[name]

    # Exclude everything else (HDAC, PI3K, DUB, Aurora, calcium channel, etc.)
    return None


def get_scaffold(compound_smiles, smiles_to_scaffold):
    return smiles_to_scaffold.get(compound_smiles, compound_smiles)


def build_signature(Y, sig_indices, background_indices=None):
    """Build mean signature; optionally contrast against background."""
    mean_sig = np.nanmean(Y[sig_indices], axis=0)
    if background_indices is not None and len(background_indices) > 0:
        bg = np.nanmean(Y[background_indices], axis=0)
        mean_sig = mean_sig - bg
    return mean_sig


def signature_quality(Y, sig_indices, mean_sig):
    """Median correlation of individual signatures to the mean signature."""
    corrs = []
    for idx in sig_indices:
        y = Y[idx]
        if np.nanstd(y) < 1e-6:
            continue
        c, _ = spearmanr(y, mean_sig, nan_policy='omit')
        corrs.append(c)
    return float(np.nanmedian(corrs)) if corrs else np.nan


def main():
    logger.info("=== Improved autophagy signature extraction v2 ===")

    # Load HAMDB and curate conservatively
    logger.info("Loading and conservatively curating HAMDB...")
    hamdb = pd.read_csv('hamdb_autophagy_directions.csv')
    directions = []
    for _, row in hamdb.iterrows():
        directions.append(conservative_direction(row))
    hamdb['conservative_direction'] = directions

    activator_pcids = get_numeric_pcids(hamdb[hamdb['conservative_direction'] == 'activator']['Pubchem_CID'])
    inhibitor_pcids = get_numeric_pcids(hamdb[hamdb['conservative_direction'] == 'inhibitor']['Pubchem_CID'])
    logger.info(f"  Conservative activators: {len(activator_pcids)}, inhibitors: {len(inhibitor_pcids)}")

    # Load pert_info
    logger.info("Loading L1000 pert_info...")
    pert_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz', sep='\t',
                            usecols=['pert_id', 'pert_iname', 'canonical_smiles', 'pubchem_cid'])
    pert_info['numeric_pcid'] = pert_info['pubchem_cid'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else None
    )

    # Map to SMILES
    activator_smiles = set(pert_info[pert_info['numeric_pcid'].isin(activator_pcids)]['canonical_smiles'].dropna().unique())
    inhibitor_smiles = set(pert_info[pert_info['numeric_pcid'].isin(inhibitor_pcids)]['canonical_smiles'].dropna().unique())
    activator_smiles = {s for s in activator_smiles if s != '-666'}
    inhibitor_smiles = {s for s in inhibitor_smiles if s != '-666'}
    logger.info(f"  Activator SMILES in L1000: {len(activator_smiles)}, inhibitor SMILES: {len(inhibitor_smiles)}")

    # Load full cache
    logger.info("Loading full cache...")
    d = np.load('diag_cache_v9_morgan_full.npz', allow_pickle=True)
    Y_full = d['Y'].astype(np.float32)
    smiles_full = d['smiles']
    cell_ids_full = d['cell_ids']
    doses_full = d['doses']
    times_full = d['times']
    gene_symbols = list(d['gene_symbols'])
    cell_types = list(d['cell_types'])

    # Map strict gene set
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    strict_indices = [gene_to_idx[g] for g in STRICT_AUTOPHAGY_GENES if g in gene_to_idx]
    missing = [g for g in STRICT_AUTOPHAGY_GENES if g not in gene_to_idx]
    if missing:
        logger.warning(f"  {len(missing)} strict genes not in L1000: {missing}")
    logger.info(f"  Using {len(strict_indices)} strict autophagy genes")

    # Filter: 24h, dose 1-10 uM
    dose_mask = (doses_full >= 1.0) & (doses_full <= 10.0)
    time_mask = times_full == 24
    full_mask = dose_mask & time_mask
    Y = Y_full[np.ix_(full_mask, strict_indices)]
    smiles = smiles_full[full_mask]
    cell_ids = cell_ids_full[full_mask]
    logger.info(f"  24h, 1-10uM signatures: {Y.shape[0]}")

    # Get scaffolds for splitting
    logger.info("Loading/cacheing scaffolds...")
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    smiles_to_scaffold = {}
    for sm in set(smiles.tolist()) | activator_smiles | inhibitor_smiles:
        try:
            mol = Chem.MolFromSmiles(sm)
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol else sm
        except Exception:
            scaf = sm
        smiles_to_scaffold[sm] = scaf

    # Build per-cell-line signatures
    logger.info("Building per-cell-line contrastive signatures...")
    signatures = {}
    cell_diag = []

    for cell_idx, cell_name in enumerate(cell_types):
        cell_mask = cell_ids == cell_idx
        if cell_mask.sum() == 0:
            continue
        Y_cell = Y[cell_mask]
        smiles_cell = smiles[cell_mask]

        act_mask_cell = np.array([s in activator_smiles for s in smiles_cell])
        inh_mask_cell = np.array([s in inhibitor_smiles for s in smiles_cell])
        hamdb_mask_cell = act_mask_cell | inh_mask_cell
        nonhamdb_mask_cell = ~hamdb_mask_cell

        n_act = act_mask_cell.sum()
        n_inh = inh_mask_cell.sum()
        n_non = nonhamdb_mask_cell.sum()

        if n_act < 5 or n_inh < 3 or n_non < 20:
            continue

        act_indices = np.where(act_mask_cell)[0]
        inh_indices = np.where(inh_mask_cell)[0]
        non_indices = np.where(nonhamdb_mask_cell)[0]

        # Background: random subset of non-HAMDB, same size as activator set
        rng = np.random.default_rng(42)
        bg_indices = rng.choice(non_indices, size=min(n_act, n_non), replace=False)

        act_sig = build_signature(Y_cell, act_indices, bg_indices)
        inh_sig = build_signature(Y_cell, inh_indices, bg_indices)

        act_quality = signature_quality(Y_cell, act_indices, act_sig)
        inh_quality = signature_quality(Y_cell, inh_indices, inh_sig)

        signatures[cell_name] = {
            'activator': act_sig,
            'inhibitor': inh_sig,
            'background': np.nanmean(Y_cell[bg_indices], axis=0),
            'act_indices_global': np.where(full_mask)[0][cell_mask][act_indices],
            'inh_indices_global': np.where(full_mask)[0][cell_mask][inh_indices],
        }

        cell_diag.append({
            'cell_line': cell_name,
            'n_act': int(n_act),
            'n_inh': int(n_inh),
            'n_non': int(n_non),
            'act_quality': act_quality,
            'inh_quality': inh_quality,
        })
        logger.info(f"  {cell_name}: act={n_act}, inh={n_inh}, non={n_non}, act_quality={act_quality:.3f}, inh_quality={inh_quality:.3f}")

    if not signatures:
        logger.error("No cell lines had enough data!")
        return

    # Save signatures
    np.savez_compressed('autophagy_signatures_v2_per_cell.npz',
                        signatures=signatures,
                        gene_symbols=np.array([STRICT_AUTOPHAGY_GENES[i] for i in range(len(strict_indices))], dtype=object),
                        cell_lines=np.array(list(signatures.keys()), dtype=object))
    logger.info(f"Saved autophagy_signatures_v2_per_cell.npz with {len(signatures)} cell lines")

    # Cross-validation: build signature on 70% of activator scaffolds, test on 30%
    logger.info("\n=== Cross-validation ===")
    cv_results = []
    for cell_name, sigs in signatures.items():
        cell_idx = list(cell_types).index(cell_name)
        cell_mask_local = cell_ids == cell_idx
        Y_cell = Y[cell_mask_local]
        smiles_cell = smiles[cell_mask_local]

        act_smiles_cell = set(s for s in smiles_cell if s in activator_smiles)
        scaffolds = list(set(smiles_to_scaffold[s] for s in act_smiles_cell))
        rng = np.random.default_rng(42)
        rng.shuffle(scaffolds)
        n_train = int(0.7 * len(scaffolds))
        train_scafs = set(scaffolds[:n_train])
        test_scafs = set(scaffolds[n_train:])

        act_mask = np.array([s in activator_smiles for s in smiles_cell])
        train_mask = np.array([smiles_to_scaffold[s] in train_scafs for s in smiles_cell]) & act_mask
        test_mask = np.array([smiles_to_scaffold[s] in test_scafs for s in smiles_cell]) & act_mask

        if train_mask.sum() < 5 or test_mask.sum() < 3:
            continue

        # Build signature on train
        non_mask = ~np.array([s in activator_smiles or s in inhibitor_smiles for s in smiles_cell])
        non_indices = np.where(non_mask)[0]
        bg = np.nanmean(Y_cell[rng.choice(non_indices, size=min(train_mask.sum(), len(non_indices)), replace=False)], axis=0)
        train_sig = np.nanmean(Y_cell[train_mask], axis=0) - bg

        # Test: correlate test signatures to train signature
        test_corrs = []
        for idx in np.where(test_mask)[0]:
            c, _ = spearmanr(Y_cell[idx], train_sig, nan_policy='omit')
            test_corrs.append(c)
        median_test = float(np.nanmedian(test_corrs))

        # Compare to random
        rand_corrs = []
        for _ in range(50):
            rand_idx = rng.choice(non_indices, size=test_mask.sum(), replace=False)
            for idx in rand_idx:
                c, _ = spearmanr(Y_cell[idx], train_sig, nan_policy='omit')
                rand_corrs.append(c)
        median_rand = float(np.nanmedian(rand_corrs))

        cv_results.append({
            'cell_line': cell_name,
            'median_test_corr': median_test,
            'median_random_corr': median_rand,
            'delta': median_test - median_rand,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
        })
        logger.info(f"  {cell_name}: test={median_test:.3f}, random={median_rand:.3f}, delta={median_test-median_rand:.3f}")

    # Improved inverse search: contrastive score per compound
    logger.info("\n=== Improved inverse search ===")
    unique_smiles = sorted(set(smiles.tolist()))
    records = []
    for sm in unique_smiles:
        if sm == '-666':
            continue
        mask = smiles == sm
        n_sigs = mask.sum()
        if n_sigs < 2:
            continue  # consistency filter

        scores_act = []
        scores_inh = []
        scores_diff = []
        for idx in np.where(mask)[0]:
            cell_name = cell_types[cell_ids[idx]]
            if cell_name not in signatures:
                continue
            y = Y[idx]
            if np.nanstd(y) < 1e-6:
                continue
            act_sig = signatures[cell_name]['activator']
            inh_sig = signatures[cell_name]['inhibitor']
            bg = signatures[cell_name]['background']
            c_act = spearmanr(y, act_sig, nan_policy='omit')[0]
            c_inh = spearmanr(y, inh_sig, nan_policy='omit')[0]
            c_bg = spearmanr(y, bg, nan_policy='omit')[0]
            scores_act.append(c_act)
            scores_inh.append(c_inh)
            scores_diff.append(c_act - c_inh - c_bg)

        if not scores_act:
            continue

        records.append({
            'canonical_smiles': sm,
            'n_sigs': int(n_sigs),
            'activator_corr_mean': float(np.nanmean(scores_act)),
            'inhibitor_corr_mean': float(np.nanmean(scores_inh)),
            'contrastive_score': float(np.nanmean(scores_diff)),
            'is_hamdb_activator': sm in activator_smiles,
            'is_hamdb_inhibitor': sm in inhibitor_smiles,
        })

    df = pd.DataFrame(records)

    # Permutation null for contrastive score
    logger.info("Computing permutation null for contrastive score...")
    rng = np.random.default_rng(42)
    null_scores = []
    nonhamdb = df[(~df['is_hamdb_activator']) & (~df['is_hamdb_inhibitor'])]
    for _ in range(1000):
        sample = nonhamdb.sample(n=min(100, len(nonhamdb)), replace=False, random_state=rng)
        null_scores.extend(sample['contrastive_score'].tolist())
    null_scores = np.array(null_scores)
    mu_null = null_scores.mean()
    sd_null = null_scores.std()

    df['zscore'] = (df['contrastive_score'] - mu_null) / sd_null
    from scipy.stats import norm
    df['pvalue'] = 1 - norm.cdf(df['zscore'])

    # Save
    df.to_csv('autophagy_signature_hits_all_v2.csv', index=False)

    top_activator = df[(~df['is_hamdb_activator'])].sort_values('contrastive_score', ascending=False).head(200)
    top_inhibitor = df[(~df['is_hamdb_inhibitor'])].sort_values('contrastive_score', ascending=True).head(200)
    top_activator.to_csv('activator_signature_hits_v2.csv', index=False)
    top_inhibitor.to_csv('inhibitor_signature_hits_v2.csv', index=False)

    logger.info(f"Saved autophagy_signature_hits_all_v2.csv ({len(df)} compounds)")
    logger.info(f"Saved activator_signature_hits_v2.csv and inhibitor_signature_hits_v2.csv")

    # Save diagnostics
    diag_df = pd.DataFrame(cell_diag)
    cv_df = pd.DataFrame(cv_results)
    diag_df.to_csv('autophagy_signature_cell_diag_v2.csv', index=False)
    cv_df.to_csv('autophagy_signature_cv_v2.csv', index=False)

    logger.info("\nSummary:")
    logger.info(f"  Cell lines with signatures: {len(signatures)}")
    logger.info(f"  Compounds ranked: {len(df)}")
    logger.info(f"  Top contrastive score: {df['contrastive_score'].max():.4f}")
    logger.info(f"  HAMDB activator enrichment top 50: {df.sort_values('contrastive_score', ascending=False).head(50)['is_hamdb_activator'].mean():.1%}")


if __name__ == '__main__':
    main()
