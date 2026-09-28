"""
Verify empirical DDIT4 and FOXO3 values for Rapamycin in A549 at 6h
from GSE92742 Level 5 MODZ data.

Reports raw MODZ z-scores and DMSO-baseline-corrected values.
"""
import sys
import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

GCTX_PATH = "data/l1000/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
SIG_INFO_PATH = "data/l1000/GSE92742_Broad_LINCS_sig_info.txt.gz"
PERT_INFO_PATH = "data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz"
GENE_INFO_PATH = "data/l1000/GSE92742_Broad_LINCS_gene_info.txt.gz"


def load_gene_symbols():
    row_df = parse(GCTX_PATH, row_meta_only=True)
    gene_info = pd.read_csv(GENE_INFO_PATH, sep='\t')
    gene_map = dict(zip(gene_info['pr_gene_id'].astype(str), gene_info['pr_gene_symbol']))
    symbols = [gene_map.get(str(i), str(i)) for i in row_df.index]
    return symbols


def load_baselines():
    """Compute per-cell-type DMSO mean profile."""
    sig_info = pd.read_csv(SIG_INFO_PATH, sep='\t', low_memory=False)
    dmso_sigs = sig_info[
        (sig_info['pert_type'] == 'ctl_vehicle') |
        (sig_info['pert_iname'].str.upper().isin(['DMSO', 'UNTRT', 'UNTRTED']))
    ]
    col_df = parse(GCTX_PATH, col_meta_only=True)
    all_sig_ids = set(col_df.index.tolist())
    dmso_sig_ids = list(set(dmso_sigs['sig_id'].values) & all_sig_ids)

    dmso_indexed = dmso_sigs.set_index('sig_id')
    cell_sums = {}
    cell_counts = {}

    chunk_size = 5000
    for start in range(0, len(dmso_sig_ids), chunk_size):
        chunk = dmso_sig_ids[start:start + chunk_size]
        gctoo = parse(GCTX_PATH, cid=chunk)
        data = gctoo.data_df.values
        loaded_ids = gctoo.col_metadata_df.index.tolist()
        for j, sid in enumerate(loaded_ids):
            row = dmso_indexed.loc[sid]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            ct = row['cell_id']
            if ct not in cell_sums:
                cell_sums[ct] = np.zeros(data.shape[0])
                cell_counts[ct] = 0
            prof = data[:, j]
            if not np.any(np.isnan(prof)):
                cell_sums[ct] += prof
                cell_counts[ct] += 1

    baselines = {ct: cell_sums[ct] / cell_counts[ct] for ct in cell_sums if cell_counts[ct] > 0}
    return baselines


def main():
    print("=" * 80)
    print("GROUND-TRUTH VERIFICATION: Rapamycin / A549 / 6h")
    print("=" * 80)

    symbols = load_gene_symbols()
    gene_to_idx = {g: i for i, g in enumerate(symbols)}
    for gene in ['DDIT4', 'FOXO3']:
        if gene not in gene_to_idx:
            print(f"ERROR: {gene} not found in gene list")
            sys.exit(1)

    # Load metadata
    sig_info = pd.read_csv(SIG_INFO_PATH, sep='\t', low_memory=False)
    pert_info = pd.read_csv(PERT_INFO_PATH, sep='\t', low_memory=False)
    meta = sig_info.merge(pert_info[['pert_id', 'canonical_smiles', 'inchi_key']], on='pert_id', how='left')
    meta.set_index('sig_id', inplace=True)
    meta['pert_iname_lower'] = meta['pert_iname'].str.lower()

    # Find Rapamycin / A549 / 6h signatures
    mask = (meta['pert_iname_lower'].isin(['rapamycin', 'sirolimus'])) & \
           (meta['pert_time'] == 6) & \
           (meta['cell_id'] == 'A549')
    matching = meta[mask].copy()
    print(f"\nFound {len(matching)} Rapamycin/sirolimus signatures in A549 at 6h")
    print(matching[['pert_iname', 'pert_dose', 'pert_dose_unit', 'pert_time', 'cell_id']].head(10))

    if len(matching) == 0:
        print("No matching signatures found.")
        sys.exit(0)

    sig_ids = matching.index.tolist()
    gctoo = parse(GCTX_PATH, cid=sig_ids)
    # cmapPy may reorder columns; map sig_id -> column position
    col_order = gctoo.col_metadata_df.index.tolist()
    sig_to_col = {sig: i for i, sig in enumerate(col_order)}
    data = gctoo.data_df.values  # [n_genes, n_sigs]

    # Raw values
    ddit4_raw = data[gene_to_idx['DDIT4'], :]
    foxo3_raw = data[gene_to_idx['FOXO3'], :]

    print("\n--- RAW MOD Z-SCORES ---")
    print(f"DDIT4:  mean={np.mean(ddit4_raw):+.4f}, std={np.std(ddit4_raw):.4f}, median={np.median(ddit4_raw):+.4f}, n={len(ddit4_raw)}")
    print(f"FOXO3:  mean={np.mean(foxo3_raw):+.4f}, std={np.std(foxo3_raw):.4f}, median={np.median(foxo3_raw):+.4f}, n={len(foxo3_raw)}")

    # Per-signature table
    print("\n--- Per-signature raw values (top 10 by dose) ---")
    matching = matching.sort_values('pert_dose', ascending=False)
    for idx, (_, row) in enumerate(matching.head(10).iterrows()):
        sig = row.name
        j = sig_to_col[sig]
        print(f"  {sig:<60} | dose={row['pert_dose']:>8} | DDIT4={ddit4_raw[j]:+7.4f} | FOXO3={foxo3_raw[j]:+7.4f}")

    # Baseline-corrected values
    print("\n--- DMSO-BASELINE-CORRECTED VALUES ---")
    baselines = load_baselines()
    if 'A549' in baselines:
        bl = baselines['A549']
        ddit4_corr = ddit4_raw - bl[gene_to_idx['DDIT4']]
        foxo3_corr = foxo3_raw - bl[gene_to_idx['FOXO3']]
        print(f"A549 DMSO baseline: DDIT4={bl[gene_to_idx['DDIT4']]:+.4f}, FOXO3={bl[gene_to_idx['FOXO3']]:+.4f}")
        print(f"DDIT4 corrected: mean={np.mean(ddit4_corr):+.4f}, std={np.std(ddit4_corr):.4f}, median={np.median(ddit4_corr):+.4f}")
        print(f"FOXO3 corrected: mean={np.mean(foxo3_corr):+.4f}, std={np.std(foxo3_corr):.4f}, median={np.median(foxo3_corr):+.4f}")
    else:
        print("A549 baseline not found; cannot correct.")

    print("\n--- HIGHEST-DOSE SINGLE SIGNATURE ---")
    best_sig = matching.iloc[0].name
    j = sig_to_col[best_sig]
    print(f"Signature: {best_sig}")
    print(f"  Raw:      DDIT4={ddit4_raw[j]:+.4f}, FOXO3={foxo3_raw[j]:+.4f}")
    if 'A549' in baselines:
        print(f"  Corrected: DDIT4={ddit4_corr[j]:+.4f}, FOXO3={foxo3_corr[j]:+.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
