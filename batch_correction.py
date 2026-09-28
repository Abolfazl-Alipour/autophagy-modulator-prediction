"""
Batch Correction for L1000 Data.

Strategy: Compute per-cell-type mean expression from DMSO/vehicle controls,
then subtract from each drug-treated signature. This removes cell-type baseline
expression (XIST, COL11A1, etc.) while preserving drug-specific perturbation.

NOTE: L1000 Level 5 (MODZ) data is already z-scored within each plate,
but NOT mean-centered per cell type across experiments. The dominant
variance is still cell-type identity, which drowns out drug responses.
"""

import os
import gc
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from cmapPy.pandasGEXpress.parse import parse

logger = logging.getLogger('NAFS2')

def compute_cell_type_baselines(gctx_path, sig_info_path, gene_symbols, chunk_size=10000):
    """
    Compute mean expression profile per cell type from DMSO controls.
    
    For each cell type, we average all DMSO-treated signatures to get
    the cell-type baseline. This baseline is then subtracted from all
    drug-treated signatures for that cell type.
    
    Returns: dict[cell_type -> np.array[12328]]
    """
    sig_info = pd.read_csv(sig_info_path, sep='\t', low_memory=False)
    
    # Filter to DMSO/vehicle controls
    dmso_sigs = sig_info[
        (sig_info['pert_type'] == 'ctl_vehicle') | 
        (sig_info['pert_iname'].str.upper().isin(['DMSO', 'UNTRT', 'UNTRTED']))
    ]
    
    logger.info(f"Found {len(dmso_sigs)} DMSO/vehicle control signatures")
    logger.info(f"Cell types in controls: {dmso_sigs['cell_id'].nunique()}")
    
    # Parse GCTX column metadata to get all sig_ids
    col_df = parse(str(gctx_path), col_meta_only=True)
    all_sig_ids = col_df.index.tolist()
    
    # Find which DMSO sig_ids are actually in the GCTX
    dmso_sig_ids = set(dmso_sigs['sig_id'].values) & set(all_sig_ids)
    logger.info(f"DMSO sigs present in GCTX: {len(dmso_sig_ids)}")
    
    # Build cell_type -> [sig_id list] mapping for DMSOs
    dmso_sigs_indexed = dmso_sigs.set_index('sig_id')
    cell_dmso_map = {}
    for sig_id in dmso_sig_ids:
        if sig_id in dmso_sigs_indexed.index:
            row = dmso_sigs_indexed.loc[sig_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            ct = row['cell_id']
            if ct not in cell_dmso_map:
                cell_dmso_map[ct] = []
            cell_dmso_map[ct].append(sig_id)
    
    logger.info(f"Cell types with DMSO controls: {len(cell_dmso_map)}")
    for ct, sigs in sorted(cell_dmso_map.items(), key=lambda x: -len(x[1]))[:10]:
        logger.info(f"  {ct}: {len(sigs)} DMSO signatures")
    
    # Compute mean profile per cell type by loading chunks
    n_genes = len(gene_symbols)
    cell_sums = {ct: np.zeros(n_genes) for ct in cell_dmso_map}
    cell_counts = {ct: 0 for ct in cell_dmso_map}
    
    # Process in chunks to avoid loading everything at once
    dmso_id_list = list(dmso_sig_ids)
    for start in range(0, len(dmso_id_list), chunk_size):
        chunk_ids = dmso_id_list[start:start+chunk_size]
        logger.info(f"Loading DMSO chunk {start}:{start+len(chunk_ids)} ({len(chunk_ids)} sigs)...")
        
        gctoo = parse(str(gctx_path), cid=chunk_ids)
        data = gctoo.data_df.values  # [n_genes, n_sigs]
        sig_ids_loaded = gctoo.col_metadata_df.index.tolist()
        
        for j, sid in enumerate(sig_ids_loaded):
            if sid in dmso_sigs_indexed.index:
                row = dmso_sigs_indexed.loc[sid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                ct = row['cell_id']
                if ct in cell_sums:
                    profile = data[:, j]
                    if not np.any(np.isnan(profile)):
                        cell_sums[ct] += profile
                        cell_counts[ct] += 1
        
        del gctoo, data
        gc.collect()
    
    # Compute means
    baselines = {}
    for ct in cell_dmso_map:
        if cell_counts[ct] > 0:
            baselines[ct] = cell_sums[ct] / cell_counts[ct]
            logger.info(f"  {ct}: baseline from {cell_counts[ct]} DMSO sigs, mean abs = {np.mean(np.abs(baselines[ct])):.4f}")
        else:
            logger.warning(f"  {ct}: no valid DMSO profiles, using zero baseline")
            baselines[ct] = np.zeros(n_genes)
    
    return baselines


def save_baselines(baselines, gene_symbols, path='cell_type_baselines.npz'):
    """Save computed baselines to disk."""
    arrays = {f"ct_{ct}": arr for ct, arr in baselines.items()}
    arrays['cell_types'] = np.array(list(baselines.keys()))
    arrays['gene_symbols'] = np.array(gene_symbols)
    np.savez(path, **arrays)
    logger.info(f"Saved {len(baselines)} cell-type baselines to {path}")


def load_baselines(path='cell_type_baselines.npz'):
    """Load precomputed baselines from disk."""
    data = np.load(path, allow_pickle=True)
    cell_types = data['cell_types']
    baselines = {}
    for ct in cell_types:
        key = f"ct_{ct}"
        if key in data:
            baselines[str(ct)] = data[key]
    logger.info(f"Loaded {len(baselines)} cell-type baselines from {path}")
    return baselines


def apply_batch_correction(profile, cell_type, baselines):
    """
    Subtract cell-type baseline from a gene expression profile.
    
    profile: np.array[12328] - raw MODZ z-scores
    cell_type: str - cell line name (e.g., 'A549')
    baselines: dict[str -> np.array[12328]] - per-cell-type means from DMSO
    
    Returns: np.array[12328] - batch-corrected profile
    """
    if cell_type in baselines:
        return profile - baselines[cell_type]
    else:
        # If no baseline available, return raw profile (conservative fallback)
        return profile


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    
    gctx_path = "data/l1000/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
    sig_info_path = "data/l1000/GSE92742_Broad_LINCS_sig_info.txt.gz"
    
    # Get gene symbols
    row_df = parse(gctx_path, row_meta_only=True)
    gene_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_gene_info.txt.gz', sep='\t')
    gene_map = dict(zip(gene_info['pr_gene_id'].astype(str), gene_info['pr_gene_symbol']))
    gene_symbols = [gene_map.get(str(i), str(i)) for i in row_df.index]
    
    logger.info(f"Computing cell-type baselines from DMSO controls...")
    baselines = compute_cell_type_baselines(gctx_path, sig_info_path, gene_symbols)
    save_baselines(baselines, gene_symbols)
    
    # Quick sanity check: what are the top genes in the A549 baseline?
    if 'A549' in baselines:
        bl = baselines['A549']
        gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
        sorted_idx = np.argsort(bl)
        logger.info("A549 baseline — top 10 upregulated (to be subtracted):")
        for i in sorted_idx[-10:][::-1]:
            logger.info(f"  {gene_symbols[i]}: {bl[i]:.4f}")
        logger.info("A549 baseline — top 10 downregulated:")
        for i in sorted_idx[:10]:
            logger.info(f"  {gene_symbols[i]}: {bl[i]:.4f}")
        
        # Check XIST specifically
        if 'XIST' in gene_to_idx:
            logger.info(f"XIST baseline value in A549: {bl[gene_to_idx['XIST']]:.4f}")
        if 'COL11A1' in gene_to_idx:
            logger.info(f"COL11A1 baseline value in A549: {bl[gene_to_idx['COL11A1']]:.4f}")
