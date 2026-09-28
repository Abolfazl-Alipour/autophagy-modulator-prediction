"""
Cache training data for diagnostics.
Loads the same 18k-ish signatures used by V5/V6 (control SMILES excluded),
saves Uni-Mol embeddings, Morgan fingerprints, transcriptome profiles,
cell types, SMILES, and scaffolds to diag_cache.npz.
"""
from pathlib import Path
import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(str(Path(__file__).resolve().parent))
from dataset_l1000_fixed import L1000DataManager, L1000ChunkedDataset, get_murcko_scaffold
from batch_correction import load_baselines

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('diag_cache')


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def main():
    SUBSET_LIMIT = 20000
    CACHE_PATH = 'diag_cache.npz'

    logger.info("Loading Uni-Mol embeddings...")
    unimol_embs = torch.load('unimol_embeddings_l1000_10k.pt', map_location='cpu', weights_only=True)

    logger.info("Loading metadata...")
    meta_df = pd.read_csv('embedded_compounds_metadata_10k.csv')
    control_smiles = set(meta_df[meta_df['is_control'] == True]['smiles'].tolist())

    logger.info("Setting up DataManager...")
    dm = L1000DataManager(
        gctx_path="data/l1000/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        sig_info_path="data/l1000/GSE92742_Broad_LINCS_sig_info.txt.gz",
        pert_info_path="data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz"
    )
    dm.load_metadata()
    cell_types = dm.meta['cell_id'].dropna().unique().tolist()
    cell_type_map = {ct: i for i, ct in enumerate(cell_types)}
    baselines = load_baselines('cell_type_baselines.npz')

    logger.info("Loading chunked dataset (this may take a few minutes)...")
    dataset = L1000ChunkedDataset(
        dm, unimol_embs, cell_type_map,
        chunk_size=10000, cache_dir="chunk_cache_12k_v3",
        subset_limit=SUBSET_LIMIT,
        baselines=baselines,
        exclude_smiles=control_smiles
    )

    n = len(dataset.valid_indices)
    logger.info(f"Dataset loaded: {n} signatures")

    # Build arrays
    logger.info("Extracting embeddings, fingerprints, and metadata...")
    X_unimol = np.zeros((n, 512), dtype=np.float32)
    X_morgan = np.zeros((n, 2048), dtype=np.float32)
    Y = dataset.profiles.astype(np.float32)  # [N, 12328]
    cell_ids = np.zeros(n, dtype=np.int32)
    smiles_list = []
    scaffolds = []
    doses = np.zeros(n, dtype=np.float32)

    for i, (chunk_idx, local_idx, sig_id, smiles) in enumerate(dataset.valid_indices):
        X_unimol[i] = unimol_embs[smiles].numpy() if isinstance(unimol_embs[smiles], torch.Tensor) else unimol_embs[smiles]
        X_morgan[i] = smiles_to_morgan(smiles)
        cell_ids[i] = dataset[i]['cell_type'].item()
        smiles_list.append(smiles)
        scaffolds.append(get_murcko_scaffold(smiles))
        # dose
        info = dm.meta.loc[sig_id] if sig_id in dm.meta.index else {}
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
        try:
            doses[i] = float(info.get('pert_dose', 0.0))
        except Exception:
            doses[i] = 0.0

    # Save
    logger.info(f"Saving cache to {CACHE_PATH}...")
    np.savez(
        CACHE_PATH,
        X_unimol=X_unimol,
        X_morgan=X_morgan,
        Y=Y,
        cell_ids=cell_ids,
        doses=doses,
        smiles=np.array(smiles_list, dtype=object),
        scaffolds=np.array(scaffolds, dtype=object),
        cell_types=np.array(cell_types, dtype=object),
        gene_symbols=np.array(dm.gene_symbols, dtype=object)
    )
    logger.info("Cache saved.")


if __name__ == '__main__':
    main()
