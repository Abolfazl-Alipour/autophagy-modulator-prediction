#!/usr/bin/env python3
"""
Head-to-head ChEMBL enrichment comparison:
  - binary_morgan_only_xgb.pkl  vs.
  - binary_morgan_clamp_xgb.pkl

Both models are from train_3class_autophagy_classifier.py (structure-only).
We score the same ChEMBL compounds with both, then compare enrichment for
autophagy-active / mTOR/PI3K target annotations at the same thresholds.

Outputs:
  - chembl_enrichment_morgan_vs_clamp.csv
"""
import warnings
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

WORK_DIR = Path(__file__).resolve().parent
CHEMBL_CANDIDATES_PATH = WORK_DIR / 'chembl_neutral_candidates.csv'
MORGAN_MODEL_PATH = WORK_DIR / 'binary_morgan_only_xgb.pkl'
CLAMP_MODEL_PATH = WORK_DIR / 'binary_morgan_clamp_xgb.pkl'
CACHE_DIR = WORK_DIR / 'chembl_cache'

RANDOM_STATE = 42
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
CLAMP_BATCH_SIZE = 64
N_JOBS = 12


def smiles_to_morgan(smiles, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def load_clamp_model():
    from clamp.models.pretrained import PretrainedCLAMP
    model = PretrainedCLAMP(device='cpu')
    model.eval()
    return model


def clamp_embed_smiles_list(smiles_list, clamp_model, batch_size=CLAMP_BATCH_SIZE):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            emb = clamp_model.encode_smiles(batch).cpu().numpy().astype(np.float32)
            embeddings.append(emb)
    return np.vstack(embeddings)


def compute_enrichment(df, score_col, thresholds):
    """Compute annotation enrichment at thresholds."""
    n_total = len(df)
    rows = []
    for t in thresholds:
        subset = df[df[score_col] >= t]
        if len(subset) == 0:
            continue
        rows.append({
            'threshold': t,
            'n_compounds': len(subset),
            'pct_of_total': 100 * len(subset) / n_total,
            'autophagy_active_pct': 100 * subset['autophagy_active'].fillna(False).astype(bool).mean(),
            'mtor_pi3k_pct': 100 * subset['mtor_pi3k_target'].fillna(False).astype(bool).mean(),
            'lysosomal_pct': 100 * subset['lysosomal_target'].fillna(False).astype(bool).mean(),
        })
    return pd.DataFrame(rows)


def main():
    logger.info('=== Loading ChEMBL candidates with annotations ===')
    chembl_df = pd.read_csv(CHEMBL_CANDIDATES_PATH)
    chembl_df = chembl_df.dropna(subset=['smiles']).drop_duplicates(subset=['smiles']).reset_index(drop=True)

    # Load existing annotations if available from cascade output
    cascade_path = WORK_DIR / 'chembl_cascade_predictions_all.csv'
    if cascade_path.exists():
        ann_df = pd.read_csv(cascade_path, usecols=['chembl_id', 'autophagy_active', 'mtor_pi3k_target', 'lysosomal_target'])
        chembl_df = chembl_df.merge(ann_df, on='chembl_id', how='left')
        logger.info(f'Loaded annotations for {chembl_df["autophagy_active"].notna().sum()} compounds')
    else:
        logger.warning('No cascade annotations found; annotation columns will be empty')
        chembl_df['autophagy_active'] = False
        chembl_df['mtor_pi3k_target'] = False
        chembl_df['lysosomal_target'] = False

    logger.info(f'ChEMBL compounds to score: {len(chembl_df)}')

    logger.info('=== Computing Morgan fingerprints ===')
    morgan_fps = Parallel(n_jobs=N_JOBS)(delayed(smiles_to_morgan)(s) for s in chembl_df['smiles'])
    valid_mask = np.array([fp is not None for fp in morgan_fps])
    chembl_df = chembl_df[valid_mask].reset_index(drop=True)
    morgan_matrix = np.vstack([fp for fp in morgan_fps if fp is not None])
    logger.info(f'Valid compounds after Morgan: {len(chembl_df)}')

    logger.info('=== Computing CLAMP embeddings ===')
    clamp_model = load_clamp_model()
    clamp_matrix = clamp_embed_smiles_list(chembl_df['smiles'].tolist(), clamp_model)
    del clamp_model

    logger.info('=== Loading binary models ===')
    with open(MORGAN_MODEL_PATH, 'rb') as f:
        morgan_model = pickle.load(f)
    with open(CLAMP_MODEL_PATH, 'rb') as f:
        clamp_model = pickle.load(f)

    logger.info('=== Scoring ChEMBL ===')
    chembl_df['modulator_proba_morgan'] = morgan_model.predict_proba(morgan_matrix)[:, 1]
    chembl_df['modulator_proba_clamp'] = clamp_model.predict_proba(np.hstack([morgan_matrix, clamp_matrix]))[:, 1]

    logger.info('=== Computing enrichment ===')
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    morgan_enrich = compute_enrichment(chembl_df, 'modulator_proba_morgan', thresholds)
    morgan_enrich['model'] = 'Morgan only'

    clamp_enrich = compute_enrichment(chembl_df, 'modulator_proba_clamp', thresholds)
    clamp_enrich['model'] = 'Morgan + CLAMP'

    comparison = pd.concat([morgan_enrich, clamp_enrich], ignore_index=True)
    comparison = comparison[['model', 'threshold', 'n_compounds', 'pct_of_total',
                              'autophagy_active_pct', 'mtor_pi3k_pct', 'lysosomal_pct']]
    comparison = comparison.sort_values(['threshold', 'model']).reset_index(drop=True)

    out_path = WORK_DIR / 'chembl_enrichment_morgan_vs_clamp.csv'
    comparison.to_csv(out_path, index=False)
    logger.info(f'\nSaved {out_path}')
    logger.info('\n' + comparison.to_string(index=False))

    # Top hits overlap
    logger.info('\n=== Top-200 hit overlap between models ===')
    top_morgan = set(chembl_df.nlargest(200, 'modulator_proba_morgan')['chembl_id'])
    top_clamp = set(chembl_df.nlargest(200, 'modulator_proba_clamp')['chembl_id'])
    logger.info(f'Overlap: {len(top_morgan & top_clamp)} / 200')


if __name__ == '__main__':
    main()
