#!/usr/bin/env python3
"""
Generate ROC curve data for the binary autophagy modulator classifiers.

Loads the saved XGB models from train_3class_autophagy_classifier.py,
reconstructs the same scaffold-aware train/test split, and computes
FPR/TPR curves for:
  - XGBoost: Morgan only
  - XGBoost: Morgan + CLAMP
  - Ridge baseline: Morgan only
  - Ridge baseline: Morgan + CLAMP

Outputs:
  - binary_roc_curves.npz  (FPR/TPR/thresholds per model)
  - binary_roc_curves.csv  (AUROC summary)
"""
import warnings
import pickle
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import RidgeClassifier
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

WORK_DIR = Path(__file__).resolve().parent
HAMDB_PATH = WORK_DIR / 'hamdb_autophagy_directions.csv'
CHEMBL_CANDIDATES_PATH = WORK_DIR / 'chembl_neutral_candidates.csv'

RANDOM_STATE = 42
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
CLAMP_BATCH_SIZE = 64
N_JOBS = 12


def get_murcko_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf else None
    except Exception:
        return None


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


def compute_scaffolds(smiles_list, n_jobs=N_JOBS):
    return Parallel(n_jobs=n_jobs)(delayed(get_murcko_scaffold)(s) for s in smiles_list)


def cluster_scaffolds(scaffold_smiles, cutoff=0.7):
    """Cluster scaffold SMILES by Tanimoto similarity using Butina."""
    unique_scaffolds = list(set(s for s in scaffold_smiles if s))
    logger.info(f'Clustering {len(unique_scaffolds)} unique scaffolds...')

    fps = []
    valid = []
    for smi in unique_scaffolds:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        fps.append(fp)
        valid.append(smi)

    n = len(fps)
    if n == 0:
        return {}
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    scaffold_to_cluster = {}
    for cid, members in enumerate(clusters):
        for idx in members:
            scaffold_to_cluster[valid[idx]] = cid
    return scaffold_to_cluster


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


def load_hamdb():
    hamdb = pd.read_csv(HAMDB_PATH)
    hamdb = hamdb.dropna(subset=['Canonical_SMILES'])
    hamdb['scaffold'] = compute_scaffolds(hamdb['Canonical_SMILES'].tolist(), n_jobs=N_JOBS)
    hamdb = hamdb.dropna(subset=['scaffold'])
    return hamdb


def load_chembl_neutrals():
    df = pd.read_csv(CHEMBL_CANDIDATES_PATH)
    df = df.dropna(subset=['smiles']).drop_duplicates(subset=['smiles'])
    df['scaffold'] = compute_scaffolds(df['smiles'].tolist(), n_jobs=N_JOBS)
    df = df.dropna(subset=['scaffold'])
    return df


def sample_neutrals_by_scaffold_family(chembl_df, n_per_family=5, random_state=RANDOM_STATE):
    scaffold_to_cluster = cluster_scaffolds(chembl_df['scaffold'].tolist(), cutoff=0.7)
    chembl_df = chembl_df.copy()
    chembl_df['scaffold_family'] = chembl_df['scaffold'].map(scaffold_to_cluster)
    chembl_df = chembl_df.dropna(subset=['scaffold_family'])

    rng = np.random.default_rng(random_state)
    sampled = []
    for family, group in chembl_df.groupby('scaffold_family'):
        if len(group) <= n_per_family:
            sampled.append(group)
        else:
            sampled.append(group.sample(n=n_per_family, random_state=rng.integers(0, 2**31)))

    return pd.concat(sampled, ignore_index=True)


def stratified_scaffold_split(df, test_frac=0.2, random_state=RANDOM_STATE):
    """Split by scaffold family, stratified by class."""
    df = df.copy()
    rng = np.random.default_rng(random_state)

    scaffolds = df['scaffold'].unique()
    rng.shuffle(scaffolds)

    class_scaffolds = defaultdict(list)
    for scaf in scaffolds:
        classes = set(df[df['scaffold'] == scaf]['label'])
        for c in classes:
            class_scaffolds[c].append(scaf)

    test_scaffolds = set()
    for c, scafs in class_scaffolds.items():
        n_test = max(1, int(len(scafs) * test_frac))
        test_scaffolds.update(rng.choice(scafs, size=n_test, replace=False))

    test_mask = df['scaffold'].isin(test_scaffolds)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    return train_df, test_df


def roc_from_scores(y_true, y_score, label):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    logger.info(f'{label} AUROC = {auroc:.4f}')
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auroc': auroc,
    }


def main():
    logger.info('=== Loading HAMDB ===')
    hamdb = load_hamdb()
    activators = hamdb[hamdb['autophagy_direction'] == 'activator'].copy()
    inhibitors = hamdb[hamdb['autophagy_direction'] == 'inhibitor'].copy()
    logger.info(f'HAMDB activators: {len(activators)}, inhibitors: {len(inhibitors)}')

    logger.info('=== Loading ChEMBL neutrals ===')
    chembl_all = load_chembl_neutrals()
    chembl_neutral = sample_neutrals_by_scaffold_family(chembl_all, n_per_family=5)
    logger.info(f'Sampled {len(chembl_neutral)} ChEMBL neutral compounds')

    logger.info('=== Building dataset ===')
    train_records = []
    for _, row in activators.iterrows():
        train_records.append({
            'name': row.get('Chemcial_Name'),
            'smiles': row['Canonical_SMILES'],
            'scaffold': row['scaffold'],
            'source': 'HAMDB',
            'label': 'activator',
            'binary_label': 1,
        })
    for _, row in inhibitors.iterrows():
        train_records.append({
            'name': row.get('Chemcial_Name'),
            'smiles': row['Canonical_SMILES'],
            'scaffold': row['scaffold'],
            'source': 'HAMDB',
            'label': 'inhibitor',
            'binary_label': 1,
        })
    for _, row in chembl_neutral.iterrows():
        train_records.append({
            'name': row['chembl_id'],
            'smiles': row['smiles'],
            'scaffold': row['scaffold'],
            'source': 'ChEMBL',
            'label': 'neutral',
            'binary_label': 0,
        })

    train_df = pd.DataFrame(train_records)
    logger.info(f'Total compounds: {len(train_df)}')
    logger.info(train_df['label'].value_counts())

    logger.info('=== Computing Morgan fingerprints ===')
    morgan_fps = Parallel(n_jobs=N_JOBS)(delayed(smiles_to_morgan)(s) for s in train_df['smiles'])
    valid_mask = np.array([fp is not None for fp in morgan_fps])
    train_df = train_df[valid_mask].reset_index(drop=True)
    morgan_matrix = np.vstack([fp for fp in morgan_fps if fp is not None])
    morgan_cols = [f'morgan_{i}' for i in range(MORGAN_BITS)]
    train_df[morgan_cols] = morgan_matrix

    logger.info('=== Computing CLAMP embeddings ===')
    clamp_model = load_clamp_model()
    clamp_embs = clamp_embed_smiles_list(train_df['smiles'].tolist(), clamp_model)
    clamp_cols = [f'clamp_{i}' for i in range(clamp_embs.shape[1])]
    train_df[clamp_cols] = clamp_embs
    del clamp_model

    logger.info('=== Splitting data ===')
    train_split, test_split = stratified_scaffold_split(train_df, test_frac=0.2)
    logger.info(f'Train: {len(train_split)}, Test: {len(test_split)}')
    logger.info('Test class distribution:')
    logger.info(test_split['label'].value_counts())

    feature_sets = {
        'Morgan only': morgan_cols,
        'Morgan + CLAMP': morgan_cols + clamp_cols,
    }

    roc_data = {}
    auroc_records = []

    logger.info('=== Loading saved XGB models and computing ROC curves ===')
    for feat_name, feat_cols in feature_sets.items():
        model_key = 'binary_morgan_only' if feat_name == 'Morgan only' else 'binary_morgan_clamp'
        model_path = WORK_DIR / f'{model_key}_xgb.pkl'

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        X_test = test_split[feat_cols].values
        y_test = test_split['binary_label'].values
        y_proba = model.predict_proba(X_test)[:, 1]

        roc = roc_from_scores(y_test, y_proba, f'XGB {feat_name}')
        roc_data[f'XGB {feat_name}'] = roc
        auroc_records.append({'model': f'XGB {feat_name}', 'auroc': roc['auroc']})

    logger.info('=== Training Ridge baselines and computing ROC curves ===')
    for feat_name, feat_cols in feature_sets.items():
        X_train = train_split[feat_cols].values
        y_train = train_split['binary_label'].values
        X_test = test_split[feat_cols].values
        y_test = test_split['binary_label'].values

        clf = RidgeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_score = clf.decision_function(X_test)

        roc = roc_from_scores(y_test, y_score, f'Ridge {feat_name}')
        roc_data[f'Ridge {feat_name}'] = roc
        auroc_records.append({'model': f'Ridge {feat_name}', 'auroc': roc['auroc']})

    logger.info('=== Saving ROC curve data ===')
    np.savez(
        WORK_DIR / 'binary_roc_curves.npz',
        **{k: np.array([v['fpr'], v['tpr'], v['thresholds']], dtype=object)
           for k, v in roc_data.items()},
        aurocs=np.array([(k, v['auroc']) for k, v in roc_data.items()], dtype=object)
    )

    auroc_df = pd.DataFrame(auroc_records)
    auroc_df.to_csv(WORK_DIR / 'binary_roc_curves.csv', index=False)
    logger.info('\nAUROC summary:')
    logger.info(auroc_df.to_string(index=False))

    logger.info('Done.')


if __name__ == '__main__':
    main()
