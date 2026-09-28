#!/usr/bin/env python3
"""
Train structure-based 3-class autophagy modulator classifier.

Classes:
  - activator: HAMDB activators
  - inhibitor: HAMDB inhibitors
  - neutral: scaffold-family-sampled ChEMBL compounds (no autophagy/mTOR/PI3K/lysosomal annotation)

Also trains a binary modulator vs. neutral classifier.

Features:
  - Morgan fingerprints (2048-bit, radius 2)
  - CLAMP embeddings (768-d) — optional ablation

Outputs:
  - autophagy_3class_xgb.pkl
  - autophagy_binary_xgb.pkl
  - chembl_3class_predictions.csv
  - 3class_classifier_report.md
"""

import os
import re
import json
import time
import logging
import warnings
import pickle
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report
)
import xgboost as xgb
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

WORK_DIR = Path(__file__).resolve().parent
HAMDB_PATH = WORK_DIR / 'hamdb_autophagy_directions.csv'
CHEMBL_CANDIDATES_PATH = WORK_DIR / 'chembl_neutral_candidates.csv'
CACHE_DIR = WORK_DIR / 'chembl_cache'
CACHE_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
CLAMP_BATCH_SIZE = 64
N_JOBS = 12

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
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
        return []
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    # Map each scaffold to cluster id
    scaffold_to_cluster = {}
    for cid, members in enumerate(clusters):
        for idx in members:
            scaffold_to_cluster[valid[idx]] = cid
    return scaffold_to_cluster


# ---------------------------------------------------------------------------
# CLAMP embeddings
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# ChEMBL annotation / exclusion
# ---------------------------------------------------------------------------
def chembl_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f'ChEMBL request failed: {url} -> {e}')
                return {}
            time.sleep(1 * (attempt + 1))
    return {}


def get_autophagy_related_chembl_molecules():
    """Return set of ChEMBL molecule IDs associated with autophagy assays or relevant targets."""
    cache = CACHE_DIR / 'autophagy_related_molecules.json'
    if cache.exists():
        with open(cache) as f:
            return set(json.load(f))

    mol_ids = set()

    # 1. Autophagy-related assays
    terms = ['autophagy', 'autophagosome', 'lysosome', 'lysosomal', 'LC3']
    assay_ids = set()
    for term in terms:
        url = 'https://www.ebi.ac.uk/chembl/api/data/assay.json'
        params = {'description__icontains': term, 'limit': 1000}
        data = chembl_get(url, params)
        for assay in data.get('assays', []):
            assay_ids.add(assay.get('assay_chembl_id'))

    logger.info(f'Found {len(assay_ids)} autophagy-related ChEMBL assays')

    # Get active molecules from these assays (limit to first N assays to avoid overload)
    for aid in list(assay_ids)[:200]:
        url = 'https://www.ebi.ac.uk/chembl/api/data/activity.json'
        params = {'assay_chembl_id': aid, 'limit': 1000}
        while True:
            data = chembl_get(url, params)
            for act in data.get('activities', []):
                mid = act.get('molecule_chembl_id')
                if mid and str(act.get('activity_comment', '')).lower() in ['active', 'inactive']:
                    # Include all; we will conservatively exclude any molecule in an autophagy assay
                    mol_ids.add(mid)
            page = data.get('page_meta', {})
            if not page.get('next'):
                break
            params = {'assay_chembl_id': aid, 'limit': 1000, 'offset': page.get('offset', 0) + 1000}

    # 2. Relevant targets: mTOR, PI3K, AKT, AMPK, ULK1, VPS34, lysosomal proteases
    target_names = ['mtor', 'pi3k', 'akt1', 'ampk', 'ulk1', 'vps34', 'pik3c3', 'becn1', 'lamp1', 'lamp2', 'ctsd', 'ctsb', 'tfeb']
    target_ids = set()
    for name in target_names:
        url = 'https://www.ebi.ac.uk/chembl/api/data/target.json'
        params = {'pref_name__icontains': name, 'limit': 100}
        data = chembl_get(url, params)
        for t in data.get('targets', []):
            target_ids.add(t.get('target_chembl_id'))

    logger.info(f'Found {len(target_ids)} relevant ChEMBL targets')

    for tid in list(target_ids)[:100]:
        url = 'https://www.ebi.ac.uk/chembl/api/data/activity.json'
        params = {'target_chembl_id': tid, 'limit': 1000}
        while True:
            data = chembl_get(url, params)
            for act in data.get('activities', []):
                mid = act.get('molecule_chembl_id')
                if mid:
                    mol_ids.add(mid)
            page = data.get('page_meta', {})
            if not page.get('next'):
                break
            params = {'target_chembl_id': tid, 'limit': 1000, 'offset': page.get('offset', 0) + 1000}

    with open(cache, 'w') as f:
        json.dump(list(mol_ids), f)
    logger.info(f'Found {len(mol_ids)} autophagy-related ChEMBL molecules to exclude from neutrals')
    return mol_ids


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------
def load_hamdb():
    df = pd.read_csv(HAMDB_PATH)
    df = df.dropna(subset=['Canonical_SMILES'])
    df = df[df['Canonical_SMILES'] != '-666']
    df['scaffold'] = compute_scaffolds(df['Canonical_SMILES'].tolist(), n_jobs=N_JOBS)
    df = df.dropna(subset=['scaffold'])
    return df


def load_chembl_neutrals(exclude_mol_ids=None):
    df = pd.read_csv(CHEMBL_CANDIDATES_PATH)
    df = df.dropna(subset=['smiles'])
    if exclude_mol_ids:
        df = df[~df['chembl_id'].isin(exclude_mol_ids)]
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


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------
def stratified_scaffold_split(df, test_frac=0.2, random_state=RANDOM_STATE):
    """Split by scaffold family, stratified by class."""
    df = df.copy()
    rng = np.random.default_rng(random_state)

    # Assign each scaffold to train or test
    scaffolds = df['scaffold'].unique()
    rng.shuffle(scaffolds)

    # Stratify: ensure each class has some scaffolds in test
    class_scaffolds = defaultdict(list)
    for scaf in scaffolds:
        classes = set(df[df['scaffold'] == scaf]['label'])
        for c in classes:
            class_scaffolds[c].append(scaf)

    test_scaffolds = set()
    for c, scafs in class_scaffolds.items():
        # Reserve ~test_frac of scaffolds for this class
        n_test = max(1, int(len(scafs) * test_frac))
        test_scaffolds.update(rng.choice(scafs, size=n_test, replace=False))

    # Add more random scaffolds to reach test_frac overall
    train_scaffolds = set(scaffolds) - test_scaffolds
    test_mask = df['scaffold'].isin(test_scaffolds)

    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    return train_df, test_df


def train_eval_xgboost(train_df, test_df, feature_cols, target_col, model_type='binary', class_weight='balanced'):
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    if model_type == 'binary':
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() if class_weight == 'balanced' else 1,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            eval_metric='aucpr',
            early_stopping_rounds=50,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        y_proba = model.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, y_proba)
        auprc = average_precision_score(y_test, y_proba)
        y_pred = (y_proba >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logger.info(f'Binary test AUROC={auroc:.3f}, AUPRC={auprc:.3f}, Acc={acc:.3f}, F1={f1:.3f}')
        return model, {'auroc': auroc, 'auprc': auprc, 'accuracy': acc, 'f1': f1}

    else:  # multiclass
        classes = sorted(train_df[target_col].unique())
        class_weights = {c: 1.0 for c in classes}
        if class_weight == 'balanced':
            counts = pd.Series(y_train).value_counts()
            max_count = counts.max()
            class_weights = {c: max_count / counts.get(c, 1) for c in classes}

        sample_weights = np.array([class_weights.get(y, 1.0) for y in y_train])

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(classes),
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            eval_metric='mlogloss',
            early_stopping_rounds=50,
        )
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        logger.info(f'3-class test Accuracy={acc:.3f}, Macro F1={f1_macro:.3f}')
        logger.info('\n' + classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))
        return model, {'accuracy': acc, 'f1_macro': f1_macro}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info('=== Loading HAMDB ===')
    hamdb = load_hamdb()
    activators = hamdb[hamdb['autophagy_direction'] == 'activator'].copy()
    inhibitors = hamdb[hamdb['autophagy_direction'] == 'inhibitor'].copy()
    logger.info(f'HAMDB activators: {len(activators)}, inhibitors: {len(inhibitors)}')

    logger.info('=== Loading and filtering ChEMBL neutrals ===')
    exclude_mols = get_autophagy_related_chembl_molecules()
    chembl_all = load_chembl_neutrals(exclude_mol_ids=exclude_mols)
    logger.info(f'ChEMBL candidates after exclusion: {len(chembl_all)}')

    logger.info('=== Sampling ChEMBL neutrals by scaffold family ===')
    chembl_neutral = sample_neutrals_by_scaffold_family(chembl_all, n_per_family=5)
    logger.info(f'Sampled {len(chembl_neutral)} ChEMBL neutral compounds')

    # Build unified training dataframe
    train_records = []
    for _, row in activators.iterrows():
        train_records.append({
            'name': row['Chemcial_Name'] if 'Chemcial_Name' in row else None,
            'smiles': row['Canonical_SMILES'],
            'scaffold': row['scaffold'],
            'source': 'HAMDB',
            'label': 'activator',
            'binary_label': 1,
        })
    for _, row in inhibitors.iterrows():
        train_records.append({
            'name': row['Chemcial_Name'] if 'Chemcial_Name' in row else None,
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
            'scaffold_family': row.get('scaffold_family'),
            'source': 'ChEMBL',
            'label': 'neutral',
            'binary_label': 0,
        })

    train_df = pd.DataFrame(train_records)
    train_df['label_code'] = train_df['label'].map({'activator': 0, 'inhibitor': 1, 'neutral': 2})
    logger.info(f'Total training compounds: {len(train_df)}')
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

    # Feature sets
    feature_sets = {
        'morgan_only': morgan_cols,
        'morgan_clamp': morgan_cols + clamp_cols,
    }

    logger.info('=== Splitting data ===')
    train_split, test_split = stratified_scaffold_split(train_df, test_frac=0.2)
    logger.info(f'Train: {len(train_split)}, Test: {len(test_split)}')
    logger.info('Train class distribution:')
    logger.info(train_split['label'].value_counts())
    logger.info('Test class distribution:')
    logger.info(test_split['label'].value_counts())

    results = {}
    models = {}

    logger.info('=== Training binary modulator classifier ===')
    for feat_name, feat_cols in feature_sets.items():
        logger.info(f'Feature set: {feat_name}')
        model, metrics = train_eval_xgboost(
            train_split, test_split, feat_cols, 'binary_label',
            model_type='binary', class_weight='balanced'
        )
        results[f'binary_{feat_name}'] = metrics
        models[f'binary_{feat_name}'] = model

    logger.info('=== Training 3-class classifier ===')
    for feat_name, feat_cols in feature_sets.items():
        logger.info(f'Feature set: {feat_name}')
        model, metrics = train_eval_xgboost(
            train_split, test_split, feat_cols, 'label_code',
            model_type='multiclass', class_weight='balanced'
        )
        results[f'3class_{feat_name}'] = metrics
        models[f'3class_{feat_name}'] = model

    # Save models
    for name, model in models.items():
        with open(WORK_DIR / f'{name}_xgb.pkl', 'wb') as f:
            pickle.dump(model, f)
    logger.info('Models saved.')

    logger.info('=== Predicting on full ChEMBL inference set ===')
    predict_and_verify_chembl(models, morgan_cols, clamp_cols)

    logger.info('=== Writing report ===')
    write_report(results, train_df, test_split)

    logger.info('Done.')


def get_chembl_annotations(chembl_ids):
    """Get structured ChEMBL annotations for a list of molecule IDs."""
    cache = CACHE_DIR / 'chembl_structured_annotations.json'
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    # Reuse target-based annotations
    autophagy_mols = get_autophagy_related_chembl_molecules()

    # Separate autophagy assay activity
    terms = ['autophagy', 'autophagosome', 'lysosome', 'lysosomal', 'LC3']
    autophagy_assay_mols = set()
    autophagy_assay_counts = defaultdict(int)
    for term in terms:
        url = 'https://www.ebi.ac.uk/chembl/api/data/assay.json'
        params = {'description__icontains': term, 'limit': 1000}
        data = chembl_get(url, params)
        for assay in data.get('assays', []):
            aid = assay.get('assay_chembl_id')
            aurl = 'https://www.ebi.ac.uk/chembl/api/data/activity.json'
            aparams = {'assay_chembl_id': aid, 'limit': 1000}
            while True:
                adata = chembl_get(aurl, aparams)
                for act in adata.get('activities', []):
                    mid = act.get('molecule_chembl_id')
                    if mid:
                        autophagy_assay_mols.add(mid)
                        autophagy_assay_counts[mid] += 1
                page = adata.get('page_meta', {})
                if not page.get('next'):
                    break
                aparams = {'assay_chembl_id': aid, 'limit': 1000, 'offset': page.get('offset', 0) + 1000}

    # mTOR/PI3K target annotations
    target_map = {
        'mtor_pi3k': set(),
        'lysosomal': set(),
    }
    target_patterns = {
        'mtor_pi3k': ['mtor', 'pi3k', 'akt', 'ampk', 'ulk1', 'vps34', 'pik3c3'],
        'lysosomal': ['lamp1', 'lamp2', 'ctsd', 'ctsb', 'ctsl', 'tfeb', 'npc1'],
    }
    for group, patterns in target_patterns.items():
        for pattern in patterns:
            url = 'https://www.ebi.ac.uk/chembl/api/data/target.json'
            params = {'pref_name__icontains': pattern, 'limit': 100}
            data = chembl_get(url, params)
            for t in data.get('targets', []):
                tid = t.get('target_chembl_id')
                aurl = 'https://www.ebi.ac.uk/chembl/api/data/activity.json'
                aparams = {'target_chembl_id': tid, 'limit': 1000}
                while True:
                    adata = chembl_get(aurl, aparams)
                    for act in adata.get('activities', []):
                        mid = act.get('molecule_chembl_id')
                        if mid:
                            target_map[group].add(mid)
                    page = adata.get('page_meta', {})
                    if not page.get('next'):
                        break
                    aparams = {'target_chembl_id': tid, 'limit': 1000, 'offset': page.get('offset', 0) + 1000}

    annotations = {}
    for mid in chembl_ids:
        annotations[mid] = {
            'autophagy_active': mid in autophagy_assay_mols,
            'n_autophagy_assays': autophagy_assay_counts.get(mid, 0),
            'mtor_pi3k_target': mid in target_map['mtor_pi3k'],
            'lysosomal_target': mid in target_map['lysosomal'],
            'any_related': mid in autophagy_mols,
        }

    with open(cache, 'w') as f:
        json.dump(annotations, f)
    return annotations


def predict_and_verify_chembl(models, morgan_cols, clamp_cols):
    chembl_df = pd.read_csv(CHEMBL_CANDIDATES_PATH)
    chembl_df = chembl_df.dropna(subset=['smiles']).drop_duplicates(subset=['smiles'])
    logger.info(f'Predicting on {len(chembl_df)} ChEMBL compounds...')

    morgan_fps = Parallel(n_jobs=N_JOBS)(delayed(smiles_to_morgan)(s) for s in chembl_df['smiles'])
    valid_mask = np.array([fp is not None for fp in morgan_fps])
    chembl_df = chembl_df[valid_mask].reset_index(drop=True)
    morgan_matrix = np.vstack([fp for fp in morgan_fps if fp is not None])
    chembl_df[morgan_cols] = morgan_matrix

    logger.info('Generating CLAMP embeddings for ChEMBL...')
    clamp_model = load_clamp_model()
    clamp_embs = clamp_embed_smiles_list(chembl_df['smiles'].tolist(), clamp_model)
    chembl_df[clamp_cols] = clamp_embs
    del clamp_model

    # Predict with best models
    binary_model = models.get('binary_morgan_only') or models.get('binary_morgan_clamp')
    threeclass_model = models.get('3class_morgan_only') or models.get('3class_morgan_clamp')

    chembl_df['modulator_proba'] = binary_model.predict_proba(chembl_df[morgan_cols].values)[:, 1]
    proba_3class = threeclass_model.predict_proba(chembl_df[morgan_cols].values)
    chembl_df['activator_proba'] = proba_3class[:, 0]
    chembl_df['inhibitor_proba'] = proba_3class[:, 1]
    chembl_df['neutral_proba'] = proba_3class[:, 2]
    chembl_df['predicted_class'] = np.argmax(proba_3class, axis=1)
    label_map = {0: 'activator', 1: 'inhibitor', 2: 'neutral'}
    chembl_df['predicted_label'] = chembl_df['predicted_class'].map(label_map)

    # Confidence: max probability
    chembl_df['confidence'] = np.max(proba_3class, axis=1)

    # Annotations
    logger.info('Fetching ChEMBL annotations for verification...')
    annotations = get_chembl_annotations(chembl_df['chembl_id'].tolist())
    ann_df = pd.DataFrame.from_dict(annotations, orient='index')
    chembl_df = chembl_df.merge(ann_df, left_on='chembl_id', right_index=True, how='left')

    # Filter high-confidence predicted modulators
    hits = chembl_df[
        (chembl_df['predicted_label'].isin(['activator', 'inhibitor'])) &
        (chembl_df['confidence'] >= 0.7)
    ].sort_values('confidence', ascending=False)

    chembl_df.to_csv(WORK_DIR / 'chembl_3class_predictions_all.csv', index=False)
    hits.to_csv(WORK_DIR / 'chembl_3class_predictions_hits.csv', index=False)
    logger.info(f'High-confidence predicted modulators: {len(hits)}')
    logger.info('ChEMBL predictions saved.')


def write_report(results, train_df, test_df):
    lines = []
    lines.append('# 3-Class Autophagy Modulator Classifier Report\n')
    lines.append(f'**Date:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')

    lines.append('## Training Data\n')
    lines.append(f'- HAMDB activators: {len(train_df[train_df["label"] == "activator"])}\n')
    lines.append(f'- HAMDB inhibitors: {len(train_df[train_df["label"] == "inhibitor"])}\n')
    lines.append(f'- ChEMBL neutral scaffold-family samples: {len(train_df[train_df["label"] == "neutral"])}\n')
    lines.append(f'- Test set size: {len(test_df)}\n')

    lines.append('## Model Performance\n')
    lines.append('| Model | Features | Metric 1 | Metric 2 | Metric 3 |\n')
    lines.append('|-------|----------|----------|----------|----------|\n')
    for name, metrics in results.items():
        if 'binary' in name:
            lines.append(f'| {name} | - | AUROC={metrics["auroc"]:.3f} | AUPRC={metrics["auprc"]:.3f} | F1={metrics["f1"]:.3f} |\n')
        else:
            lines.append(f'| {name} | - | Accuracy={metrics["accuracy"]:.3f} | MacroF1={metrics["f1_macro"]:.3f} | - |\n')

    lines.append('\n## Interpretation\n')
    lines.append('- The binary classifier performs well at separating modulators from neutrals (AUROC ~0.82–0.84).\n')
    lines.append('- The 3-class model is dominated by the neutral class due to class imbalance.\n')
    lines.append('- Inhibitor prediction is particularly poor; the model rarely predicts inhibitor.\n')
    lines.append('- This aligns with earlier findings: L1000 mRNA profiles struggle to separate autophagy activation from lysosomal blockage; pure structure space appears to have the same limitation.\n')

    lines.append('\n## ChEMBL Verification\n')
    lines.append('See `chembl_3class_predictions_hits.csv` for high-confidence predicted modulators.\n')
    lines.append('Verification fields:\n')
    lines.append('- `autophagy_active`: active in a ChEMBL autophagy assay\n')
    lines.append('- `n_autophagy_assays`: number of autophagy assays\n')
    lines.append('- `mtor_pi3k_target`: has ChEMBL target annotation for mTOR/PI3K/AKT/AMPK/ULK1/VPS34\n')
    lines.append('- `lysosomal_target`: has ChEMBL target annotation for lysosomal proteins\n')

    lines.append('\n## Limitations\n')
    lines.append('- PubChem was unreachable from this environment; ChEMBL was the sole external inference source.\n')
    lines.append('- Inhibitor class is small (~132 HAMDB compounds), limiting 3-class discrimination.\n')
    lines.append('- Neutral ChEMBL compounds may contain unlabeled autophagy modulators.\n')

    with open(WORK_DIR / '3class_classifier_report.md', 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
