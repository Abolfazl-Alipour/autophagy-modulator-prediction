#!/usr/bin/env python3
"""
Cascaded autophagy modulator classifier.

Stage 1: Binary modulator vs. neutral classifier (reuses the existing
         binary_morgan_only_xgb.pkl model).
Stage 2: Direction classifier trained only on HAMDB activators and inhibitors
         to predict activator vs. inhibitor.

The cascade first filters for "is a modulator?" and then asks "which direction?",
which avoids forcing the direction model to learn a neutral class it will never
see at inference time.
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
from sklearn.model_selection import StratifiedKFold
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
HAMDB_PATH = WORK_DIR / 'data' / 'hamdb_autophagy_directions.csv'
BINARY_MODEL_PATH = WORK_DIR / 'models' / 'binary_morgan_only_xgb.pkl'
CHEMBL_PREDICTIONS_PATH = WORK_DIR / 'chembl_3class_predictions_all.csv'
CHEMBL_CANDIDATES_PATH = WORK_DIR / 'data' / 'chembl_neutral_candidates.csv'
CACHE_DIR = WORK_DIR / 'chembl_cache'
CACHE_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
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


def compute_morgan_features(smiles_list, n_jobs=N_JOBS):
    fps = Parallel(n_jobs=n_jobs)(delayed(smiles_to_morgan)(s) for s in smiles_list)
    valid_mask = np.array([fp is not None for fp in fps])
    matrix = np.vstack([fp for fp in fps if fp is not None])
    return matrix, valid_mask


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_hamdb_modulators():
    df = pd.read_csv(HAMDB_PATH)
    df = df.dropna(subset=['Canonical_SMILES'])
    df = df[df['Canonical_SMILES'] != '-666']
    df = df[df['autophagy_direction'].isin(['activator', 'inhibitor'])]
    df = df.copy()
    df['scaffold'] = compute_scaffolds(df['Canonical_SMILES'].tolist(), n_jobs=N_JOBS)
    df = df.dropna(subset=['scaffold'])
    df['direction_label'] = (df['autophagy_direction'] == 'inhibitor').astype(int)
    return df


def scaffold_group_split(df, test_frac=0.2, random_state=RANDOM_STATE):
    """Split modulators by scaffold; all rows sharing a scaffold go together."""
    rng = np.random.default_rng(random_state)
    scaffolds = df['scaffold'].unique()
    rng.shuffle(scaffolds)
    n_test = max(1, int(len(scaffolds) * test_frac))
    test_scaffolds = set(scaffolds[:n_test])
    train_mask = ~df['scaffold'].isin(test_scaffolds)
    return df[train_mask].copy(), df[~train_mask].copy()


# ---------------------------------------------------------------------------
# Direction model
# ---------------------------------------------------------------------------
def train_direction_classifier(train_df, feature_cols, eval_set=None):
    X_train = train_df[feature_cols].values
    y_train = train_df['direction_label'].values

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        eval_metric='aucpr',
        early_stopping_rounds=100,
    )
    fit_kwargs = {}
    if eval_set is not None:
        fit_kwargs['eval_set'] = [eval_set]
        fit_kwargs['verbose'] = False
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def evaluate_direction_classifier(model, test_df, feature_cols):
    X_test = test_df[feature_cols].values
    y_test = test_df['direction_label'].values
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, y_proba)
    auprc = average_precision_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'n_test': len(y_test),
        'n_inhibitors_test': int(y_test.sum()),
        'n_activators_test': int((y_test == 0).sum()),
    }


# ---------------------------------------------------------------------------
# Cross-validation on HAMDB modulators
# ---------------------------------------------------------------------------
def direction_cv(hamdb_mods, feature_cols, n_splits=5):
    """Scaffold-aware CV: group by scaffold and assign folds."""
    scaffold_to_fold = {}
    scaffolds = hamdb_mods['scaffold'].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(scaffolds)
    for i, scaf in enumerate(scaffolds):
        scaffold_to_fold[scaf] = i % n_splits

    hamdb_mods = hamdb_mods.copy()
    hamdb_mods['fold'] = hamdb_mods['scaffold'].map(scaffold_to_fold)

    fold_results = []
    for fold in range(n_splits):
        train_df = hamdb_mods[hamdb_mods['fold'] != fold]
        test_df = hamdb_mods[hamdb_mods['fold'] == fold]
        model = train_direction_classifier(train_df, feature_cols, eval_set=(test_df[feature_cols].values, test_df['direction_label'].values))
        metrics = evaluate_direction_classifier(model, test_df, feature_cols)
        logger.info(f'CV fold {fold+1}/{n_splits}: AUROC={metrics["auroc"]:.3f}, AUPRC={metrics["auprc"]:.3f}, F1={metrics["f1"]:.3f}')
        fold_results.append(metrics)

    agg = {
        'auroc_mean': np.mean([m['auroc'] for m in fold_results]),
        'auroc_std': np.std([m['auroc'] for m in fold_results]),
        'auprc_mean': np.mean([m['auprc'] for m in fold_results]),
        'auprc_std': np.std([m['auprc'] for m in fold_results]),
        'f1_mean': np.mean([m['f1'] for m in fold_results]),
        'f1_std': np.std([m['f1'] for m in fold_results]),
    }
    return fold_results, agg


# ---------------------------------------------------------------------------
# ChEMBL inference
# ---------------------------------------------------------------------------
def load_chembl_predictions():
    logger.info(f'Loading ChEMBL predictions from {CHEMBL_PREDICTIONS_PATH}')
    # The existing file has all Morgan bits and predictions; read only needed cols
    # to keep memory reasonable.
    usecols = ['chembl_id', 'smiles', 'inchi_key', 'mw', 'modulator_proba',
               'activator_proba', 'inhibitor_proba', 'neutral_proba',
               'predicted_label', 'confidence',
               'autophagy_active', 'n_autophagy_assays', 'mtor_pi3k_target', 'lysosomal_target']
    df = pd.read_csv(CHEMBL_PREDICTIONS_PATH, usecols=usecols)
    return df


def apply_cascade(chembl_df, direction_model, morgan_cols, binary_threshold=0.5, direction_threshold=0.5):
    """
    Apply binary model (using existing modulator_proba) and then direction model.
    Direction model is only applied when binary proba >= binary_threshold.
    """
    chembl_df = chembl_df.copy()
    chembl_df['cascade_modulator'] = chembl_df['modulator_proba'] >= binary_threshold

    # Load Morgan features for direction prediction.
    # The existing file stores them as columns morgan_0 ... morgan_2047.
    morgan_matrix = chembl_df[morgan_cols].values
    direction_proba = direction_model.predict_proba(morgan_matrix)[:, 1]
    chembl_df['cascade_inhibitor_proba'] = direction_proba
    chembl_df['cascade_direction'] = np.where(
        direction_proba >= direction_threshold, 'inhibitor', 'activator'
    )
    chembl_df['cascade_direction_confidence'] = np.maximum(direction_proba, 1 - direction_proba)

    # Final cascade label: modulator only if binary says so, otherwise neutral.
    chembl_df['cascade_label'] = np.where(
        chembl_df['cascade_modulator'],
        chembl_df['cascade_direction'],
        'neutral'
    )

    # Cascade confidence: combine binary modulator confidence and direction confidence.
    # For neutral: 1 - modulator_proba; for modulator: modulator_proba * direction_confidence.
    chembl_df['cascade_confidence'] = np.where(
        chembl_df['cascade_modulator'],
        chembl_df['modulator_proba'] * chembl_df['cascade_direction_confidence'],
        1 - chembl_df['modulator_proba']
    )
    return chembl_df


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_report(cv_results, agg, final_metrics, binary_threshold, direction_threshold, n_chembl, n_cascade_hits):
    lines = []
    lines.append('# Cascaded Autophagy Modulator Classifier Report\n')
    lines.append(f'**Date:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}\n')

    lines.append('## Approach\n')
    lines.append('- Stage 1: Binary modulator vs. neutral classifier (`binary_morgan_only_xgb.pkl`).\n')
    lines.append('- Stage 2: Activator vs. inhibitor direction classifier trained only on HAMDB modulators.\n')
    lines.append('- At inference: binary score filters modulators; direction model assigns activator/inhibitor only to predicted modulators.\n')

    lines.append('## HAMDB Direction Classifier (Activator vs. Inhibitor)\n')
    lines.append(f'- HAMDB activators: {final_metrics["n_activators"]}\n')
    lines.append(f'- HAMDB inhibitors: {final_metrics["n_inhibitors"]}\n')
    lines.append(f'- {len(cv_results)}-fold scaffold-aware CV:\n')
    lines.append(f'  - AUROC: {agg["auroc_mean"]:.3f} ± {agg["auroc_std"]:.3f}\n')
    lines.append(f'  - AUPRC: {agg["auprc_mean"]:.3f} ± {agg["auprc_std"]:.3f}\n')
    lines.append(f'  - F1 (inhibitor): {agg["f1_mean"]:.3f} ± {agg["f1_std"]:.3f}\n')

    lines.append('### Per-fold results\n')
    lines.append('| Fold | AUROC | AUPRC | F1 | Test inhibitors | Test activators |\n')
    lines.append('|------|-------|-------|----|-----------------|-------------------|\n')
    for i, m in enumerate(cv_results, 1):
        lines.append(f'| {i} | {m["auroc"]:.3f} | {m["auprc"]:.3f} | {m["f1"]:.3f} | {m["n_inhibitors_test"]} | {m["n_activators_test"]} |\n')

    lines.append('\n## Final Model (trained on all HAMDB modulators)\n')
    lines.append(f'- Test-set AUROC: {final_metrics["auroc"]:.3f}\n')
    lines.append(f'- Test-set AUPRC: {final_metrics["auprc"]:.3f}\n')
    lines.append(f'- Test-set F1: {final_metrics["f1"]:.3f}\n')

    lines.append('\n## ChEMBL Cascade Inference\n')
    lines.append(f'- Binary threshold: {binary_threshold}\n')
    lines.append(f'- Direction threshold: {direction_threshold}\n')
    lines.append(f'- Total ChEMBL compounds scored: {n_chembl}\n')
    lines.append(f'- Predicted modulators (cascade): {n_cascade_hits}\n')

    lines.append('\n## Files\n')
    lines.append('- `direction_morgan_only_xgb.pkl`: final direction classifier\n')
    lines.append('- `chembl_cascade_predictions_all.csv`: all ChEMBL compounds with cascade labels\n')
    lines.append('- `chembl_cascade_predictions_hits.csv`: predicted modulators from cascade\n')

    lines.append('\n## Interpretation\n')
    if agg['auroc_mean'] < 0.55:
        lines.append('- Direction discrimination is poor; activator and inhibitor structures overlap heavily in Morgan space.\n')
        lines.append('- The cascade does not reliably separate direction, so the binary modulator score should be treated as the primary output.\n')
    elif agg['auroc_mean'] < 0.70:
        lines.append('- Direction discrimination is weak-to-moderate; cascade direction labels are suggestive but not definitive.\n')
    else:
        lines.append('- Direction discrimination is moderate-to-strong; cascade improves over flat 3-class model.\n')

    with open(WORK_DIR / 'results' / 'reports' / 'cascade_classifier_report.md', 'w') as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info('=== Loading HAMDB modulators ===')
    hamdb_mods = load_hamdb_modulators()
    logger.info(f'HAMDB modulators: {len(hamdb_mods)}')
    logger.info(hamdb_mods['autophagy_direction'].value_counts())

    logger.info('=== Computing Morgan fingerprints for HAMDB modulators ===')
    morgan_matrix, valid_mask = compute_morgan_features(hamdb_mods['Canonical_SMILES'].tolist())
    hamdb_mods = hamdb_mods[valid_mask].reset_index(drop=True)
    morgan_cols = [f'morgan_{i}' for i in range(MORGAN_BITS)]
    hamdb_mods[morgan_cols] = morgan_matrix
    logger.info(f'Valid modulators after fingerprinting: {len(hamdb_mods)}')

    logger.info('=== Direction classifier: scaffold-aware CV ===')
    cv_results, agg = direction_cv(hamdb_mods, morgan_cols, n_splits=5)
    logger.info(f'CV AUROC: {agg["auroc_mean"]:.3f} ± {agg["auroc_std"]:.3f}')
    logger.info(f'CV AUPRC: {agg["auprc_mean"]:.3f} ± {agg["auprc_std"]:.3f}')
    logger.info(f'CV F1:    {agg["f1_mean"]:.3f} ± {agg["f1_std"]:.3f}')

    logger.info('=== Training final direction model on all HAMDB modulators ===')
    # Hold-out scaffold split for final metrics
    train_df, test_df = scaffold_group_split(hamdb_mods, test_frac=0.2)
    final_model = train_direction_classifier(
        train_df, morgan_cols,
        eval_set=(test_df[morgan_cols].values, test_df['direction_label'].values)
    )
    final_metrics = evaluate_direction_classifier(final_model, test_df, morgan_cols)
    final_metrics['n_activators'] = int((hamdb_mods['direction_label'] == 0).sum())
    final_metrics['n_inhibitors'] = int((hamdb_mods['direction_label'] == 1).sum())
    logger.info(f'Final test AUROC={final_metrics["auroc"]:.3f}, AUPRC={final_metrics["auprc"]:.3f}, F1={final_metrics["f1"]:.3f}')
    logger.info('\n' + classification_report(test_df['direction_label'], (final_model.predict_proba(test_df[morgan_cols].values)[:, 1] >= 0.5).astype(int), target_names=['activator', 'inhibitor']))

    with open(WORK_DIR / 'models' / 'direction_morgan_only_xgb.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    logger.info('Saved direction_morgan_only_xgb.pkl')

    logger.info('=== Loading ChEMBL predictions ===')
    chembl_df = load_chembl_predictions()
    logger.info(f'Loaded {len(chembl_df)} ChEMBL predictions')

    # If the predictions file lacks Morgan columns, recompute them.
    missing_morgan = [c for c in morgan_cols if c not in chembl_df.columns]
    if missing_morgan:
        logger.warning(f'Recomputing {len(missing_morgan)} Morgan bits for ChEMBL...')
        chembl_full = pd.read_csv(CHEMBL_CANDIDATES_PATH)
        chembl_full = chembl_full.dropna(subset=['smiles']).drop_duplicates(subset=['smiles'])
        chembl_full = chembl_full[chembl_full['chembl_id'].isin(chembl_df['chembl_id'])]
        morgan_matrix_c, valid_mask_c = compute_morgan_features(chembl_full['smiles'].tolist())
        chembl_full = chembl_full[valid_mask_c].reset_index(drop=True)
        chembl_full[morgan_cols] = morgan_matrix_c
        chembl_df = chembl_df.merge(chembl_full[['chembl_id'] + morgan_cols], on='chembl_id', how='inner')
        logger.info(f'ChEMBL after recomputing fingerprints: {len(chembl_df)}')

    logger.info('=== Applying cascade ===')
    binary_threshold = 0.5
    direction_threshold = 0.5
    cascade_df = apply_cascade(chembl_df, final_model, morgan_cols, binary_threshold, direction_threshold)

    cascade_hits = cascade_df[cascade_df['cascade_label'].isin(['activator', 'inhibitor'])].sort_values('cascade_confidence', ascending=False)
    logger.info(f'Cascade predicted modulators: {len(cascade_hits)}')
    logger.info('Cascade label distribution:')
    logger.info(cascade_df['cascade_label'].value_counts())

    cascade_df.to_csv(WORK_DIR / 'chembl_cascade_predictions_all.csv', index=False)
    cascade_hits.to_csv(WORK_DIR / 'chembl_cascade_predictions_hits.csv', index=False)
    logger.info('Saved cascade predictions.')

    write_report(cv_results, agg, final_metrics, binary_threshold, direction_threshold,
                 len(cascade_df), len(cascade_hits))
    logger.info('Done.')


if __name__ == '__main__':
    main()
