#!/usr/bin/env python3
"""
Ablation study for activator-vs-inhibitor direction prediction.

Compares:
1. Morgan fingerprints only
2. RDKit 2D descriptors only
3. Morgan + 2D descriptors
4. Balanced subsampling of activators to match inhibitor count
5. XGBoost with scale_pos_weight vs. no weighting
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import xgboost as xgb
from joblib import Parallel, delayed

WORK_DIR = Path(__file__).resolve().parent
HAMDB_PATH = WORK_DIR / 'hamdb_autophagy_directions.csv'
RANDOM_STATE = 42
N_JOBS = 12

DESC_NAMES = [
    'MolWt', 'MolLogP', 'MolMR', 'TPSA',
    'NumHAcceptors', 'NumHDonors', 'NumRotatableBonds',
    'NumAromaticRings', 'NumAliphaticRings', 'NumSaturatedRings',
    'NumAromaticHeterocycles', 'NumAliphaticHeterocycles', 'NumSaturatedHeterocycles',
    'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'RingCount'
]


def get_murcko_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf) if scaf else None
    except Exception:
        return None


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
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


def smiles_to_desc(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        vals = []
        for name in DESC_NAMES:
            fn = getattr(Descriptors, name)
            v = fn(mol)
            vals.append(v)
        return np.array(vals, dtype=np.float32)
    except Exception:
        return None


def compute_features(df):
    morgan_fps = Parallel(n_jobs=N_JOBS)(delayed(smiles_to_morgan)(s) for s in df['Canonical_SMILES'])
    descs = Parallel(n_jobs=N_JOBS)(delayed(smiles_to_desc)(s) for s in df['Canonical_SMILES'])
    valid = np.array([m is not None and d is not None for m, d in zip(morgan_fps, descs)])
    df = df[valid].reset_index(drop=True)
    morgan_matrix = np.vstack([m for m in morgan_fps if m is not None])
    desc_matrix = np.vstack([d for d in descs if d is not None])
    morgan_cols = [f'morgan_{i}' for i in range(morgan_matrix.shape[1])]
    desc_cols = [f'desc_{name}' for name in DESC_NAMES]
    df[morgan_cols] = morgan_matrix
    df[desc_cols] = desc_matrix
    return df, morgan_cols, desc_cols


def load_data():
    df = pd.read_csv(HAMDB_PATH)
    df = df.dropna(subset=['Canonical_SMILES'])
    df = df[df['Canonical_SMILES'] != '-666']
    df = df[df['autophagy_direction'].isin(['activator', 'inhibitor'])].copy()
    df['scaffold'] = Parallel(n_jobs=N_JOBS)(delayed(get_murcko_scaffold)(s) for s in df['Canonical_SMILES'])
    df = df.dropna(subset=['scaffold'])
    df['direction_label'] = (df['autophagy_direction'] == 'inhibitor').astype(int)
    return df


def scaffold_cv(df, feature_cols, n_splits=5, balanced=False, scale_pos_weight=True):
    scaffolds = df['scaffold'].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(scaffolds)
    scaffold_to_fold = {s: i % n_splits for i, s in enumerate(scaffolds)}
    df = df.copy()
    df['fold'] = df['scaffold'].map(scaffold_to_fold)

    fold_metrics = []
    for fold in range(n_splits):
        train_df = df[df['fold'] != fold].copy()
        test_df = df[df['fold'] == fold].copy()

        if balanced:
            # Downsample activators to match inhibitor count
            n_inh = (train_df['direction_label'] == 1).sum()
            act_df = train_df[train_df['direction_label'] == 0].sample(n=n_inh, random_state=RANDOM_STATE)
            inh_df = train_df[train_df['direction_label'] == 1]
            train_df = pd.concat([act_df, inh_df], ignore_index=True)

        X_train = train_df[feature_cols].values
        y_train = train_df['direction_label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['direction_label'].values

        spw = (y_train == 0).sum() / (y_train == 1).sum() if scale_pos_weight else 1
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            eval_metric='aucpr',
            early_stopping_rounds=100,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        fold_metrics.append({
            'auroc': roc_auc_score(y_test, y_proba),
            'auprc': average_precision_score(y_test, y_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'n_test': len(y_test),
            'n_inh_test': int(y_test.sum()),
        })
    return fold_metrics


def summarize(name, fold_metrics):
    print(f'\n{name}:')
    print(f'  AUROC: {np.mean([m["auroc"] for m in fold_metrics]):.3f} ± {np.std([m["auroc"] for m in fold_metrics]):.3f}')
    print(f'  AUPRC: {np.mean([m["auprc"] for m in fold_metrics]):.3f} ± {np.std([m["auprc"] for m in fold_metrics]):.3f}')
    print(f'  F1:    {np.mean([m["f1"] for m in fold_metrics]):.3f} ± {np.std([m["f1"] for m in fold_metrics]):.3f}')
    print(f'  Precision: {np.mean([m["precision"] for m in fold_metrics]):.3f}')
    print(f'  Recall:    {np.mean([m["recall"] for m in fold_metrics]):.3f}')


def main():
    df = load_data()
    print(f'Loaded {len(df)} HAMDB modulators:')
    print(df['autophagy_direction'].value_counts())

    df, morgan_cols, desc_cols = compute_features(df)
    print(f'After feature computation: {len(df)}')

    configs = [
        ('Morgan only', morgan_cols, False, True),
        ('Descriptors only', desc_cols, False, True),
        ('Morgan + descriptors', morgan_cols + desc_cols, False, True),
        ('Morgan balanced', morgan_cols, True, False),
        ('Morgan + descriptors balanced', morgan_cols + desc_cols, True, False),
    ]

    results = []
    for name, cols, balanced, spw in configs:
        fold_metrics = scaffold_cv(df, cols, n_splits=5, balanced=balanced, scale_pos_weight=spw)
        summarize(name, fold_metrics)
        results.append({
            'name': name,
            'auroc_mean': np.mean([m['auroc'] for m in fold_metrics]),
            'auroc_std': np.std([m['auroc'] for m in fold_metrics]),
            'auprc_mean': np.mean([m['auprc'] for m in fold_metrics]),
            'auprc_std': np.std([m['auprc'] for m in fold_metrics]),
            'f1_mean': np.mean([m['f1'] for m in fold_metrics]),
            'f1_std': np.std([m['f1'] for m in fold_metrics]),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(WORK_DIR / 'direction_feature_ablation.csv', index=False)
    print('\nSaved direction_feature_ablation.csv')
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
