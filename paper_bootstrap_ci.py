#!/usr/bin/env python3
"""Bootstrap 95% CIs for the structure-only binary classifiers on the
reconstructed scaffold-aware split (same pipeline as generate_roc_curves.py).

Outputs: paper/bootstrap_auroc_ci.csv
"""
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src' / 'clamp'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_roc_curves as g

WORK_DIR = Path(__file__).resolve().parent
N_BOOTSTRAP = 1000
rng = np.random.default_rng(42)


def main():
    hamdb = g.load_hamdb()
    activators = hamdb[hamdb['autophagy_direction'] == 'activator'].copy()
    inhibitors = hamdb[hamdb['autophagy_direction'] == 'inhibitor'].copy()
    print(f'HAMDB: {len(activators)} activators, {len(inhibitors)} inhibitors')

    chembl_all = g.load_chembl_neutrals()
    chembl_neutral = g.sample_neutrals_by_scaffold_family(chembl_all, n_per_family=5)
    print(f'Sampled {len(chembl_neutral)} ChEMBL neutrals')

    records = []
    for df, label in [(activators, 'activator'), (inhibitors, 'inhibitor')]:
        for _, row in df.iterrows():
            records.append({'name': row.get('Chemcial_Name'), 'smiles': row['Canonical_SMILES'],
                            'scaffold': row['scaffold'], 'source': 'HAMDB', 'label': label, 'binary_label': 1})
    for _, row in chembl_neutral.iterrows():
        records.append({'name': row['chembl_id'], 'smiles': row['smiles'], 'scaffold': row['scaffold'],
                        'source': 'ChEMBL', 'label': 'neutral', 'binary_label': 0})
    train_df = pd.DataFrame(records)

    from joblib import Parallel, delayed
    morgan_fps = Parallel(n_jobs=12)(delayed(g.smiles_to_morgan)(s) for s in train_df['smiles'])
    valid = np.array([fp is not None for fp in morgan_fps])
    train_df = train_df[valid].reset_index(drop=True)
    morgan_matrix = np.vstack([fp for fp in morgan_fps if fp is not None])
    morgan_cols = [f'morgan_{i}' for i in range(2048)]
    train_df[morgan_cols] = morgan_matrix

    print('Computing CLAMP embeddings...')
    clamp_model = g.load_clamp_model()
    clamp_embs = g.clamp_embed_smiles_list(train_df['smiles'].tolist(), clamp_model)
    clamp_cols = [f'clamp_{i}' for i in range(clamp_embs.shape[1])]
    train_df[clamp_cols] = clamp_embs
    del clamp_model

    train_split, test_split = g.stratified_scaffold_split(train_df, test_frac=0.2)
    print(f'Train: {len(train_split)}, Test: {len(test_split)}')
    print(test_split['label'].value_counts())

    import xgboost as xgb
    from sklearn.metrics import roc_curve

    results = []
    roc_data = {}
    saved_pkl_aurocs = {}
    for feat_name, feat_cols in {'Morgan only': morgan_cols,
                                 'Morgan + CLAMP': morgan_cols + clamp_cols}.items():
        key = 'binary_morgan_only' if feat_name == 'Morgan only' else 'binary_morgan_clamp'
        X_train = train_split[feat_cols].values.astype(np.float32)
        y_train = train_split['binary_label'].values
        X_test = test_split[feat_cols].values.astype(np.float32)
        y_test = test_split['binary_label'].values

        # Retrained model (honest evaluation on this split's held-out scaffolds)
        clf = xgb.XGBClassifier(
            objective='binary:logistic', n_estimators=1000, max_depth=6,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42, n_jobs=12, eval_metric='logloss')
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        point = roc_auc_score(y_test, proba)
        fpr, tpr, thr = roc_curve(y_test, proba)
        roc_data[f'XGB {feat_name}'] = np.array([fpr, tpr, thr], dtype=object)

        # Saved-pickle model on the same test set (cross-split memorization check)
        with open(WORK_DIR / f'{key}_xgb.pkl', 'rb') as f:
            saved_model = pickle.load(f)
        saved_proba = saved_model.predict_proba(X_test)[:, 1]
        saved_pkl_aurocs[feat_name] = roc_auc_score(y_test, saved_proba)

        n = len(y_test)
        boot = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            if len(np.unique(y_test[idx])) < 2:
                boot[b] = np.nan
            else:
                boot[b] = roc_auc_score(y_test[idx], proba[idx])
        boot = boot[~np.isnan(boot)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f'{feat_name}: AUROC {point:.4f} 95% CI [{lo:.4f}, {hi:.4f}] (n_test={n}, '
              f'pos={int(y_test.sum())}, boot n={len(boot)})')
        results.append({'model': feat_name, 'auroc_point': round(point, 4),
                        'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4),
                        'n_test': n, 'n_positive': int(y_test.sum()),
                        'n_bootstrap': len(boot)})

    out = pd.DataFrame(results)
    out.to_csv(WORK_DIR / 'paper' / 'bootstrap_auroc_ci.csv', index=False)
    np.savez(WORK_DIR / 'paper' / 'reconstructed_xgb_roc.npz',
             **roc_data,
             aurocs=np.array([(f'XGB {m}', r) for m, r in
                              zip(out['model'], out['auroc_point'])], dtype=object))
    print(out.to_string(index=False))
    print('Saved-pkl models on this test set (cross-split check):',
          {k: round(v, 4) for k, v in saved_pkl_aurocs.items()})


if __name__ == '__main__':
    main()
