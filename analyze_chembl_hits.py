#!/usr/bin/env python3
"""
Analyze ChEMBL predictions from the autophagy modulator classifiers.

Produces:
- Precision-recall enrichment statistics at binary modulator score thresholds
- Comparison of flat 3-class vs. cascaded predictions
- Top candidate lists with ChEMBL annotations
"""

import numpy as np
import pandas as pd
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(WORK_DIR / 'chembl_cascade_predictions_all.csv')
    n_total = len(df)
    print(f'Total ChEMBL compounds scored: {n_total}')

    # Any ChEMBL annotation flags
    df['any_annotation'] = (
        df['autophagy_active'].fillna(False).astype(bool) |
        df['mtor_pi3k_target'].fillna(False).astype(bool) |
        df['lysosomal_target'].fillna(False).astype(bool)
    )

    print('\n=== Overall annotation rates ===')
    for col in ['autophagy_active', 'mtor_pi3k_target', 'lysosomal_target', 'any_annotation']:
        rate = df[col].fillna(False).astype(bool).mean() * 100
        count = df[col].fillna(False).astype(bool).sum()
        print(f'  {col}: {count} / {n_total} ({rate:.2f}%)')

    print('\n=== Binary modulator score threshold analysis ===')
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rows = []
    for t in thresholds:
        subset = df[df['modulator_proba'] >= t]
        if len(subset) == 0:
            continue
        row = {
            'threshold': t,
            'n_compounds': len(subset),
            'pct_of_total': 100 * len(subset) / n_total,
            'autophagy_active_pct': 100 * subset['autophagy_active'].fillna(False).astype(bool).mean(),
            'mtor_pi3k_pct': 100 * subset['mtor_pi3k_target'].fillna(False).astype(bool).mean(),
            'lysosomal_pct': 100 * subset['lysosomal_target'].fillna(False).astype(bool).mean(),
            'any_annotation_pct': 100 * subset['any_annotation'].mean(),
        }
        rows.append(row)
    stats = pd.DataFrame(rows)
    print(stats.to_string(index=False))

    print('\n=== Cascade high-confidence modulators (binary >= 0.7) ===')
    cascade_high = df[df['modulator_proba'] >= 0.7]
    print(f'Count: {len(cascade_high)}')
    print('Direction distribution:')
    print(cascade_high['cascade_label'].value_counts())
    print('Flat 3-class distribution:')
    print(cascade_high['predicted_label'].value_counts())
    print('Annotation rates:')
    for col in ['autophagy_active', 'mtor_pi3k_target', 'lysosomal_target', 'any_annotation']:
        rate = cascade_high[col].fillna(False).astype(bool).mean() * 100
        print(f'  {col}: {rate:.2f}%')

    print('\n=== Agreement between flat 3-class and cascade direction ===')
    mod_subset = df[df['modulator_proba'] >= 0.5]
    agreement = (mod_subset['predicted_label'] == mod_subset['cascade_label']).mean()
    print(f'Agreement on modulator direction (binary>=0.5): {agreement:.3f}')

    print('\n=== Top 20 binary modulator hits ===')
    top_binary = df.nlargest(20, 'modulator_proba')
    cols_show = ['chembl_id', 'smiles', 'modulator_proba', 'predicted_label', 'cascade_label',
                 'cascade_inhibitor_proba', 'autophagy_active', 'mtor_pi3k_target', 'lysosomal_target']
    print(top_binary[cols_show].to_string(index=False))

    print('\n=== Known autophagy-active compounds in high-confidence hits ===')
    known_active = cascade_high[cascade_high['autophagy_active'].fillna(False)].sort_values('modulator_proba', ascending=False)
    print(f'Count: {len(known_active)}')
    if len(known_active) > 0:
        print(known_active[['chembl_id', 'smiles', 'modulator_proba', 'predicted_label', 'cascade_label']].head(20).to_string(index=False))

    print('\n=== Known mTOR/PI3K targets in high-confidence hits ===')
    known_mtor = cascade_high[cascade_high['mtor_pi3k_target'].fillna(False)].sort_values('modulator_proba', ascending=False)
    print(f'Count: {len(known_mtor)}')
    if len(known_mtor) > 0:
        print(known_mtor[['chembl_id', 'smiles', 'modulator_proba', 'predicted_label', 'cascade_label']].head(20).to_string(index=False))

    # Save enrichment table
    stats.to_csv(WORK_DIR / 'results' / 'chembl_binary_threshold_enrichment.csv', index=False)
    print('\nSaved chembl_binary_threshold_enrichment.csv')


if __name__ == '__main__':
    main()
