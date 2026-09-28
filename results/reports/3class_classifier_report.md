# 3-Class Autophagy Modulator Classifier Report
**Date:** 2026-06-23 03:50
## Training Data
- HAMDB activators: 514
- HAMDB inhibitors: 132
- HAMDB neutral: 167 (excluded from training)
- ChEMBL neutral scaffold-family samples: 5893
- Test set size: 1376
## Model Performance
| Model | Features | Metric 1 | Metric 2 | Metric 3 |
|-------|----------|----------|----------|----------|
| binary_morgan_only | - | AUROC=0.858 | AUPRC=0.642 | F1=0.595 |
| binary_morgan_clamp | - | AUROC=0.866 | AUPRC=0.629 | F1=0.422 |
| 3class_morgan_only | - | Accuracy=0.900 | MacroF1=0.532 | - |
| 3class_morgan_clamp | - | Accuracy=0.900 | MacroF1=0.455 | - |

## Interpretation
- The binary classifier performs well at separating modulators from neutrals (AUROC ~0.86).
- The 3-class model is dominated by the neutral class due to class imbalance.
- Inhibitor prediction is particularly poor; the model rarely predicts inhibitor.
- This aligns with earlier findings: L1000 mRNA profiles struggle to separate autophagy activation from lysosomal blockage; pure structure space appears to have the same limitation.

## ChEMBL Verification
See `chembl_3class_predictions_hits.csv` for high-confidence predicted modulators.
Verification fields:
- `autophagy_active`: active in a ChEMBL autophagy assay
- `n_autophagy_assays`: number of autophagy assays
- `mtor_pi3k_target`: has ChEMBL target annotation for mTOR/PI3K/AKT/AMPK/ULK1/VPS34
- `lysosomal_target`: has ChEMBL target annotation for lysosomal proteins

## Limitations
- PubChem was unreachable from this environment; ChEMBL was the sole external inference source.
- Inhibitor class is small (~132 HAMDB compounds), limiting 3-class discrimination.
- Neutral ChEMBL compounds may contain unlabeled autophagy modulators.
