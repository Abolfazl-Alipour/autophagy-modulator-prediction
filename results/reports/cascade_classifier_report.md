# Cascaded Autophagy Modulator Classifier Report
**Date:** 2026-06-23 06:37
## Approach
- Stage 1: Binary modulator vs. neutral classifier (`binary_morgan_only_xgb.pkl`).
- Stage 2: Activator vs. inhibitor direction classifier trained only on HAMDB modulators.
- At inference: binary score filters modulators; direction model assigns activator/inhibitor only to predicted modulators.
## HAMDB Direction Classifier (Activator vs. Inhibitor)
- HAMDB activators: 514
- HAMDB inhibitors: 132
- 5-fold scaffold-aware CV:
  - AUROC: 0.534 ± 0.045
  - AUPRC: 0.366 ± 0.058
  - F1 (inhibitor): 0.326 ± 0.048
### Per-fold results
| Fold | AUROC | AUPRC | F1 | Test inhibitors | Test activators |
|------|-------|-------|----|-----------------|-------------------|
| 1 | 0.598 | 0.447 | 0.393 | 25 | 96 |
| 2 | 0.499 | 0.379 | 0.298 | 30 | 105 |
| 3 | 0.526 | 0.291 | 0.294 | 20 | 109 |
| 4 | 0.478 | 0.310 | 0.270 | 36 | 108 |
| 5 | 0.571 | 0.403 | 0.372 | 21 | 96 |

## Final Model (trained on all HAMDB modulators)
- Test-set AUROC: 0.498
- Test-set AUPRC: 0.213
- Test-set F1: 0.231

## ChEMBL Cascade Inference
- Binary threshold: 0.5
- Direction threshold: 0.5
- Total ChEMBL compounds scored: 99433
- Predicted modulators (cascade): 21897

## Files
- `direction_morgan_only_xgb.pkl`: final direction classifier
- `chembl_cascade_predictions_all.csv`: all ChEMBL compounds with cascade labels
- `chembl_cascade_predictions_hits.csv`: predicted modulators from cascade

## Interpretation
- Direction discrimination is poor; activator and inhibitor structures overlap heavily in Morgan space.
- The cascade does not reliably separate direction, so the binary modulator score should be treated as the primary output.
