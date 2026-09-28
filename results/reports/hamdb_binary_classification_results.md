# XGBoost Binary Autophagy Modulator Classification Results

## Dataset
- **Task**: Binary classification — autophagy modulator (1) vs. non-modulator (0)
- **Positives**: 227 HAMDB compounds present in L1000 (matched by PubChem CID)
- **Negatives**: 227 random L1000 compounds not in HAMDB
- **Total compounds for split**: 454
- **Signatures**: 9,804 train / 3,856 test
- **Split**: 70/30 scaffold-aware (Murcko scaffold family); no compound appears in both splits
- **Mechanism overlap enforced**: mTOR and PI3K positives present in both train and test
- **Model**: `XGBClassifier` (no MLP)
- **Features**: CLAMP (768-d) + context **vs.** Morgan (2048-d) + context
- **Context**: Cell type one-hot + log10(dose) + time one-hot

## Results (`hamdb_binary_xgb_results.csv`)

| Feature Set | Signature AUROC | Signature AUPRC | Compound AUROC | Compound AUPRC | mTOR AUROC | PI3K AUROC | Activator AUROC | Rapamycin Proba | Chloroquine Proba |
|-------------|-----------------|-----------------|----------------|----------------|------------|------------|-----------------|-----------------|-------------------|
| CLAMP+context | 0.669 | 0.875 | 0.800 | 0.749 | 0.460 | 0.482 | 0.595 | 0.706 | 0.548 |
| **Morgan+context** | **0.765** | **0.914** | **0.845** | **0.810** | **0.629** | **0.657** | **0.897** | **0.981** | **0.992** |

## Interpretation

1. **Morgan+context clearly outperforms CLAMP+context** on every metric. For this task, the 2048-bit Morgan fingerprint carries more useful signal than the 768-dim CLAMP embedding.
2. **Held-out canonical controls are correctly classified** by Morgan+context:
   - Rapamycin (mTOR inhibitor): probability = 0.981
   - Chloroquine (lysosomal inhibitor): probability = 0.992
3. **Compound-level AUROC (0.845) is much better than signature-level AUROC (0.765)**, showing that averaging predictions across multiple cell/dose/time signatures per compound reduces noise.
4. **Activator discrimination is strong** (AUROC 0.897), while mTOR/PI3K discrimination is moderate (AUROC 0.63–0.66). This suggests the model is learning general autophagy-activation chemistry better than fine-grained kinase-family distinctions.
5. **Context matters.** Both runs include cell type, dose, and time; removing context would likely lower AUROC further.

## Why this succeeds where regression failed

| Aspect | Regression (199 genes) | Binary Classification |
|--------|------------------------|----------------------|
| Output | 199 continuous values | 1 binary label |
| Target signal | Weak, noisy, correlated genes | Clean HAMDB annotation |
| Held-out Rapamycin/Chloroquine | Failed (R² < 0) | Succeeded (proba > 0.98) |
| Best feature | CLAMP (weak) | Morgan+context (strong) |

## Files
- `hamdb_binary_classifier_xgb.py` — training and evaluation script
- `hamdb_binary_xgb_results.csv` — results table
- `comprehensive_walkthrough.md` — full project narrative including this experiment
