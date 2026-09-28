# Corrections to the Original Blog Post

This document lists factual errors, omissions, and clarifications needed in the original project narrative.

## 1. "Morgan fingerprints beat CLAMP on every metric"

**Correction:** This is only true for the L1000-context models. In the structure-only setting, Morgan + CLAMP (AUROC 0.866) slightly outperformed Morgan only (AUROC 0.858). CLAMP helps when cellular context is absent; Morgan dominates when context is present.

## 2. The ChEMBL enrichment was attributed to the best model

**Correction:** The cascade enrichment numbers in the original post came from `binary_morgan_only_xgb.pkl`, not the best standalone model. The best structure-only model was `binary_morgan_clamp_xgb.pkl` (AUROC 0.866). When scored with Morgan + CLAMP, ChEMBL enrichment at threshold 0.95 improves from 1.61% to 6.41% autophagy-active and from 3.23% to 13.99% mTOR/PI3K — roughly 4× better.

## 3. The V6 collapse figure was shown as a hexbin density plot

**Correction:** The original infographic used hexbin, which obscures that each point is one gene. The updated figure uses individual scatter points (n = 12,328) with transparency to show the real data distribution.

## 4. AUROC values were displayed as a bar chart

**Correction:** AUROC should be shown as ROC curves with the actual FPR/TPR trace. The updated figure shows ROC curves for all four L1000-context models plus the random diagonal, with AUROC values in the legend.

## 5. "Direction prediction is a future problem"

**Clarification:** Direction prediction failed because there are only 132 HAMDB inhibitors with valid scaffolds. This is a hard data limitation, not just a modeling problem. The flat 3-class models also failed (macro F1 0.532 and 0.455), confirming that direction is genuinely underdetermined with current data.

## 6. Number of models trained

**Correction:** The post did not clearly enumerate all models. Six standalone binary models were trained (4 L1000-context, 2 structure-only), plus 2 flat 3-class models and 1 cascaded two-stage model. The full comparison is in `binary_models_comparison.csv`.

## 7. "2,000-compound training set had no biology"

**Clarification:** This referred to the initial smoke-test subset. Later analysis showed that 222 HAMDB compounds (26%) were already present in the full 20,315-compound L1000 cache. The problem was not zero mechanism coverage but that the signal was buried under mostly non-autophagy compounds.

## 8. CLAMP was not tested for direction prediction

**Correction:** The original direction ablation tested Morgan, descriptors, and balanced variants, but never CLAMP. The 3-class models included Morgan + CLAMP, but the dedicated cascade direction model was Morgan-only. A CLAMP direction model remains untested.
