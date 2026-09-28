# Structure-Based Prediction of Autophagy Modulators: From Transcriptome Collapse to Morgan Fingerprint Classifiers

**A systematic benchmark of molecular representations and model architectures for autophagy modulator classification.**

---

## Abstract

We benchmarked structure-based and transcriptome-based approaches for predicting autophagy modulators using LINCS L1000 transcriptomic profiles and the Human Autophagy Modulator Database (HAMDB). A Uni-Mol neural network trained to predict gene-expression responses collapsed on biological controls: Rapamycin and Wortmannin, two compounds with opposing autophagy biology, were predicted with Spearman ρ = +0.97. Structure-based classifiers outperformed transcriptome regression by a wide margin. An XGBoost classifier using 2,048-bit Morgan fingerprints with L1000 context features achieved a compound-level AUROC of 0.845 on a scaffold-aware held-out test set and correctly ranked canonical controls. A structure-only Morgan + CLAMP model achieved the highest standalone binary AUROC (0.866) and showed 4× higher ChEMBL annotation enrichment than the Morgan-only model used in the original cascade. Activator-vs-inhibitor direction prediction remained near random (AUROC ~0.53), limited by only 132 curated inhibitors. Our results show that binary modulator classification is tractable from structure alone, but direction prediction and transcriptome regression are not yet reliable for autophagy drug discovery.

---

## 1. Introduction

Autophagy is a conserved cellular degradation pathway and an attractive therapeutic target in aging, neurodegeneration, and cancer. Small-molecule modulators of autophagy — both activators and inhibitors — have been curated in databases such as HAMDB, and large-scale perturbational transcriptomics (LINCS L1000) offer a potentially rich training signal for predictive models.

Several groups have attempted to learn structure-to-transcriptome mappings (DLEPS, DeepCE, CIGER, TranSiGen, MiTCP), but recent work has shown that deep-learning perturbation models rarely beat simple linear baselines and often predict near-zero fold-changes. We set out to determine whether this limitation also applies to autophagy modulator prediction, and whether simpler structure-based classifiers could provide a more reliable computational screen.

Our central question was: **Can we predict whether a small molecule modulates autophagy, and if so, in which direction, using either transcriptomic profiles or chemical structure?**

---

## 2. Methods

### 2.1 Data

- **HAMDB**: 841 curated autophagy modulators. After filtering to compounds with valid canonical SMILES and assignable direction, we retained 514 activators and 132 inhibitors.
- **LINCS L1000**: 471,660 Level5 COMPZ signatures across 20,315 compounds and 76 cell lines. We used 227 HAMDB modulators with matching L1000 signatures as positives for the context-aware binary classifiers.
- **ChEMBL**: ~99,433 small molecules used as an external inference set. For structure-only training we sampled ~5,893 scaffold-family-balanced neutral compounds from ChEMBL after excluding molecules with autophagy/mTOR/PI3K/lysosomal annotations.

### 2.2 Molecular Representations

- **Morgan fingerprints**: 2,048-bit circular fingerprints (RDKit, radius 2).
- **CLAMP embeddings**: 768-dimensional molecule–bioassay contrastive embeddings from the pretrained CLAMP model.
- **Uni-Mol embeddings**: 512-dimensional 3D-aware molecular embeddings used in the V6 transcriptome-prediction model.
- **Context features** (for L1000-context models): one-hot cell type, log10 dose, and one-hot time.

### 2.3 Model Architectures

We trained and compared:

1. **V6 Uni-Mol neural network**: predicts full 12,328-gene L1000 logFC vectors from Uni-Mol embeddings and cell type.
2. **L1000-context binary classifiers**: XGBoost and Ridge classifiers using CLAMP or Morgan fingerprints plus context features.
3. **Structure-only binary classifiers**: XGBoost using Morgan only or Morgan + CLAMP.
4. **3-class classifiers**: XGBoost multiclass models predicting activator / inhibitor / neutral.
5. **Cascaded classifier**: Stage 1 (binary modulator vs neutral) followed by Stage 2 (activator vs inhibitor direction).

All splits were scaffold-aware using Murcko scaffolds to avoid trivial train/test leakage by chemical similarity.

### 2.4 Evaluation

- **Binary classification**: AUROC, AUPRC, F1, accuracy; compound-level metrics by averaging signature probabilities.
- **Direction classification**: 5-fold scaffold-aware cross-validation AUROC, AUPRC, F1.
- **External validation**: ChEMBL annotation enrichment at modulator-score thresholds.

---

## 3. Results

### 3.1 Transcriptome Regression Collapses on Biological Controls

We trained a Uni-Mol-based neural network (V6) to predict 12,328-gene logFC responses. When we compared predictions for Rapamycin (mTOR inhibitor, induces autophagy) and Wortmannin (PI3K inhibitor, blocks autophagy), the model predicted nearly identical transcriptomic profiles: Spearman ρ = +0.97 across all genes (Figure 1). This demonstrates that the black-box transcriptome model failed to learn mechanistically distinct perturbation responses and instead collapsed toward a common output. The high correlation is not evidence of success; it is evidence of failure to separate opposing biology.

**Figure 1.** V6 model collapse: predicted logFC of Rapamycin versus Wortmannin across 12,328 genes (Spearman ρ = +0.97).

### 3.2 L1000-Context Binary Classifiers

We reformulated the problem as binary classification: modulator vs non-modulator, using L1000 signatures with context features. Results on the scaffold-aware held-out test set:

| Model | Architecture | Features | Compound AUROC | AUPRC | Held-out Rapamycin | Held-out Chloroquine |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Morgan+context | XGBoost | Morgan + context | **0.845** | 0.810 | 0.981 | 0.992 |
| CLAMP+context | XGBoost | CLAMP + context | 0.800 | 0.749 | 0.706 | 0.548 |
| Ridge+context (Morgan) | Ridge | Morgan + context | 0.823 | 0.765 | — | — |
| Ridge+context (CLAMP) | Ridge | CLAMP + context | 0.722 | 0.654 | — | — |

Morgan fingerprints outperformed the state-of-the-art CLAMP embedding when L1000 context was included. The canonical held-out controls (Rapamycin, Chloroquine) scored near the top with Morgan+context, confirming real biological signal.

**Figure 2.** ROC curves for L1000-context binary classifiers.

### 3.3 Structure-Only Binary Classifiers

Removing L1000 context entirely and training purely on chemical structure:

| Model | Features | Compound AUROC | AUPRC | Saved model |
| :---- | :---- | :---- | :---- | :---- |
| Morgan + CLAMP | Morgan + CLAMP | **0.866** | 0.629 | `binary_morgan_clamp_xgb.pkl` |
| Morgan only | Morgan | 0.858 | 0.642 | `binary_morgan_only_xgb.pkl` |

In the structure-only setting, adding CLAMP improved AUROC slightly (0.858 → 0.866). This is the opposite of the L1000-context result, where Morgan alone beat CLAMP. The likely explanation is that CLAMP provides useful bioassay-related signal when no cellular context is available, but is redundant once dose/time/cell-type context is included.

### 3.4 Direction Prediction Fails

We trained activator-vs-inhibitor classifiers using only HAMDB modulators. The best 5-fold scaffold-aware CV result was AUROC = 0.594 ± 0.060 (balanced Morgan sampling); the final cascade direction model achieved AUROC = 0.534 ± 0.045. Flat 3-class models achieved high accuracy (0.900) but low macro F1 (0.532 Morgan only, 0.455 Morgan + CLAMP), dominated by the large neutral class and poor inhibitor recall.

The ceiling is data: only 132 inhibitors with valid scaffolds are insufficient to learn reliable structure-based direction rules. Many inhibitors (e.g., lysosomotropic agents) and activators (e.g., mTOR inhibitors) share kinase-like scaffolds, making direction ambiguous from structure alone.

**Figure 4.** 3-class classifier performance: accuracy versus macro F1.

### 3.5 ChEMBL Enrichment: Morgan + CLAMP vs Morgan Only

We scored ~99,433 ChEMBL compounds with both structure-only binary models and measured annotation enrichment at probability thresholds. Morgan + CLAMP was dramatically more selective and more enriched:

| Threshold | Model | n | % ChEMBL | Autophagy-active % | mTOR/PI3K % |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 0.50 | Morgan + CLAMP | 2,791 | 2.8 | 1.61 | 3.08 |
| 0.50 | Morgan only | 21,897 | 22.0 | 0.40 | 0.51 |
| 0.80 | Morgan + CLAMP | 998 | 1.0 | 3.31 | 6.81 |
| 0.80 | Morgan only | 4,314 | 4.3 | 0.72 | 1.07 |
| 0.95 | Morgan + CLAMP | 343 | 0.34 | 6.41 | 13.99 |
| 0.95 | Morgan only | 310 | 0.31 | 1.61 | 3.23 |

At threshold 0.95, Morgan + CLAMP recovered 13.99% mTOR/PI3K-annotated compounds versus only 3.23% for Morgan only — a **4.3× improvement**. The top-200 hits of the two models overlapped by only 36 compounds, indicating that CLAMP shifted the prioritization toward a different but more biologically relevant chemical space.

**Figure 3.** ChEMBL annotation enrichment comparison: Morgan + CLAMP versus Morgan only.

### 3.6 Cascade Implementation

The published cascade used `binary_morgan_only_xgb.pkl` (Stage 1) and `direction_morgan_only_xgb.pkl` (Stage 2). It classified 21,897 ChEMBL compounds as modulators at threshold 0.5. Because Stage 1 used Morgan only rather than Morgan + CLAMP, the published enrichment numbers underestimate what the best structure-only model can achieve.

---

## 4. Discussion

Our results align with the broader computational perturbation biology literature: structure-to-transcriptome regression is weak, and simple structure-based classifiers often outperform learned embeddings. The key findings are:

1. **Transcriptome regression is not reliable for autophagy.** The V6 model collapsed on opposing controls, and signature-based inverse searches captured generic stress response rather than specific autophagy flux biology.
2. **Binary classification is tractable.** Morgan fingerprints + XGBoost achieve AUROC 0.845–0.866 on scaffold-aware held-out tests.
3. **Context matters.** With L1000 context, Morgan beats CLAMP. Without context, CLAMP helps slightly.
4. **Direction prediction is not solved.** Only 132 inhibitors limits any direction model; flat 3-class and cascade direction perform near random.
5. **ChEMBL enrichment is real but model-dependent.** Morgan + CLAMP gives substantially better enrichment than Morgan only.

The most useful practical output is the binary modulator score. Direction labels should not drive prioritization until more inhibitor training data are available.

---

## 5. Limitations

- Only 132 HAMDB inhibitors with valid scaffolds; direction models are underpowered.
- ChEMBL annotations are incomplete; many true autophagy modulators are unlabeled.
- L1000 mRNA profiles are an indirect proxy for autophagic flux and cannot separate activation from lysosomal blockage.
- No experimental validation of predicted ChEMBL hits.

---

## 6. Conclusion

For computational autophagy drug discovery, the best current approach is a structure-based binary modulator classifier using Morgan fingerprints, optionally augmented with CLAMP embeddings. Transcriptome regression and direction prediction remain unsolved. Future work should integrate direct flux readouts (e.g., LC3-II turnover, SQSTM1 degradation) as training labels rather than relying on mRNA proxies.

---

## Data and Code Availability

- HAMDB: http://hamdb.scbdd.com
- LINCS L1000: GSE92742
- Figures: `fig1_v6_correlation.png`, `fig2_binary_classifier_roc.png`, `fig3_chembl_enrichment.png`, `fig4_multiclass_comparison.png`, `linkedin_infographic.png`
- Tables: `binary_models_comparison.csv`, `chembl_enrichment_morgan_vs_clamp.csv`
- Scripts: `create_publication_figures.py`, `compare_chembl_clamp_morgan_enrichment.py`, `generate_roc_curves.py`

---

## References

- Ahlmann-Eltze et al. (2025). Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nature Methods* 22, 1657–1661.
- Chen et al. (2024). TranSiGen. *Nature Communications* 15, 5320.
- Adamczyk et al. (2025). Benchmarking Pretrained Molecular Embedding Models. *arXiv:2508.06199*.
- Li et al. (2025). MiTCP. *Briefings in Bioinformatics* 26(1).

---

## Appendix: Full Model Comparison Table

See `binary_models_comparison.csv` for the complete table including architecture, inputs, outputs, all metrics, model files, and notes for every binary, multiclass, and cascaded model trained in this project.
