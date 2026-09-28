# NAFS2 Diagnostic Report

**Date:** 2026-06-15  
**Agent:** OpenCode / NAFS2 V6 post-mortem diagnostics  
**Objective:** Determine why NAFS2 neural models (V3–V6) suffer from correlation collapse, and whether the problem is architectural or fundamental to the data.

---

## Executive Summary

Six iterations of the NAFS2 pipeline (V2–V6) failed to produce chemically differentiated transcriptomic predictions. The previous V6 plan proposed InfoNCE contrastive loss, compound-aware batching, and pathway-weighted MSE to fix collapse. This diagnostic suite was designed to test whether the collapse was caused by (a) weak chemical embeddings, (b) a flawed decoder/objective, or (c) an absence of signal in the data.

**Conclusion:** The signal is absent. Linear probes and Ridge baselines on the same 18,648 training signatures show that chemical structure (Uni-Mol or Morgan fingerprints) does not predict 6h L1000 gene-level logFC values. No neural architecture can fix an absent signal. The project must pivot to a different data source or target representation.

---

## Datasets and Experimental Setup

- **Primary data:** LINCS L1000 Phase I/II, GSE92742 Level 5 MODZ signatures (`GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx`)
- **Signatures used:** 18,648 batch-corrected signatures from `unimol_embeddings_l1000_10k.pt`
- **Controls excluded from training:** Rapamycin, Staurosporine, Chloroquine, Wortmannin, Tamoxifen, Metformin, and a random SMILES
- **Chemical representations tested:**
  - Uni-Mol embeddings (512-dim, precomputed, frozen)
  - Morgan fingerprints (2048-bit, radius 2)
- **Validation split:** Murcko scaffold split (80% train / 20% val)
- **Target:** Per-gene baseline-corrected logFC (profile − DMSO cell-type baseline)

---

## Diagnostic 0: Uni-Mol Embedding Quality Check

**Question:** Do the precomputed Uni-Mol embeddings encode chemical similarity in a way that correlates with transcriptomic similarity?

**Method:** Sample 200 compounds from the training set. Compute pairwise:
- Uni-Mol cosine distance
- Morgan Tanimoto distance
- Transcriptomic profile distance = 1 − Spearman(profile_i, profile_j)

Report Pearson and Spearman correlations between these three distance matrices.

**Results:**

| Correlation | Value |
|---|---|
| Pearson(Uni-Mol, Morgan) | **+0.3191** |
| Pearson(Uni-Mol, Profile) | **−0.0097** |
| Pearson(Morgan, Profile) | **+0.0073** |
| Spearman(Uni-Mol, Profile) | **−0.0045** |
| Spearman(Morgan, Profile) | **+0.0066** |

**Interpretation:** Uni-Mol embeddings correlate weakly with Morgan fingerprints (ρ ≈ 0.32) but have **no relationship** with transcriptomic profile similarity. Morgan fingerprints also have no relationship with profile similarity. The chemical representations do not capture the signal needed to predict 6h L1000 profiles.

---

## Diagnostic 1: Linear Baseline Upper Bounds

**Question:** What is the best predictive performance achievable with a simple linear model on the exact same data and targets used to train the NN?

**Method:** Train Ridge regression (α = 10) to predict the full 12,328-gene profile from several feature sets:
- Cell type only (one-hot)
- Morgan fingerprint
- Uni-Mol embedding
- Morgan + cell type
- Uni-Mol + cell type

Evaluate on scaffold-split validation. Report mean R² across genes and control-compound Spearman vs. ground truth.

**Results:**

| Feature set | Mean R² | Median R² | \|Spearman\| > 0.2 |
|---|---|---|---|
| Cell type only | **−0.0008** | −0.0022 | 15 / 12,328 |
| Morgan | **−0.1575** | −0.1489 | 0 / 12,328 |
| Uni-Mol | **−0.1105** | −0.1052 | 0 / 12,328 |
| Morgan + cell type | **−0.1583** | −0.1505 | 1 / 12,328 |
| Uni-Mol + cell type | **−0.1105** | −0.1061 | 0 / 12,328 |

**Key gene R² (Uni-Mol + cell type):**

| Gene | R² | Gene | R² |
|---|---|---|---|
| DDIT4 | −0.150 | FOXO3 | −0.112 |
| CASP3 | −0.106 | LAMP1 | −0.098 |
| SQSTM1 | −0.158 | CDKN2A | −0.063 |
| CDKN1A | −0.099 | CTSD | −0.109 |
| TFEB | −0.133 | XIST | +0.004 |

**Control compound Spearman (Uni-Mol + cell type):**

| Compound | Spearman(Pred, GT) |
|---|---|
| Rapamycin | +0.151 |
| Staurosporine | +0.641 |
| Chloroquine | −0.425 |
| Wortmannin | +0.008 |
| Tamoxifen | −0.009 |
| Metformin | −0.156 |

**Interpretation:** Chemistry predicts **worse than the mean** across almost all genes. The cell-type-only model performs best, but its signal is tiny and comes from residual cell-type differences after DMSO subtraction. The negative R² values for Morgan/Uni-Mol indicate that these features add noise, not signal.

---

## Diagnostic 2: Linear Probe on Key Genes

**Question:** Are specific biologically relevant genes predictable from chemical structure, even if the full profile is not?

**Method:** Train per-gene Ridge regressions for key autophagy/apoptosis/senescence genes using:
- Cell type only
- Dose only
- Cell type + dose
- Morgan
- Uni-Mol
- Morgan + cell type + dose
- Uni-Mol + cell type + dose

Also perform a genome-wide scan (12,328 genes) with Uni-Mol + cell + dose to estimate signal prevalence.

**Results (key genes, Uni-Mol + cell + dose):**

| Gene | R² | Spearman | Gene | R² | Spearman |
|---|---|---|---|---|---|
| DDIT4 | **−0.345** | +0.043 | FOXO3 | **−0.152** | +0.045 |
| CASP3 | **−0.097** | +0.011 | LAMP1 | **−0.098** | −0.010 |
| SQSTM1 | **−0.182** | +0.018 | CDKN2A | **−0.084** | +0.032 |
| CDKN1A | **−0.269** | +0.049 | CTSD | **−0.191** | +0.001 |
| TFEB | **−0.127** | +0.016 | BAX | **−0.029** | +0.111 |
| PARP1 | **−0.104** | +0.034 | ATG5 | **−0.120** | −0.024 |

**Genome-wide scan (12,328 genes, Uni-Mol + cell + dose):**
- Genes with |Spearman| > 0.2: **0 / 12,328**
- Genes with R² > 0: **~3 / 12,328**
- Genes with R² > 0.1: **0 / 12,328**

**Interpretation:** Even for the most biologically interpretable genes, chemical structure provides **negative predictive value**. The genome-wide scan confirms that this is not limited to key genes — essentially no gene in the 6h L1000 dataset is predictable from chemical structure at useful accuracy.

---

## Diagnostic 3: V6 Post-Training Audit (Existing Checkpoint)

**Question:** Did the V6 implementation actually achieve its stated goals?

**Method:** Load `nafs2_best_v6.pt` and run `analyze_v6_results.py`.

**Results:**

| Check | Target | Actual | Verdict |
|---|---|---|---|
| Spearman(Rapamycin, Wortmannin) | < 0.50 | **+0.9696** | ❌ Severe collapse |
| Spearman(Rapamycin, Mean A549) | < 0.50 | +0.2112 | ✅ Pass |
| DDIT4 prediction error | ≤ 1.0 | Predicted +0.276, GT −2.717, error 2.99 | ❌ Fail |
| FOXO3 prediction error | ≤ 0.5 | Predicted +0.144, GT +1.585, error 1.44 | ❌ Fail |
| Random Control mean |val| | < 0.10 | 0.0680 | ✅ Pass |

**Spearman(Pred, GT) for controls:**
- Rapamycin: +0.013
- Staurosporine: +0.045
- Chloroquine: +0.046
- Wortmannin: −0.002
- Tamoxifen: +0.022
- Metformin: −0.027

**Interpretation:** V6 failed. The model predicts a generic "active compound" profile with near-zero correlation to actual ground-truth profiles.

---

## Cross-Check: Ground Truth for Validation Targets

The V6 plan cited empirical targets of DDIT4 = −0.761 and FOXO3 = +1.502. I verified these directly from GSE92742.

**Actual Rapamycin/sirolimus A549 6h values (24 signatures):**

| Metric | DDIT4 | FOXO3 |
|---|---|---|
| Highest-dose signature, corrected | **−2.7165** | **+1.5849** |
| Mean corrected | −0.7909 | +2.7561 |
| Median corrected | −2.3234 | +2.7346 |
| Std | 3.2563 | 1.2650 |

**Finding:** The V6 targets are inconsistent. DDIT4 −0.761 matches the all-signature mean, while FOXO3 +1.502 matches a single raw signature. The mean is misleading because DDIT4 is bimodal across replicates. The validation targets were cherry-picked, not derived from a principled protocol.

---

## Synthesis: Root Cause

The collapse is not a modeling artifact. It is a **data-limitation** root cause:

1. **No chemical-transcriptomic correlation at 6h.** Uni-Mol and Morgan distances do not correlate with profile distances.
2. **Linear models fail before any neural model is trained.** Ridge with chemistry features achieves negative R².
3. **The signal is below the noise floor genome-wide.** 0 / 12,328 genes are predictable with |Spearman| > 0.2.
4. **Cell type dominates what little signal exists.** After DMSO subtraction, residual cell-type effects explain more variance than chemistry.
5. **Validation targets are noisy and inconsistent.** Even if a model could predict perfectly, the empirical targets vary wildly across replicates.

**Therefore:** InfoNCE, multi-task heads, pathway weighting, and compound-aware batching cannot succeed on this task as formulated. The input (chemical structure) and output (6h L1000 gene-level logFC) are effectively decoupled.

---

## Implications for V7

The V6 approach is obsolete. Any viable V7 must change one or more of the following:

| Component | V6 Approach | V7 Alternatives |
|---|---|---|
| **Time point** | 6h | **24h signatures** (autophagy flux is transcriptionally visible at 12–24h) |
| **Target space** | 12,328 genes | **Pathway scores**, mechanism classes, or a small gene panel |
| **Chemical representation** | Frozen Uni-Mol | Fine-tuned encoder, or multi-modal representation |
| **Data source** | L1000 only | DRUG-seq, sci-Plex 3, Chem-PerturBridge 24h data |
| **Task framing** | Regression to profile | **Classification** by mechanism or outcome |

**Recommended V7 path:** Use 24h L1000 data if available; otherwise pivot to mechanism-class prediction on 6h data. Do not attempt 12,328-gene regression from 6h L1000.

---

## Scripts and Artifacts

| File | Purpose |
|---|---|
| `diag_cache_data.py` | Caches training data for diagnostics |
| `diag_0_unimol_quality.py` | Uni-Mol / Morgan / profile distance correlation |
| `diag_1_linear_baselines.py` | Ridge baseline upper bounds |
| `diag_2_gene_probes.py` | Per-gene linear probes (parallelized with joblib) |
| `verify_rapa_targets.py` | Ground-truth verification for Rapamycin A549 6h |
| `diag_1_linear_baselines.log` | Output of Diagnostic 1 |
| `diag_2_gene_probes.log` | Output of Diagnostic 2 |

---

## Tests Run

1. **V6 post-training audit** (`analyze_v6_results.py`) — 2/5 checks passed.
2. **Ground-truth verification** (`verify_rapa_targets.py`) — verified DDIT4/FOXO3 values across 24 Rapamycin signatures.
3. **Embedding quality check** (`diag_0_unimol_quality.py`) — 200-compound pairwise distance analysis.
4. **Linear baseline upper bounds** (`diag_1_linear_baselines.py`) — 5 Ridge models on scaffold-split data.
5. **Linear gene probes** (`diag_2_gene_probes.py`) — per-gene Ridge probes on 18 key genes + genome-wide scan.

---

## Bottom Line

**Stop trying to predict 12,328 6h L1000 genes from chemical structure.** The data does not support it. V7 must use later time points, coarser targets, or different data.
