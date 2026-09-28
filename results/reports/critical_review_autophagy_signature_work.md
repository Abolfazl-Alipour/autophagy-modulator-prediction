# Brutal Scientific Teardown: HAMDB Direction Curation and Autophagy Signature Inverse Search

**Date:** 2026-06-20  
**Review scope:** `curate_hamdb_directions.py`, `extract_autophagy_signatures.py`, `annotate_autophagy_signature_hits.py`, and all derived results.

---

## TL;DR

The inverse-search pipeline is **scientifically weak** as currently implemented. The mean "activator" signature is highly reproducible but captures a **generic stress response** shared by many random compounds, not a specific autophagy-activation program. The within-class signal is almost nonexistent (individual activator signatures correlate ~0.07 with their own mean), and the top hits are dominated by DNA-damaging agents and unannotated BRD compounds. The only redeeming feature is a ~15-fold enrichment of HAMDB activators in the top 50 hits, which suggests the approach has *some* signal but is far from reliable. The inhibitor signature is even less specific.

**Verdict:** The curated HAMDB labels and the binary Morgan classifier are salvageable. The transcriptomic inverse-search signature is not ready for biological prioritization without major redesign.

---

## 1. HAMDB Direction Curation: Directionally Useful, But Not a True Manual Review

### What we did
- 813 HAMDB compounds classified as activator (492), inhibitor (132), or neutral (189).
- Used explicit HAMDB labels, plain `Activator`/`Inhibitor` categories, target-based rules, and manual overrides.

### What's wrong
1. **Not a true manual literature review.** The user explicitly asked to "search one by one in the autophagy database HAMDB." We did rule-based inference plus overrides for famous compounds. Many labels are educated guesses, not validated annotations.
2. **Over-simplification of mixed targets.** Examples:
   - HDAC inhibitors are context-dependent: some induce autophagy, some block it.
   - PI3K inhibitors can activate or block autophagy depending on isoform and cell context.
   - DUB inhibitors are mechanistically heterogeneous; some stabilize autophagy proteins, others block flux.
   - Calcium channel blockers were left neutral despite literature reports that some (e.g., verapamil) modulate autophagy.
3. **Plain `Activator` / `Inhibitor` assumption.** We assumed these HAMDB categories mean autophagy activator/inhibitor. They almost certainly do, but this was not verified row-by-row.
4. **The neutral class is a black box.** 189 compounds are unclassified. Some are well-known autophagy modulators with poor annotation (e.g., common drugs, metabolites) that manual review could resolve.

### Bottom line
The curation is **directionally reasonable** as a first-pass scaffold, but it should not be treated as ground truth. Any downstream result depends on its quality, and misclassification noise is guaranteed.

---

## 2. The Mean Activator Signature: Stable But Not Specific

### What we did
- Extracted 3,872 24h signatures from 171 HAMDB-curated activators.
- Computed the mean 199-gene autophagy signature.

### Diagnostic results

| Test | Result | Interpretation |
|------|--------|----------------|
| Split-half reproducibility | Spearman = 0.971 ± 0.006 | The mean signature is extremely stable. |
| Individual signature vs. mean | Spearman = 0.071 ± 0.270 (median 0.064) | **Individual activator signatures barely resemble the mean.** |
| Random compounds vs. mean | Spearman = 0.713 ± 0.032 (max 0.785) | **Almost any random compound's mean signature correlates ~0.7 with the "activator" signature.** |
| HAMDB activator enrichment in top 50 | 12.0% vs. 0.8% baseline (~15×) | Weak but real enrichment. |
| HAMDB activator enrichment in top 500 | 4.2% vs. 0.8% baseline (~5×) | Enrichment decays rapidly. |

### What this means
The mean signature is a **stable average of many noisy profiles**, but the noise is so high that individual activator profiles are only weakly related to it. Worse, the signature matches random compounds almost as well as it matches activators, because it reflects a **generic stress/perturbation response** common to many L1000 compounds.

The 15× enrichment in the top 50 is the only encouraging sign, but even there, 88% of hits are not HAMDB activators. By top 500, the signal is barely above background.

### Why the signature is generic
- **LINCS L1000 is a perturbation atlas.** Most strong perturbagens induce a shared stress program: p53, DNA damage, unfolded protein response, cell-cycle arrest. This overlaps with autophagy gene expression because autophagy is a generic stress response.
- **199 autophagy genes include non-specific genes.** BAX, AKT1, HMGB1, HSPA8, DDIT4 are not exclusive to autophagy. They respond to many stressors.
- **Averaging across cell lines and doses blurs mechanism-specific signal.** The mean signature is dominated by high-prevalence cell lines (MCF7, PC3, A549) and broad transcriptional patterns.

---

## 3. The Mean Inhibitor Signature: Even Weaker

### Diagnostic results

| Test | Result |
|------|--------|
| Split-half reproducibility | Spearman = 0.706 ± 0.045 |
| Individual signature vs. mean | Spearman = 0.061 ± 0.196 (median 0.069) |
| Random compounds vs. mean | Spearman = 0.543 ± 0.047 (max 0.637) |
| Correlation with activator signature | Spearman = 0.63 |

### What this means
The inhibitor signature is **less stable and less specific** than the activator signature. Its moderate correlation (0.63) with the activator signature confirms that blocking autophagic flux triggers compensatory autophagy gene transcription, making the two classes transcriptomically similar.

The top inhibitor hits include actinomycin-D (a HAMDB activator), tributyltin (a generic toxicant), and many unannotated BRD compounds. This is not a reliable inhibitor-detection signature.

---

## 4. The Inverse Search Hits: Mostly Stress, Not Autophagy

### Top activator hits
- BRD-K77681376, BRD-K04076294, BRD-K02208014: unannotated.
- ivermectin: literature-supported autophagy inducer.
- GSK-2126458: dual PI3K/mTOR inhibitor, known autophagy inducer. **Good validation.**
- triptolide, daunorubicin, etoposide, doxorubicin: ER stress / DNA damage → generic autophagy.
- fluvastatin: statin, literature-supported.

### Problem
The list is a mix of:
1. **True autophagy inducers** (GSK-2126458, ivermectin, statins) — minority.
2. **Generic stressors** (DNA-damaging agents, ER stressors) — majority.
3. **Unannotated BRD compounds** — mechanism unknown.

Without orthogonal validation (LC3 turnover, p62 flux, target confirmation), these hits cannot be prioritized as autophagy modulators.

### Statistical issue
We report Spearman correlations but no p-values, false-discovery rates, or confidence intervals. A correlation of 0.50 sounds impressive but is common for random compounds against this generic signature.

---

## 5. Methodological Flaws

### Averaging across all contexts
- Cell type, dose, and time are ignored when computing the mean signature. MCF7 contributes 727 activator signatures; other cell lines contribute <50. The signature is effectively an MCF7/PC3/A549 average.
- A compound may induce autophagy in one cell type and apoptosis in another. Averaging destroys this information.

### No statistical controls
- No permutation test until the diagnostic script.
- No correction for multiple signatures per compound.
- No comparison to negative-control signatures (e.g., DMSO, random SMILES).

### Circular reasoning risk
- HAMDB compounds are used to define the signature, then HAMDB compounds are counted in the enrichment analysis. We excluded them from the published "top non-HAMDB hits" lists, but the signature itself is learned from them, so any test using the same data is circular.

### Gene set not specific
- The 199-gene set includes core autophagy machinery but also many genes that are generic stress markers. A more specific gene set (e.g., only LC3/GABARAP family, ATG genes, lysosomal biogenesis genes) might improve specificity.

---

## 6. What Is Actually Salvageable

1. **Curated HAMDB labels** are a useful starting point but need expert manual review, especially the 189 neutral compounds.
2. **The binary Morgan classifier** (Experiment 5) is more robust than the signature pipeline. It correctly classifies Rapamycin and Chloroquine and has compound-level AUROC 0.845. This should be the foundation for future work.
3. **The activator signature enrichment in top 50** suggests a weak signal exists. A much stricter, context-specific signature might work.

---

## 7. What Should Be Redone or Abandoned

### Abandon: broad mean-signature inverse search
The current 199-gene, all-context, mean-signature approach is too generic to be useful. It should not be used to prioritize compounds without major redesign.

### Redesign options
1. **Context-specific signatures.** Build separate signatures for A549 24h, MCF7 24h, etc. This removes cell-line averaging artifacts.
2. **Stricter gene set.** Use only core autophagy genes (ATG1/ULK complex, ATG8 family, PI3K-III complex, lysosomal biogenesis) and exclude generic stress genes like BAX, DDIT4, AKT1.
3. **Contrastive signature.** Define the activator signature as `(mean activator) − (mean neutral)` to remove generic perturbation background.
4. **Cross-validated signatures.** Build the signature on 70% of activators, test on held-out 30%. If held-out activators don't correlate with the signature, the signature is not capturing an activator-specific program.
5. **Use flux-aware labels.** The fundamental problem is that transcriptome ≠ autophagy flux. HAMDB annotations are based on flux/biology, not mRNA. Any signature approach is an imperfect proxy. Consider using direct assay data (e.g., PubChem autophagy assays) instead of transcriptome.
6. **Add orthogonal filters.** After inverse search, require hits to have known mTOR/AMPK/ULK1/lysosomal mechanism or positive evidence in ChEMBL/PubChem autophagy assays.

---

## 8. Conclusion

The signature inverse-search pipeline **does not produce a reliable list of autophagy modulators**. It produces a list of compounds whose 24h transcriptomes resemble a generic stress response that is common in L1000. The enrichment of known activators in the very top hits is real but weak, and the approach lacks the specificity needed for biological prioritization.

The most honest takeaway: **transcriptome similarity is not a good proxy for autophagy mechanism.** The binary Morgan classifier is a more credible path forward, and any signature-based approach must be redesigned with context-specific, contrastive, and cross-validated signatures.
