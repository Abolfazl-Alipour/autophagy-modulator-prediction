# HAMDB Direction Curation and Autophagy Signature Inverse Search Results

## HAMDB Direction Curation

**Script:** `curate_hamdb_directions.py`  
**Output:** `hamdb_autophagy_directions.csv`

All 813 HAMDB compounds with SMILES were reviewed and classified by their effect on autophagy:

| Direction | Count | How determined |
|-----------|-------|----------------|
| **Activator** | 492 | Explicit HAMDB labels, target rules (mTOR/PI3K/HDAC/HSP90/MEK/BRAF/EGFR inhibitors, AMPK activators, proteasome inhibitors), and manual overrides for known inducers |
| **Inhibitor** | 132 | Explicit HAMDB labels, target rules (ULK1/VPS34/DUB/Aurora kinase/LRRK2 inhibitors, lysosomal disruptors, microtubule inhibitors), and manual overrides |
| **Neutral** | 189 | Direction unclear from available Category/Target/Description annotation |

Well-known overrides include:
- **Activator:** Rapamycin, Everolimus, Metformin, Lithium, Spermidine, Statins
- **Inhibitor:** Chloroquine, Hydroxychloroquine, Bafilomycin A1, Vinblastine, Vincristine

---

## 24h Activator and Inhibitor Transcriptomic Signatures

**Script:** `extract_autophagy_signatures.py`  
**Outputs:**
- `autophagy_activator_signature.npz`
- `autophagy_inhibitor_signature.npz`
- `autophagy_signature_differential_genes.csv`

**Dataset:** 118,088 strictly 24h L1000 signatures, restricted to the 199 autophagy genes.

| Class | Compounds in L1000 | 24h Signatures |
|-------|-------------------|----------------|
| Activators | 171 | 3,872 |
| Inhibitors | 34 | 784 |

### Key activator-up genes

| Gene | Mean Z-score | Interpretation |
|------|--------------|----------------|
| NPC1 | +0.378 | Lysosomal cholesterol export |
| ULK1 | +0.283 | Autophagy initiation kinase |
| SQSTM1 | +0.300 | Autophagy adaptor p62 |
| FOXO3 | +0.163 | Autophagy transcription factor |
| GABARAPL1 | +0.226 | ATG8 homolog |

### Key inhibitor-up genes

| Gene | Mean Z-score | Interpretation |
|------|--------------|----------------|
| DDIT4 | +0.362 | mTOR inhibition marker (likely compensatory) |
| BAX | +0.210 | Apoptosis marker |
| CFLAR | +0.220 | Stress response |

**Signature correlation:** Spearman(activator_mean, inhibitor_mean) = 0.63

---

## Inverse Search: Compounds Matching the Activator Signature

**Method:** Spearman correlation between every 24h L1000 signature and the mean activator signature. Compounds ranked by mean correlation across their 24h signatures.

**Outputs:**
- `activator_signature_hits.csv` (top 200 non-HAMDB hits)
- `autophagy_signature_hits_all_annotated.csv` (all 20,304 LINCS compounds ranked)
- `all_compounds_autophagy_signature_match.md` (human-readable summary with top 100 hits and threshold counts)

| Rank | Compound | Mean Spearman | Notes |
|------|----------|---------------|-------|
| 1 | BRD-K77681376 | 0.561 | Unannotated |
| 2 | BRD-K04076294 | 0.558 | Unannotated |
| 3 | ivermectin | 0.556 | Reported autophagy inducer |
| 4 | BRD-K02208014 | 0.555 | Unannotated |
| 5 | GSK-2126458 | 0.548 | Dual PI3K/mTOR inhibitor, known autophagy inducer |
| 6 | triptolide | 0.518 | ER stress / autophagy inducer |
| 7 | daunorubicin | 0.507 | DNA damage → autophagy |
| 8 | etoposide | 0.507 | DNA damage → autophagy |
| 9 | doxorubicin | 0.503 | DNA damage → autophagy |
| 10 | fluvastatin | 0.495 | Statin; known autophagy inducer |

---

## Inverse Search: Compounds Matching the Inhibitor Signature

**Outputs:**
- `inhibitor_signature_hits.csv` (top 200 non-HAMDB hits)
- `autophagy_signature_hits_all_annotated.csv` (all 20,304 LINCS compounds ranked)
- `all_compounds_autophagy_signature_match.md` (human-readable summary with top 100 hits and threshold counts)

| Rank | Compound | Mean Spearman | Notes |
|------|----------|---------------|-------|
| 1 | tributyltin | 0.525 | Lysosomal/mitochondrial toxicant |
| 2 | BRD-K62366931 | 0.515 | Unannotated |
| 3 | BRD-K51186655 | 0.513 | Unannotated |
| 4 | BRD-K37728465 | 0.496 | Also high on activator list |
| 5 | actinomycin-d | 0.454 | Transcription inhibitor; HAMDB-labeled activator |

The inhibitor search is noisier because lysosomal blockade triggers compensatory mTOR/autophagy gene transcription, making the inhibitor signature overlap with the activator signature.

---

## Interpretation

1. The **activator signature** is biologically coherent and recovers known autophagy inducers by inverse search.
2. The **inhibitor signature** is confounded by stress responses; lysosomal inhibitors transcriptionally resemble activators.
3. **Transcriptome alone cannot separate activation from flux blockage.** Orthogonal LC3/p62 flux assays are needed to validate hits.
4. This provides a candidate-generation workflow: curate direction labels → extract mean signature → rank all L1000 compounds by similarity.
