# Point-by-point response to outstanding audit questions

Manuscript: *Structure-Based Prediction of Autophagy Modulators: From Transcriptome Collapse to Morgan Fingerprint Classifiers* (A. Alipour)

All claims below were re-verified with code on 2026-10-14 against repository `Abolfazl-Alipour/autophagy-modulator-prediction` (commit `c2f2936`) and the archived working directory. Commands and file paths are given for each item so every answer can be re-checked independently.

---

## 1. What a cache row counts (manuscript §2.1.2)

**Question.** The feature cache holds 471,660 rows; GEO metadata for GSE92742 lists 205,034 small-molecule signatures. What are the 471,660 rows? Related: the 5,509 Murcko scaffold count and the exposure times (source: 6, 24, 48, 144 h).

**Answer.**

The cache contains **all GSE92742 Level-5 signatures except 21 without SMILES and 1,966 control-compound rows**: 473,647 − 21 − 1,966 = 471,660. It is not a small-molecule subset. Verified by re-running the count directly from `sig_info`/`pert_info`:

```
total sig_info rows: 473,647
  trt_cp 205,034 | trt_sh 154,993 | trt_sh.cgs 36,720 | trt_sh.css 24,368
  trt_oe 22,205 | ctl_vehicle 14,423 | trt_lig 8,256 | ctl_vector 6,826
  ctl_untrt 588 | + small .cns/.mut classes
rows with non-null smiles: 473,626  →  minus 17 excluded control SMILES (1,966 rows)  →  471,660 ✓
```

Composition of the 471,660 cache rows (computed from source metadata):

| pert_type | rows in cache |
|---|---|
| trt_cp (small molecule) | 203,068 |
| trt_sh + trt_sh.cgs + trt_sh.css | 216,081 |
| trt_oe + trt_oe.mut | 22,207 |
| trt_lig | 8,256 |
| ctl_vehicle / ctl_vector / ctl_untrt (+.cns) | 22,048 |

**Caveat now disclosed in the manuscript:** 256,054 of the 471,660 rows (54%) are genetic-perturbation and control signatures whose `canonical_smiles` is the placeholder `-666`; the cache builder (`v9_build_morgan_cache2.py:23-28`) converts unparseable SMILES to **all-zero 2,048-bit fingerprints** rather than dropping the row. The chemically meaningful content is 203,068 trt_cp signatures over **20,314 unique parseable structures** and **5,509 Murcko scaffolds** (the "20,315 compounds" figure counts the `-666` placeholder as one additional "compound").

The 5,509 scaffold count is real and now traceable: `results/logs/v9_morgan_build.log` (line: `Final: 471660 sigs, 20315 compounds, 5509 scaffolds`).

**Exposure times.** For trt_cp the source contains only 6 h (95,254), 24 h (109,721), 48 h (58), and 144 h (1). The 96/120/168 h values appearing in the cache time distribution come from shRNA/overexpression time courses, not small molecules. The manuscript now reports small-molecule exposure times of 6 and 24 h (with 59 additional signatures at 48/144 h).

**Suggested replacement sentence (§2.1.2).** "Our feature cache comprises 471,660 of the 473,647 GSE92742 Level-5 signatures (all perturbation types except 21 without SMILES and 1,966 control-compound rows); 203,068 are small-molecule (trt_cp) signatures spanning 20,314 unique structures and 5,509 Murcko scaffolds, predominantly at 6 and 24 h exposure."

---

## 2. Python environment that produced the final results (§2.9)

**Question.** `requirements.txt` is headed "Python 3.12" with minimum versions; the manuscript reports Python 3.13, RDKit 2026.03, XGBoost 3.4.1, PyTorch 2.12.1. Which is correct?

**Answer.** **The manuscript is correct; the repository header is wrong and should be fixed.** All final artifacts were produced in the miniconda environment:

| Package | Training env (miniconda, Python 3.13.11) | Paper venv (Python 3.12.3, created 2026-09-26) |
|---|---|---|
| PyTorch | **2.12.1 (CPU)** | 2.14.0+cpu |
| RDKit | 2026.3.2 | 2026.3.6 |
| XGBoost | **3.4.1** (stamped inside every saved pkl) | 3.4.1 |
| unimol-tools | 0.1.5 | not installed |

Evidence: `train_v6_contrastive.log` contains `/home/reza/miniconda3/lib/python3.13/site-packages/lightning/...` paths (V6 training ran under miniconda); `logs/unimol_tools_*.log` show miniconda site-packages loading `unimol_tools/weights/mol_pre_all_h220816.pt`; `json.loads(model.get_booster().save_config())['/version']` returns `[3, 4, 1]` for all four saved XGBoost pkls (verified live).

The Python 3.12 / PyTorch 2.14 venv was built on 2026-09-26, after all training, and only ran post-hoc analyses. It reproduced the bootstrap AUROC CIs byte-identically (`paper/bootstrap_auroc_ci.csv`), so the results are robust across both environments.

**Resolution.** One sentence in Methods covers it: results were produced with Python 3.13, PyTorch 2.12.1 (CPU), RDKit 2026.3, XGBoost 3.4.1; key analyses were reproduced under an independent environment (Python 3.12, PyTorch 2.14). `requirements.txt` line 1 should be annotated accordingly (done in commit `c2f2936` era; header fix pending — see Action items).

---

## 3. Provenance of the v2 conservative gene set (§2.5.2)

**Question.** The manuscript says the 70-gene set was drawn from GO:0006914 and the Human Autophagy Database; the repo has only a hard-coded list.

**Answer.** **The stated provenance is incorrect and must be corrected.** Verified by tracing `curate_autophagy_genes.py` and its log (`results/logs/curate_autophagy_genes.log`):

- KEGG pathway hsa04140 was fetched live: 169 genes.
- QuickGO queries for GO:0006914 (and 7 other GO terms) **returned 0 genes each** — GO contributed nothing to the final list.
- The 199-gene `data/autophagy_genes.txt` = KEGG hsa04140 (169) ∪ a **hand-curated "literature core" set of 135 genes** hard-coded in the script.
- The 70-gene STRICT set (`extract_autophagy_signatures_v2.py`, mirrored in `flux_informed_signature_and_assay_integration.py`) is a further manual subset: all 70 genes ∈ KEGG ∪ literature core (67/70 overlap the 199-gene file; the remaining 3, ATG4C/ATG4D/ATG9B, come from the hand-curated core).
- **HAMdb is not referenced anywhere in the gene curation chain**; it supplies compound labels only.

**Correction for the manuscript.** "a 70-gene conservative core (ATG machinery, LC3/GABARAP family, autophagy adaptors, lysosomal/TFEB, mTOR and AMPK readouts) curated from the KEGG autophagy pathway (hsa04140) and the autophagy literature." Remove GO:0006914 and HAMdb from this sentence.

---

## 4. Numbers resting on project records

### 4a. 5,893 ChEMBL negatives (§2.1.4)

**Now resolved — and the answer is nuanced.** The previously missing input (the exclusion set) is committed as `data/chembl_autophagy_exclusion_ids.json` (24,231 ChEMBL molecule IDs from live API queries, cached 2026-06-23). Re-running the full committed sampling procedure gives:

| Environment | RDKit | pandas | Sampled |
|---|---|---|---|
| Paper venv (Python 3.12) | 2026.3.6 | 3.0.6 | 5,881 |
| Miniconda (training env) | 2026.3.2 | 2.3.3 | 5,880 |
| Original run (2026-06-23) | — | — | **5,893** |

Root cause of the ±0.2% drift: `cluster_scaffolds` fed Butina with `list(set(scaffolds))`, whose order depends on the Python hash seed **and** on the scaffold strings themselves; RDKit 2026.3.2 vs 2026.3.6 canonicalize 6 scaffolds differently (verified by diffing all 99,197 scaffold strings across the two environments). Different input order changes cluster membership near the 0.7 Tanimoto cutoff, and the per-family cap of 5 compounds propagates the difference. Fixed on 2026-10-14 (`sorted(set(...))`); full evidence in `results/neutral_sampling_reproduction.md`.

**Manuscript guidance.** The trained models and every downstream metric correspond to the original 5,893-compound draw (metrics derive from saved models and committed result tables, so they are unaffected). Either write "5,893 (re-derivation from the archived code and data yields 5,880–5,881 owing to RDKit-version-dependent scaffold canonicalization; a fixed ordering has since been applied)" or "~5,900" in the main text.

### 4b. "Early stopping triggered at epoch 14" (§2.3.3)

**Confirmed.** The training log is now committed at `results/logs/train_v6_contrastive.log`:

```
Split: 17333 train samples, 1149 validation samples.        (95/5 scaffold split, seed 42)
Epoch 14 Done | Train Total Loss: 1.1611 | Train MSE: 0.2513 | Train Contra: 2.0677 | Val MSE: 0.6528
→ Early stopping triggered! No improvement in validation MSE for 10 epochs.
```

Patience (10) and the 95/5 Murcko scaffold split with excluded controls are in `train_v6_contrastive.py:100-104, 59-88`.

### 4c. Enrichment baselines 0.18% / 0.24% (§2.6)

**Verified and now committed.** Recomputed from the library-wide annotation table (99,433 ChEMBL compounds):

```
autophagy_active:   176 / 99,433 = 0.1770%
mtor_pi3k_target:   242 / 99,433 = 0.2434%
lysosomal_target:     2 / 99,433 = 0.0020%
```

Saved as `results/chembl_annotation_baselines.csv`. Enrichment factors check out: 6.41/0.177 = **36.2×**, 13.99/0.243 = **57.5×** (manuscript "~36× / ~58×"). The 0.18%/0.24% manuscript figures are correct roundings.

### 4d. Exact 227 context-aware negatives

**Now committed.** `data/context_binary_l1000_compounds.csv` contains the exact 227 positives + 227 negatives, reconstructed with the script's own procedure and seed (`RANDOM_STATE=42`): PubChem CID match → 230 unique SMILES → intersect with cache compounds → 227; negatives drawn from `rng.choice(sorted(pool), 227, replace=False)`. Verified to reproduce 227/227.

---

## 5. Repository gaps behind Methods claims — closed

### 5a. Loss equation (§2.3.2)

`loss.py` is now committed. The implemented objective is exactly:

```
L = mean_g( w_g · (p_g − t_g)² )  +  λ · InfoNCE(pred → target; τ = 0.1)
```

with per-gene weights w_g (14 landmark genes × 10.0; 26 inferred pathway genes × 2.0), InfoNCE over L2-normalized predicted-vs-target cosine similarity, same-compound pairs masked out of the softmax denominator, and λ annealed from 0.5 (`loss.py:27-58`, `train_v6_contrastive.py:32-40`).

**Important:** **there is no confidence weighting anywhere in the code.** If the manuscript equation contains a confidence term, it must be removed.

### 5b. V6 control compounds (§2.3.3)

The exclusion mechanism and the control list are now fully committed (`embedded_compounds_metadata_10k.csv`, 17 SMILES with `is_control=True`; consumed by `train_v6_contrastive.py:59-88` and `diag_cache_data.py`). The actual excluded controls are:

- rapamycin (sirolimus) × 6 SMILES
- wortmannin × 3
- tamoxifen × 3
- staurosporine × 2
- chloroquine × 1
- metformin × 1
- one random-control SMILES

**If the manuscript names nutlin-3, trichostatin A, withaferin A, or curcumin as excluded V6 controls, that is wrong** — those names occur only inside the HAMdb chemicals table, not in the training exclusion set.

Also note: `data/autophagy_genes.txt` (199 genes) **is** present in the repository; it may have been missed in the earlier audit.

---

## 6. PYTHONHASHSEED wording (§2.4.3)

**Name the variable explicitly.** Write: "split reconstruction was performed with `PYTHONHASHSEED=0` (exported by the project Makefile)." This is a standard reproducibility statement, not code-in-prose, and it is strictly more informative than "a fixed hash seed." Note the caveat discovered during this audit: hash-seed pinning alone does not make the neutral-sampling count bit-reproducible across RDKit versions (see 4a) — the scaffold-clustering input is now sorted, which removes the ordering dependence for future runs.

---

## 7. Kavox Labs affiliation (standing item)

**Remove it.** An affiliation was explicitly not requested for this manuscript, and an unconfirmed affiliation blocks submission. The author line should carry name (and contact/orcid if desired) only. This is a one-line change in the tex.

---

## Action items summary

| # | Item | Status |
|---|---|---|
| 1 | §2.1.2 rewrite with cache composition + 20,314 real compounds + 5,509 scaffolds + trt_cp exposure times | Manuscript edit needed |
| 2 | `requirements.txt` header: annotate training env (Python 3.13 / torch 2.12.1) vs paper venv | Repo edit needed |
| 3 | §2.5.2: replace "GO:0006914 and HAMdb" with "KEGG hsa04140 and the autophagy literature" | Manuscript edit needed |
| 4 | §2.1.4: 5,893 wording (accept 5,880–5,881 re-derivation note or "~5,900") | Manuscript edit needed |
| 5 | §2.3.2: delete any confidence-weighting term from the loss equation | Manuscript edit needed |
| 6 | §2.3.3: correct named controls to rapamycin/wortmannin/chloroquine/staurosporine/tamoxifen/metformin | Manuscript edit needed |
| 7 | §2.4.3: name `PYTHONHASHSEED=0` explicitly | Manuscript edit needed |
| 8 | Remove Kavox Labs affiliation | Manuscript edit needed (blocks submission) |
| 9 | All evidence files committed (logs, baselines, exclusion JSON, compound lists, loss/dataset/sampler sources, reproduction note) | **Done** (commits `a915334`, `c2f2936`) |
