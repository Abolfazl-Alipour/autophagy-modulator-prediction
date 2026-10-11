# MANUSCRIPT_FACTCHECK.md

Fact-check of *"Structure-Based Prediction of Autophagy Modulators: From Transcriptome
Collapse to Morgan Fingerprint Classifiers"* against the reproduction repo
(`Abolfazl-Alipour/autophagy-modulator-prediction`, branch `main`).

Date: 2026-10-11. Two environments exist on the analysis machine; both are cited below:

- **paper venv** `/home/reza/projects/NAFS2/.paper_venv` — created 2026-09-26
  (Python 3.12.3, torch 2.14.0+cpu, rdkit 2026.3.6, xgboost 3.4.1).
- **miniconda** `/home/reza/miniconda3` — used for all June 2026 training/embeddings
  (Python 3.13.11, torch 2.12.1, rdkit 2026.3.2, unimol-tools 0.1.5; xgboost currently
  3.2.0 but model files are stamped 3.4.1).

Raw working directory (not public): `/home/reza/projects/NAFS2`.

---

## A. Environment & provenance

### Q1. Exact software versions

**Answer (claim-by-claim):**

| Manuscript claim | Verdict | Actual |
|---|---|---|
| Python 3.12 | **Partially wrong** | Python 3.12.3 in the paper venv (created 2026-09-26, after all training); **training and embeddings ran under Python 3.13.11 (miniconda)** |
| RDKit 2026.03 | Correct | 2026.3.6 (paper venv), 2026.3.2 (miniconda, used for fingerprints during training) |
| XGBoost 3.4.1 | Correct | Embedded in every saved model's config: `[3, 4, 1]`. (miniconda currently reports 3.2.0, but the pkls were saved by 3.4.1.) |
| PyTorch 2.14 (CPU) | **Wrong for training** | torch 2.14.0+cpu only in the Sep-26 paper venv. V6 training ran under **torch 2.12.1** (miniconda). CPU is correct. |
| CLAMP via official `PretrainedCLAMP` | Correct | `generate_clamp_embeddings.py:23-24` / `generate_roc_curves.py:108-112` |

**Evidence:**
- `.paper_venv/bin/pip freeze`: `rdkit==2026.3.6`, `torch==2.14.0+cpu`, `xgboost==3.4.1`, `numpy==2.5.3`; `python --version` → `Python 3.12.3`; `pyvenv.cfg` mtime 2026-09-26.
- `train_v6_contrastive.log` contains paths `/home/reza/miniconda3/lib/python3.13/site-packages/lightning/pytorch/...` → V6 trained in miniconda.
- `/home/reza/miniconda3/bin/pip list`: `torch 2.12.1`, `rdkit 2026.3.2`, `unimol-tools 0.1.5`, `python 3.13.11`.
- XGBoost version stamp (verified live):
  `json.loads(model.get_booster().save_config())['/version']` → `[3, 4, 1]` for
  `binary_morgan_only_xgb.pkl`, `binary_morgan_clamp_xgb.pkl`, `direction_morgan_only_xgb.pkl`, `3class_morgan_only_xgb.pkl`.

**Confidence:** verified (read literally from live environments and model files).

**Recommendation:** methods should say results were produced with Python 3.13 /
PyTorch 2.12.1 (CPU) / RDKit 2026.3.x / XGBoost 3.4.1, and (if mentioned at all) that
the reproducibility venv pins Python 3.12 / torch 2.14. As written, "Python 3.12" and
"PyTorch 2.14" describe the analysis venv, not the training environment.

### Q2. CLAMP checkpoint provenance

**Answer:** Checkpoint actually used: `data/models/clamp_clip/checkpoint.pt`
(667,072,057 bytes, md5 `1871d0e0ba7aafa8783be54e579f452e`, dated 2023-02-13 on disk)
plus `hp.json` (md5 `2d14d1b6111f1dab0240e4ca9f559cf5`). Download URL (baked into
`setup.sh:22-23`): `https://cloud.ml-jku.at/s/7nxgpAQrTr69Rp2/download/checkpoint.pt`
(JKU Nextcloud share, the official CLAMP authors' host; the CLAMP repo's own
`Pretrained` class downloads from the same host family). CLAMP source pinned at commit
`61575cb` ("Modify torch.load to include weights_only parameter") of
`https://github.com/ml-jku/clamp.git` (local `git log` in `NAFS2/src/clamp`).
No semantic version identifier exists for the checkpoint itself.

- `requirements.txt:25` still says `https://zenodo.org/record/XXXX` — **placeholder, wrong**; the real URL is in `setup.sh`.
- Interface call site confirmed: `generate_clamp_embeddings.py:23-24`
  `from clamp.models.pretrained import PretrainedCLAMP` / `model = PretrainedCLAMP(device='cpu')`;
  identical wrapper in `generate_roc_curves.py:108-112`. The class auto-downloads to
  `./data/models/pretrained/` if missing (`src/clamp/clamp/models/pretrained.py:13-33`).

**Confidence:** verified (file hashes, git log, script reads).

### Q3. Uni-Mol checkpoint

**Answer:** Embeddings were produced by `embed_unimol.py:17-21`
(`UniMolRepr(model_name='unimolv1', model_size='84m')`), installed package
**unimol-tools 0.1.5** (miniconda). The pretrained weights file is
`unimol_tools/weights/mol_pre_all_h_220816.pt` (per the tool's own log line). The
output embedding tensor `models/unimol_embeddings_l1000_10k.pt` is committed, so
the checkpoint is not needed for reproduction.

**Evidence:**
- `/home/reza/miniconda3/bin/pip list` → `unimol-tools 0.1.5`.
- `logs/unimol_tools_2026-06-14-23-13-04.log:1`: "Loading pretrained weights from
  /home/reza/miniconda3/lib/python3.13/site-packages/unimol_tools/weights/mol_pre_all_h_220816.pt".
- `embed_unimol.py:49` comment "In unimol-tools 0.1.5, get_repr might return a dict or a tuple."
- unimol-tools is not pinned in `requirements.txt` (only a comment at lines 28-29) — gap.

**Confidence:** verified.

---

## B. Data & labels

### Q4. HAMdb count chain

**Answer:**
- Raw `data/hamdb_chemicals.csv` (now copied into this repo): **841 rows**.
- After the SMILES filter in `curate_hamdb_directions.py` (only
  `dropna(subset=['Canonical_SMILES'])`, line 166 — no RDKit validity check):
  **813 rows**.
- Output `data/hamdb_autophagy_directions.csv`: **514 activators / 132 inhibitors / 167 neutrals** (total 813).

Commands run:
```
wc -l data/hamdb_chemicals.csv                                  # 1667 lines = 841 data rows + header (verified: pandas len = 841)
python -c "pandas len(raw)=841; raw['Canonical_SMILES'].notna().sum()=813"
python -c "pandas value_counts on data/hamdb_autophagy_directions.csv -> activator 514, inhibitor 132, neutral 167; len=813"
```

**Note:** the earlier draft phrase "813 curated compounds" refers to the *filtered* set;
"841" is the raw HAMdb download count. The manuscript's "841 → 514/132/167" chain is correct.
Provenance gap closed during this fact-check: raw `data/hamdb_chemicals.csv` was missing
from the repo and has been added (it is required by `curate_hamdb_directions.py:165`).

**Evidence:** `curate_hamdb_directions.py:165-187`; `data/hamdb_autophagy_directions.csv` (live count).

**Confidence:** verified.

### Q5. The 227 L1000-matched modulators

**Answer:** Matching is **PubChem CID only** — no name or SMILES fallback.
Script: `hamdb_binary_classifier_xgb.py:381-401`:
1. `hamdb.dropna(subset=['Canonical_SMILES'])` → 813 compounds, 813 numeric CIDs (log: "HAMDB compounds with SMILES: 813", "HAMDB with numeric PubChem CID: 813").
2. `pert_info['numeric_pcid'].isin(hamdb_pcids)` → then drop rows without SMILES, drop `canonical_smiles == '-666'`, dedupe by canonical SMILES → **230 unique SMILES**.
3. Intersect with the compounds present in the L1000 feature cache:
   `positive_smiles = positive_smiles & unique_cache_smiles` (line 429) → **227**.

Counts verified live by re-running the matching code (2026-10-11):
```
HAMDB compounds with SMILES: 813
HAMDB with numeric PubChem CID: 813
HAMDB positives in L1000: 230        (before cache intersect)
after cache intersect: 227           (smiles.npy member of diag_cache_v9_morgan_full.npz,
                                      471,660 signatures / 20,315 unique compounds)
```
The original training-time log `hamdb_autophagy_smoke_test.log` (NAFS2) confirms the chain:
"HAMDB compounds found in L1000: 230" → "Unique compounds: 227".
`results/reports/comprehensive_scientific_report.md` states "227 HAMDB modulators ...
(matched by PubChem CID)".

**Confidence:** verified (re-executed the exact matching code).

### Q6. 25-gene flux-contrastive core

**Answer:** The 25-gene list was **not saved in the repo** as a standalone gene list at
fact-check time. It is selected dynamically at runtime by
`flux_informed_signature_and_assay_integration.py:168-240`
(`select_flux_informative_genes`, `N_TOP_GENES = 25` at line 53), and the run's ranked
table was saved as `flux_informative_genes.csv` in the working directory only.

**Action taken:** copied to **`data/flux_informative_genes.csv`** in this repo
(25 rows, ranks 1-25: DDIT4, BNIP3, BNIP3L, BECN1, NPC1, WIPI1, CTSL, TSC2, ULK1,
ATG13, GABARAPL1, WIPI2, EIF4EBP1, GABARAPL2, MAP1LC3B, ATG101, LAMP1, CTSB, CTSD,
TSC1, SQSTM1, LAMP2, RB1CC1, ATG12, ATG3). The same table appears in
`results/reports/flux_signature_summary.md` ("Gene Selection").

**Confidence:** verified (file read; provenance gap closed).

---

## C. Number traceability

### Q7. V6 collapse audit (+0.970 / +0.434 / ±0.2 vs ±3)

**Answer:** Computed from `paper/fig1_data.npz` (cached by `extract_fig1_data.py:93-106`,
printed by `analyze_v6_results.py:139-140,174-188`). Live recompute:
```
model Spearman(Rapamycin, Wortmannin) = +0.9696   (manuscript: +0.970) ✓
GT     Spearman(Rapamycin, Wortmannin) = +0.4335  (manuscript: +0.434) ✓
pred range [-0.294, +0.293]                       (manuscript "±0.2 logFC": max |value| = 0.29 — claim is approximate)
GT   range [-6.812, +5.675]                       (manuscript "±3": actual span is ±3 to ±7 depending on side — "±3" understates the tails)
```
**Confidence:** verified (recomputed from cached vectors). Suggest manuscript say
"predictions span ±0.3" and "observed responses span roughly ±3 to ±7".

### Q8. Genome-wide diagnostics (−0.0097; 0/12,328; R² −0.11..−0.16)

**Answer:**
- Pearson(Uni-Mol distance, profile distance) = **−0.0097** — it is a *distance-matrix*
  correlation over 200 sampled compounds (`diag_0_unimol_quality.py:88,92`), recorded in
  `results/reports/NAFS2_DIAGNOSTIC_REPORT.md:46`. The manuscript phrase "between Uni-Mol
  embeddings and expression" is imprecise; the number itself is correct.
- Genes with |Spearman| > 0.2, structure-only ridge probes: **0 / 12,328** (Morgan),
  **0 / 12,328** (Uni-Mol), **0 / 12,328** (Uni-Mol+cell); Morgan+cell has 1/12,328;
  cell-type-only has 15/12,328. `diag_1_linear_baselines.log` (sections `--- morgan ---`,
  `--- unimol ---`, `--- unimol+cell_type ---`); same table in `NAFS2_DIAGNOSTIC_REPORT.md`.
- Ridge mean R²: Morgan **−0.1575**, Uni-Mol **−0.1105** → "≈ −0.11 to −0.16" ✓
  (`diag_1_linear_baselines.log`, `evaluate_ridge` at `diag_1_linear_baselines.py:35-67`).

**Confidence:** verified (log + report read literally).

### Q9. "61% Spearman improvement with 90× fewer compounds"

**Answer:** The +61% is verified; the "90×" is a different denominator than the experiment used.
- HAMDB-only training (`hamdb_autophagy_smoke_test.py`, 227 compounds / 11,298 signatures,
  seed split 9723/1575): MLP `mean_gene_spearman = 0.08245`
  (`hamdb_autophagy_smoke_test.log`, 2026-06-19 17:00).
- Full 20k-subset training: MLP `mean_gene_spearman = 0.05105`
  (execution log 2026-06-19 09:25; also `comprehensive_walkthrough.md:213` table "0.051 → 0.082, +61%").
- (0.08245 − 0.05105)/0.05105 = **+61.5%** ✓ (`hamdb_autophagy_results.md:24`).
- "90× fewer": 227 vs the **full L1000 library of 20,315 unique compounds** = 89.5×
  (inferred). But the +61% experiment compared 227 compounds against the **20k-signature
  subset (~2,000 unique compounds)** = only **8.8× fewer**. The sentence as written
  conflates the two denominators.

**Confidence:** verified for 0.051/0.082/+61%; inferred (flagged) for 90×.

**Provenance gaps closed:** `hamdb_autophagy_smoke_test.py` and raw
`data/hamdb_chemicals.csv` were missing from the repo; both added.

### Q10. Context-aware split sizes (9,804 / 3,856 / 137)

**Answer:** Test side verified: `results/binary_models_comparison.csv`
(`test_set_notes`: "Test: 137 compounds / 3856 signatures; Rapamycin/Chloroquine held out")
and `results/reports/hamdb_binary_classification_results.md` ("Signatures: 9,804 train / 3,856 test").
Train side (9,804) is **not captured in any log file**; the script prints it at
`hamdb_binary_classifier_xgb.py:482` (`logger.info(f"  Train signatures: ...")`) but the
training run's log was not preserved. Arithmetic is internally consistent:
227+227 = 454 compounds; 137 test → 317 train; 11,298 positive signatures
(smoke-test log) + 2,362 negative = 13,660 total; 13,660 − 3,856 = 9,804 ✓.

**Confidence:** test side verified; train side inferred (consistent arithmetic, no literal log).

### Q11. Cascade direction stage numbers

**Answer:** All verified from `results/reports/cascade_classifier_report.md` (generated by
`cascade_autophagy_classifier.py`) and `results/direction_feature_ablation.csv`
(generated by `direction_feature_ablation.py`; copied into repo during this fact-check):
- 5-fold scaffold CV: AUROC **0.534 ± 0.045**, AUPRC **0.366 ± 0.058**, F1 **0.326 ± 0.048** ✓
- Per-fold AUROC 0.478 / 0.499 / 0.526 / 0.571 / 0.598 → range **0.478–0.598** ✓
- Held-out test: AUROC **0.498** (also AUPRC 0.213, F1 0.231) ✓
- Class-balanced Morgan sampling: **0.594 ± 0.060** (`direction_feature_ablation.csv` row "Morgan balanced") ✓
- Physicochemical descriptors: **0.454 ± 0.082** (row "Descriptors only"; manuscript says 0.454) ✓

**Confidence:** verified.

### Q12. Signature-matching diagnostics

**Answer:** All verified:
- Activator split-half ρ = **0.971 ± 0.006** — `results/autophagy_signature_diagnostics.csv`
  (`activator_split_half_mean=0.97146, _std=0.00622`); quoted in
  `results/reports/critical_review_autophagy_signature_work.md:47`. (Inhibitor split-half is lower: 0.706 ± 0.045.)
- Compound-vs-consensus: **0.071 ± 0.270** (median 0.064) — CSV col `activator_self_corr_mean=0.07100`;
  review line 48.
- Random compounds vs activator mean: **0.713 ± 0.032** (max 0.785) — CSV `null_vs_activator_mean=0.71336`;
  review line 49. (The ±0.032 std is quoted in the review, not stored in the CSV.)
- Inhibitor/activator consensus correlation **0.63** — `results/reports/hamdb_signature_inverse_search_results.md:55`.
- Naive inverse search **~15× at top-50** (12.0% vs 0.8% baseline) — review line 50.
- Redesigned flux top-200: **2.5%** HAMDB activators — `results/reports/flux_signature_summary.md`
  ("Top 200: 2.5% HAMDB activators"); top-50 0.0%, top-100 1.0%.
- **5 of top 200** flux hits ChEMBL-autophagy-annotated — `results/flux_signature_hits_top200_chembl.csv`
  (`chembl_active` True×5 of 200; also 5 HAMDB activators in the same list).

**Confidence:** verified.

### Q13. "Two independent direction models agreed on only 55.9% of ChEMBL predictions"

**Answer:** The two models are the **flat 3-class classifier** (`predicted_label`) and the
**cascaded binary→direction model** (`cascade_label`). Script:
`analyze_chembl_hits.py:68-71`:
```
mod_subset = df[df['modulator_proba'] >= 0.5]
agreement = (mod_subset['predicted_label'] == mod_subset['cascade_label']).mean()
print(f'Agreement on modulator direction (binary>=0.5): {agreement:.3f}')
```
Reported as 55.9% in `comprehensive_walkthrough.md:626,724,865` ("3-class vs. cascade
agreement 55.9%"). Caveat: the agreement is computed **only over compounds flagged as
modulators (proba ≥ 0.5, 21,897 of 99,433)**, not all ChEMBL predictions — the manuscript
sentence should say so. Note also `analyze_chembl_hits.py` reads
`chembl_cascade_predictions_all.csv`, which is **not in the repo** (839 MB working file);
the literal 0.559 stdout was not preserved in any log.

**Confidence:** verified for the comparison identity and code path; the literal 0.559
value is attested by the walkthrough report (inferred from code + report, not re-run).

### Q14. "Top-200 hits of the two models overlap by only 36 compounds"

**Answer:** The two models are the structure-only **Morgan-only** and **Morgan + CLAMP**
binary scorers over ChEMBL. Script: `compare_chembl_clamp_morgan_enrichment.py:152-157`:
```
top_morgan = set(chembl_df.nlargest(200, 'modulator_proba_morgan')['chembl_id'])
top_clamp  = set(chembl_df.nlargest(200, 'modulator_proba_clamp')['chembl_id'])
logger.info(f'Overlap: {len(top_morgan & top_clamp)} / 200')
```
Value 36 quoted in `results/reports/comprehensive_scientific_report.md:113` ("overlapped by
only 36 compounds"). The literal stdout was not preserved in a log; input prediction CSV
is not in the repo. (Note: this "two models" pair — both binary modulator scorers — is a
*different pair* than Q13's direction-model pair; the manuscript uses "two models" in both
places and should name them explicitly.)

**Confidence:** verified for script and reported value (report attestation; not re-run).

### Q15. "1,155 of 2,048 bits carry nonzero gain"

**Answer:** **Verified by direct computation.** `paper_feature_importance.py:98-99`:
```
scores = model.get_booster().get_score(importance_type="gain")
print(f"\nfeatures with nonzero gain: {len(scores)} / {N_BITS}")   # N_BITS = 2048 (line 33)
```
Live run against `models/binary_morgan_only_xgb.pkl`:
`features with nonzero gain: 1155 / 2048` ✓

**Confidence:** verified.

### Q16. "Saved-pkl models inflated AUROC to 0.97 on the reconstructed test set"

**Answer:** The value is printed only to stdout by `paper_bootstrap_ci.py:91,118`
(`saved_pkl_aurocs[feat_name] = roc_auc_score(y_test, saved_proba)`;
`print('Saved-pkl models on this test set (cross-split check):', ...)`). It is **not
stored in any CSV, log, or report** in the repo. Re-run launched during this fact-check:
```
cd autophagy-modulator-prediction
PYTHONHASHSEED=0 .paper_venv/bin/python paper_bootstrap_ci.py
```
(requires CLAMP checkpoint at `data/models/pretrained/` and `src/clamp`; both symlinked
from the working dir for the run). **Result: re-run completed 2026-10-11 and reproduced
`paper/bootstrap_auroc_ci.csv` exactly (Morgan only 0.8769 [0.8431, 0.9087];
Morgan + CLAMP 0.8394 [0.7994, 0.8751]; n_test = 1,365), with printed saved-pkl values:**
```
Saved-pkl models on this test set (cross-split check): {'Morgan only': 0.974, 'Morgan + CLAMP': 0.9774}
```
So the manuscript's "inflated AUROC to 0.97" is confirmed; the precise values are
**0.974 (Morgan only)** and **0.9774 (Morgan + CLAMP)**.

For reference, the honest retrained numbers from `paper/bootstrap_auroc_ci.csv` (verified):
Morgan only 0.8769 [0.8431, 0.9087], Morgan + CLAMP 0.8394 [0.7994, 0.8751], n_test = 1,365.

**Confidence:** verified (re-run reproduced the CSV exactly and printed the values above).

---

## D. Methods precision

### Q17. V6 loss weights (14 landmark @ 10.0, 26 non-landmark @ 2.0)

**Answer:** **Verified.**
- `train_v6_contrastive.py:128`: `landmark_genes = ['CTSD','SQSTM1','CTSL','CDKN1A','CASP3','GAA','DDIT4','ATG5','IL1B','CDKN2A','CASP7','BAX','PARP1','CCL2']` → **14 genes**.
- `train_v6_contrastive.py:129-135`: `pathway_genes_all` = 40 genes; `inferred_genes = [g for g in pathway_genes_all if g not in landmark_genes]` → all 14 landmarks are inside the 40 → **26 inferred genes**.
- `train_v6_contrastive.py:142-143`: `landmark_weight=10.0`, `inferred_weight=2.0`. ✓

**Confidence:** verified (counted lists in code).

### Q18. V6 data scope (20k subset; 95/5 scaffold split; controls excluded)

**Answer:** **Verified.**
- `train_v6_contrastive.py:47`: `SUBSET_LIMIT = 20000`; log header "Subset Limit: 20000 | Epochs: 50".
- `train_v6_contrastive.py:100-104`: `np.random.seed(42)`, shuffle unique Murcko scaffolds,
  `split_idx = int(len(unique_scaffolds) * 0.95)` → **95/5 scaffold train/val split**
  (log: early stopping at epoch 14 on val MSE).
- Controls: `train_v6_contrastive.py:59-88` collects `meta_df[meta_df['is_control'] == True]`
  SMILES and scaffolds and passes `exclude_smiles=control_smiles_list,
  exclude_scaffolds=control_scaffolds` into the dataset. `NAFS2_DIAGNOSTIC_REPORT.md`
  lists the excluded controls: Rapamycin, Staurosporine, Chloroquine, Wortmannin,
  Tamoxifen, Metformin (+ a random SMILES). Held-out controls were evaluated separately
  (`analyze_v6_results.py`).

**Confidence:** verified.

### Q19. Dead 15% inner validation split in `hamdb_binary_classifier_xgb.py`

**Answer:** **Confirmed — the split does nothing.** `hamdb_binary_classifier_xgb.py:219-223`
builds `val_idx` / `train_inner_mask` (15% of train held out), but the fit call at
**lines 242-245** is:
```
model.fit(
    X_train[train_inner_mask], y_train[train_inner_mask],
    verbose=False
)
```
No `eval_set` argument → **no early stopping ever occurred**; the "validation" 15% was
simply discarded from training (all 1,000 trees always trained). Safe for the manuscript
to omit the split, but methods must not claim early stopping. (Contrast with
`train_v6_contrastive.py`, which does implement real early stopping on the 5% scaffold
val split — "Early stopping triggered! No improvement in validation MSE for 10 epochs",
`train_v6_contrastive.log`.)

**Confidence:** verified (code read).

### Q20. Contradictions in `results/reports/`

**Checked:** all 9 files — `3class_classifier_report.md`, `blog_post_corrections.md`,
`cascade_classifier_report.md`, `comprehensive_scientific_report.md`,
`critical_review_autophagy_signature_work.md`, `flux_signature_summary.md`,
`hamdb_binary_classification_results.md`, `hamdb_signature_inverse_search_results.md`,
`NAFS2_DIAGNOSTIC_REPORT.md` — plus `chembl_binary_threshold_enrichment.csv` and
`chembl_enrichment_morgan_vs_clamp.csv`.

**No file contradicts the current manuscript's headline numbers.** Notes:
1. `blog_post_corrections.md` documents that the *original blog post* (not the manuscript)
   had errors: Morgan-beats-CLAMP only holds with context; the blog's cascade enrichment
   came from the Morgan-only model, not the best (Morgan + CLAMP) model. The manuscript
   already incorporates these corrections (enrichment table matches
   `chembl_enrichment_morgan_vs_clamp.csv`: 343 cpds / 6.41% / 13.99% at 0.95).
2. `blog_post_corrections.md` §7 says "222 HAMDB compounds (26%) were already present in
   the full 20,315-compound L1000 cache" — reconciles with 230/227 here: 222 is the count
   after `generate_roc_curves.load_hamdb` drops rows without a valid Murcko scaffold.
3. `3class_classifier_report.md` adds F1 = 0.595 (Morgan binary) / 0.422 (+CLAMP) —
   not cited in the manuscript; no conflict.
4. `flux_signature_summary.md` confirms the honest framing that the redesigned flux
   signature underperforms the naive one in HAMDB recovery (top-50 0.0%, top-200 2.5%).

**Confidence:** verified (files read).

---

## Manuscript sentences that are wrong, imprecise, or unverifiable

1. **"Python 3.12" / "PyTorch 2.14 (CPU)"** (methods/availability): wrong for training.
   Training used Python 3.13.11 / torch 2.12.1 (CPU, miniconda); the 3.12/2.14 venv was
   built 2026-09-26 for post-hoc analyses. RDKit 2026.03 and XGBoost 3.4.1 are correct.
2. **"Pearson −0.0097 between Uni-Mol embeddings and expression"**: imprecise. It is
   Pearson(Uni-Mol cosine distance, transcriptomic profile distance) over 200 sampled
   compounds (a distance-matrix correlation). Number correct, description should be fixed.
3. **"90× fewer compounds"**: the +61% comparison was 227 vs ~2,000 compounds (8.8×);
   90× is only true against the full 20,315-compound L1000 library, which was not the
   training comparator. Reword or give both denominators.
4. **"Prediction range ±0.2 logFC vs observed ±3"**: literal ranges are [−0.29, +0.29]
   predicted vs [−6.8, +5.7] observed. Suggest "±0.3" and "spanning −6.8 to +5.7".
5. **"Two independent direction models agreed on only 55.9% of ChEMBL predictions"**:
   (a) agreement was computed only among compounds with modulator score ≥ 0.5
   (21,897/99,433); (b) name the models (flat 3-class vs cascade). Also Q14's "two models"
   is a different pair (Morgan-only vs Morgan+CLAMP binary scorers) — the manuscript
   should name both pairs explicitly to avoid confusion.
6. **"227 modulators had usable L1000 signatures"**: correct, but methods should state the
   three steps (PubChem CID match → 230; valid SMILES/'-666' filter and dedup; intersect
   with cached signatures → 227) and that matching was CID-only.
7. **requirements.txt CLAMP URL `https://zenodo.org/record/XXXX`**: placeholder; real URL
   is `https://cloud.ml.jku.at/s/7nxgpAQrTr69Rp2/download/checkpoint.pt` (in `setup.sh`).
8. **"841 curated modulators from HAMdb"**: 841 is the raw row count; 813 survive the
   SMILES filter and 813 is the curated total (514+132+167). Fine if phrased as
   "raw download", wrong if phrased as "curated".
9. **Q16 "inflated AUROC to 0.97"**: confirmed by re-run on 2026-10-11; exact values
   0.974 (Morgan only) and 0.9774 (Morgan + CLAMP). Cite these precise numbers.
10. **Train split 9,804 signatures** (context-aware model): no preserved log; attested
    only by `hamdb_binary_classification_results.md` and consistent arithmetic
    (13,660 − 3,856). Acceptable, but the original training log should be archived if found.
