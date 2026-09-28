# Autophagy Modulator Prediction — Reproduction Repository

Code and data for the preprint:

> A. Alipour, *Structure-Based Prediction of Autophagy Modulators: From
> Transcriptome Collapse to Morgan Fingerprint Classifiers* (bioRxiv, 2026).

Archived at **https://doi.org/10.5281/zenodo.22986834** (concept DOI, always
resolves to the latest version). Study blog post:
https://rezaalipour.com/blog/what-broke-predicting-autophagy

This repository is a clean, self-contained copy of the study pipeline. It
contains every script, label set, trained model, result table, and report
needed to replicate all figures and numbers in the paper. Raw LINCS L1000
data (GSE92742, ~4 GB download) is the only external input; everything else
is either included or fetched by the scripts.

## Layout

| Path | Contents |
|---|---|
| `paper/` | Manuscript (`main.tex`, `main.pdf`), all figures (vector PDF + PNG), cached Fig 1 data, bootstrap-CI and feature-importance tables |
| `data/` | `hamdb_autophagy_directions.csv` (514 activators / 132 inhibitors / 167 neutral), `chembl_neutral_candidates.csv` (99,434 compounds), `autophagy_genes.txt` (199-gene set), `curated_phenotypes.csv`; see `data/README.md` for the L1000 download |
| `models/` | Trained weights: V6 transcriptome model (`nafs2_best_v6.pt`), its Uni-Mol embedding cache, both binary XGBoost classifiers, both 3-class classifiers, cascade direction model |
| `results/` | All result tables (`binary_models_comparison.csv`, `chembl_enrichment_morgan_vs_clamp.csv`, ROC-curve arrays, …) and `reports/` with the full audit trail (per-task reports, V6 post-mortem, signature teardown, blog corrections) |
| `*.py` (root) | The pipeline, staged below |

## Reproduction pipeline

Run from the repository root. All splits are seeded (42); the reconstructed
scaffold evaluation is additionally deterministic
(`PYTHONHASHSEED=0`).

**0. Environment**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# CLAMP (for embeddings): follow https://github.com/ml-jku/clamp —
# pip install from its repo, then place the package under src/clamp and the
# pretrained checkpoint under data/models/clamp_clip/checkpoint.pt
```

**1. Data curation**
```bash
python curate_hamdb_directions.py        # HAMdb labels -> data/hamdb_autophagy_directions.csv
python curate_autophagy_genes.py         # 199-gene pathway set -> data/autophagy_genes.txt
python fetch_chembl_pubchem_neutrals.py  # ChEMBL neutral library -> data/chembl_neutral_candidates.csv
# Download GSE92742 Level-5 files into data/l1000/ (see data/README.md)
```

**2. Transcriptome regression and collapse audit (paper §3.1, Fig. 1)**
```bash
python embed_unimol.py                   # Uni-Mol embeddings for L1000 compounds
python train_v6_contrastive.py           # train the V6 structure->12,328-gene model
python analyze_v6_results.py             # control audit: Rapamycin/Wortmannin Spearman
python verify_rapa_targets.py            # ground-truth target spot-checks
python diag_1_linear_baselines.py        # ridge upper bounds, genome-wide scan
python extract_fig1_data.py              # cache Fig 1 panels -> paper/fig1_data.npz
```

**3. Signature matching (paper §3.5)**
```bash
python extract_autophagy_signatures.py       # v1 mean-signature inverse search
python extract_autophagy_signatures_v2.py    # reviewed v2 redesign (contrastive, per-cell)
python diagnose_autophagy_signatures.py      # split-half / random-compound diagnostics
python flux_informed_signature_and_assay_integration.py  # 25-gene flux core + ChEMBL assays
python annotate_autophagy_signature_hits.py
```

**4. Binary modulator classification (paper §3.2, Fig. 2, Table 1)**
```bash
python generate_clamp_embeddings.py      # 768-d CLAMP embeddings
python hamdb_binary_classifier_xgb.py    # L1000-context models (227 vs 227, scaffold split)
python train_3class_autophagy_classifier.py  # structure-only binary + 3-class (646 vs 5,893)
python cascade_autophagy_classifier.py   # two-stage cascade + direction model
python direction_feature_ablation.py     # representation ablation for direction
python generate_roc_curves.py            # reconstructed scaffold split, ROC arrays
```

**5. External validation on ChEMBL (paper §3.3, Fig. 3, Table 2)**
```bash
python analyze_chembl_hits.py                    # threshold enrichment (Morgan-only model)
python compare_chembl_clamp_morgan_enrichment.py # head-to-head Morgan vs Morgan+CLAMP
```

**6. Statistics and figures for the paper**
```bash
PYTHONHASHSEED=0 python paper_bootstrap_ci.py    # retrain + 1000x bootstrap AUROC CIs
python paper_feature_importance.py               # top Morgan bits -> SMARTS + fold enrichment
python make_journal_figures.py                   # regenerates all 5 figures (PDF + PNG)
```

## Notes

- Evaluations use Murcko-scaffold-aware splits throughout
  (Bemis & Murcko 1996; MoleculeNet protocol). Random splits are not used
  for any headline number.
- The two binary XGBoost models are also provided pretrained in `models/`
  so steps 4–6 can be run without retraining.
- Large regenerable artifacts (raw L1000 matrices, multi-GB ChEMBL prediction
  dumps) are deliberately not version-controlled; the scripts above regenerate
  them.

## License

MIT (see `LICENSE`). Data resources retain their own licenses: HAMdb
(http://hamdb.scbdd.com), LINCS L1000 (GEO GSE92742), ChEMBL
(https://www.ebi.ac.uk/chembl/).
