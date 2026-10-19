# Autophagy Modulator Prediction — Reproduction Repository

Code and data for the preprint:

> A. Alipour, *Structure-Based Prediction of Autophagy Modulators: From
> Transcriptome Collapse to Morgan Fingerprint Classifiers* (bioRxiv, 2026).

Archived at **https://doi.org/10.5281/zenodo.22986834** (concept DOI, always
resolves to the latest version).

This repository contains every script, label set, trained model, and result
table needed to replicate all figures and numbers in the paper. Raw LINCS L1000
data (GSE92742, ~4 GB download) is the only external input; everything else is
either included or fetched by the scripts.

## Layout

| Path | Contents |
|---|---|
| `pipeline/` | All analysis code (31 scripts). The `Makefile` calls them in the correct order; nothing here needs to be run by hand. |
| `paper/` | Manuscript (`main.tex`, `main.pdf`), all figures (vector PDF + PNG), cached Fig 1 data, bootstrap-CI and feature-importance tables |
| `data/` | `hamdb_chemicals.csv` + `hamdb_autophagy_directions.csv` (514 activators / 132 inhibitors / 167 neutral), `chembl_neutral_candidates.csv` (99,434 compounds), `chembl_autophagy_exclusion_ids.json`, `context_binary_l1000_compounds.csv`, `autophagy_genes.txt`, `flux_informative_genes.csv`, `curated_phenotypes.csv`; see `data/README.md` for the L1000 download |
| `models/` | Trained weights: V6 transcriptome model (`nafs2_best_v6.pt`), its Uni-Mol embedding cache, both binary XGBoost classifiers, both 3-class classifiers, cascade direction model |
| `results/` | All result tables, ROC-curve arrays, `logs/` (trimmed training/build logs), and `reports/` (audit trail) |
| `docs/` | `MANUSCRIPT_FACTCHECK.md` (Q1–Q20 claim verification) and `REVIEW_RESPONSES.md` (point-by-point audit responses) |

## Quick start

```bash
./setup.sh     # venv, deps, CLAMP checkpoint download, input checks (one time)
make full      # every manuscript statistic + all 5 figures, from included caches
```

`make full` takes minutes and needs no L1000 download. `make paper` recompiles
the manuscript (tectonic or TeX Live).

## Full pipeline from raw data

Run from the repository root, in this order (or stage by stage:
`make data`, `make embeddings`, `make train`, `make classify`, `make analyze`,
then `make stats figures`). All splits are seeded (42); the reconstructed
scaffold evaluation is additionally deterministic (`PYTHONHASHSEED=0`, exported
by the `Makefile`).

1. **Data curation** — `make data`
   (`curate_hamdb_directions.py`, `curate_autophagy_genes.py`,
   `fetch_chembl_pubchem_neutrals.py`; plus the GSE92742 download, see
   `data/README.md`)
2. **Transcriptome regression + collapse audit** — `make embeddings train`
   (`embed_unimol.py`, `train_v6_contrastive.py`, `analyze_v6_results.py`,
   `diag_*` ridge/probe diagnostics, `extract_fig1_data.py`; paper §3.1, Fig. 1)
3. **Signature matching** — `make analyze`
   (`extract_autophagy_signatures*.py`, `diagnose_autophagy_signatures.py`,
   `flux_informed_signature_and_assay_integration.py`; paper §3.5)
4. **Binary modulator classification** — `make classify`
   (`generate_clamp_embeddings.py`, `hamdb_binary_classifier_xgb.py`,
   `train_3class_autophagy_classifier.py`, `cascade_autophagy_classifier.py`,
   `direction_feature_ablation.py`, `generate_roc_curves.py`; paper §3.2, Fig. 2)
5. **External validation on ChEMBL** — part of `make analyze`
   (`analyze_chembl_hits.py`, `compare_chembl_clamp_morgan_enrichment.py`;
   paper §3.3, Fig. 3)
6. **Paper statistics and figures** — `make stats figures`
   (`paper_bootstrap_ci.py`, `paper_feature_importance.py`,
   `make_journal_figures.py`)

## Notes

- Evaluations use Murcko-scaffold-aware splits throughout
  (Bemis & Murcko 1996; MoleculeNet protocol). Random splits are not used
  for any headline number.
- The binary/3-class XGBoost models are provided pretrained in `models/`, so
  `make stats` and `make figures` run without retraining.
- Large regenerable intermediates (raw L1000 matrices, multi-GB ChEMBL
  prediction dumps, embedding caches) are git-ignored; the pipeline
  regenerates them in the repository root.

## License

MIT (see `LICENSE`). Data resources retain their own licenses: HAMdb
(http://hamdb.scbdd.com), LINCS L1000 (GEO GSE92742), ChEMBL
(https://www.ebi.ac.uk/chembl/).
