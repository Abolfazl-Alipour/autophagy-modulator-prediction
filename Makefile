PYTHON := venv/bin/python
export PYTHONHASHSEED := 0

.PHONY: check setup data embeddings train classify analyze stats figures paper full clean

check: venv
	@$(PYTHON) -c "import rdkit, xgboost, sklearn, pandas, numpy; print('deps ok')"

venv:
	@echo "venv not found - run ./setup.sh first" >&2; exit 1

# ---- One-command reproduction ------------------------------------------------

## Reproduce every manuscript number from the committed inputs (minutes, no L1000).
## Requires: ./setup.sh (venv, deps, CLAMP checkpoint) once.
full: stats figures

## Everything from raw data (downloads ~4 GB LINCS L1000 into data/l1000/):
##   make setup data embeddings train classify analyze stats figures
setup:
	./setup.sh

data: venv            # HAMdb labels, autophagy gene set, ChEMBL candidates
	$(PYTHON) pipeline/curate_hamdb_directions.py
	$(PYTHON) pipeline/curate_autophagy_genes.py
	$(PYTHON) pipeline/fetch_chembl_pubchem_neutrals.py

embeddings: venv      # Uni-Mol + CLAMP embeddings (needs data/l1000/)
	$(PYTHON) pipeline/embed_unimol.py
	$(PYTHON) pipeline/generate_clamp_embeddings.py

train: venv           # V6 transcriptome model + diagnostics
	$(PYTHON) pipeline/train_v6_contrastive.py
	$(PYTHON) pipeline/analyze_v6_results.py
	$(PYTHON) pipeline/diag_cache_data.py
	$(PYTHON) pipeline/diag_0_unimol_quality.py
	$(PYTHON) pipeline/diag_1_linear_baselines.py
	$(PYTHON) pipeline/diag_2_gene_probes.py

classify: venv        # binary / 3-class / cascade classifiers + ChEMBL scoring
	$(PYTHON) pipeline/hamdb_binary_classifier_xgb.py
	$(PYTHON) pipeline/train_3class_autophagy_classifier.py
	$(PYTHON) pipeline/cascade_autophagy_classifier.py
	$(PYTHON) pipeline/direction_feature_ablation.py

analyze: venv         # signatures, flux integration, enrichment, ROC curves
	$(PYTHON) pipeline/extract_autophagy_signatures.py
	$(PYTHON) pipeline/extract_autophagy_signatures_v2.py
	$(PYTHON) pipeline/diagnose_autophagy_signatures.py
	$(PYTHON) pipeline/flux_informed_signature_and_assay_integration.py
	$(PYTHON) pipeline/hamdb_autophagy_smoke_test.py
	$(PYTHON) pipeline/analyze_chembl_hits.py
	$(PYTHON) pipeline/compare_chembl_clamp_morgan_enrichment.py
	$(PYTHON) pipeline/generate_roc_curves.py

# ---- Paper outputs ------------------------------------------------------------

stats: venv  ## bootstrap AUROC CIs + Morgan-bit feature importance
	$(PYTHON) pipeline/paper_bootstrap_ci.py
	$(PYTHON) pipeline/paper_feature_importance.py

figures: venv  ## regenerate all 5 paper figures from included caches/models
	$(PYTHON) pipeline/make_journal_figures.py

paper:  ## recompile the manuscript (tectonic or pdflatex)
	@cd paper && (tectonic main.tex || pdflatex -interaction=nonstopmode main.tex)

clean:
	rm -rf __pycache__ pipeline/__pycache__ venv
