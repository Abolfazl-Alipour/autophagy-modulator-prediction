PYTHON := venv/bin/python
export PYTHONHASHSEED := 0

.PHONY: check figures stats paper full clean

check: venv
	@$(PYTHON) -c "import rdkit, xgboost, sklearn, pandas, numpy; print('deps ok')"

figures: venv  ## regenerate all 5 paper figures from included caches/models
	$(PYTHON) make_journal_figures.py

stats: venv  ## bootstrap AUROC CIs + Morgan-bit feature importance
	$(PYTHON) paper_bootstrap_ci.py
	$(PYTHON) paper_feature_importance.py

paper:  ## recompile the manuscript (tectonic or pdflatex)
	@cd paper && (tectonic main.tex || pdflatex -interaction=nonstopmode main.tex)

## Full pipeline from raw data (requires data/l1000/ GSE92742 files, CLAMP, Uni-Mol):
##   python curate_hamdb_directions.py && python curate_autophagy_genes.py
##   python fetch_chembl_pubchem_neutrals.py
##   python embed_unimol.py && python train_v6_contrastive.py && python analyze_v6_results.py
##   python generate_clamp_embeddings.py && python hamdb_binary_classifier_xgb.py
##   python train_3class_autophagy_classifier.py && python cascade_autophagy_classifier.py
##   python generate_roc_curves.py && python analyze_chembl_hits.py
##   python compare_chembl_clamp_morgan_enrichment.py
##   make stats && make figures
full: stats figures

clean:
	rm -rf __pycache__ venv
