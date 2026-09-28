# External data

Only one dataset must be downloaded manually (too large to version-control):

**LINCS L1000 Phase I (GSE92742)** — place these files in `data/l1000/`:
- `GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx`
- `GSE92742_Broad_LINCS_sig_info.txt.gz`
- `GSE92742_Broad_LINCS_pert_info.txt.gz`

Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742
(Subramanian et al., Cell 2017).

Everything else used by the pipeline is either included in this repository
(`hamdb_autophagy_directions.csv`, `chembl_neutral_candidates.csv`,
`autophagy_genes.txt`, `curated_phenotypes.csv`) or fetched by the scripts
(ChEMBL assay annotations via the ChEMBL API, HAMdb from http://hamdb.scbdd.com).
