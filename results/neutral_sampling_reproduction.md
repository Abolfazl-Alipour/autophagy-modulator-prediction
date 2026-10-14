# Reproduction note: ChEMBL neutral sampling count

The structure-only classifiers were trained on **5,893** scaffold-family-balanced
ChEMBL neutrals (sampled 2026-06-23, seed `RANDOM_STATE=42`).

Re-running the committed pipeline today (`PYTHONHASHSEED=0`, committed exclusion list
`data/chembl_autophagy_exclusion_ids.json`, committed candidates
`data/chembl_neutral_candidates.csv`) yields:

| Environment | RDKit | pandas | Sampled neutrals |
|---|---|---|---|
| paper venv (Python 3.12) | 2026.3.6 | 3.0.6 | 5,881 |
| miniconda (Python 3.13, training env) | 2026.3.2 | 2.3.3 | 5,880 |
| original run (2026-06-23) | 2026.3.x | 2.3.x | **5,893** |

Root cause of the +-12 compound (0.2%) drift: `cluster_scaffolds` built its Butina
input as `list(set(scaffolds))`. Set iteration order depends on (a) the Python hash
seed and (b) the exact scaffold strings, and RDKit 2026.3.2 vs 2026.3.6 canonicalize
6 scaffolds differently (aromaticity normalization), changing both. Cluster
membership near the 0.7 Tanimoto cutoff then shifts slightly, and the per-family
cap of 5 compounds propagates the difference into the final count.

Fix applied 2026-10-14: the input is now `sorted(set(...))`, so future runs are
deterministic given the same RDKit version. The trained models, the saved pkls, and
every downstream number in the manuscript correspond to the original 5,893-compound
draw; the drift affects only attempts to re-derive the exact training list, not the
published metrics (which are computed from the saved models and committed result
tables).
