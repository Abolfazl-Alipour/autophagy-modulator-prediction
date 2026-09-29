#!/usr/bin/env bash
# One-time environment setup for the autophagy-modulator-prediction pipeline.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Creating virtual environment"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Installing CLAMP (pinned to the study's submodule revision)"
mkdir -p src data/models/clamp_clip
if [ ! -d src/clamp/.git ]; then
  git clone --depth 1 https://github.com/ml-jku/clamp.git src/clamp
fi
pip install loguru swifter "mhnreact @ git+https://github.com/ml-jku/mhn-react.git"

echo "==> Downloading the pretrained CLAMP checkpoint (auto-download fallback:"
echo "    the pipeline also downloads it on first use if missing)"
if [ ! -f data/models/clamp_clip/checkpoint.pt ]; then
  wget -q --show-progress -O data/models/clamp_clip/checkpoint.pt \
    https://cloud.ml.jku.at/s/7nxgpAQrTr69Rp2/download/checkpoint.pt
fi
if [ ! -f data/models/clamp_clip/hp.json ]; then
  wget -q -O data/models/clamp_clip/hp.json \
    https://cloud.ml.jku.at/s/dRX9TWPrF7WqnHd/download/hp.json
fi

echo "==> Checking required inputs"
for f in data/hamdb_autophagy_directions.csv data/chembl_neutral_candidates.csv \
         data/autophagy_genes.txt data/curated_phenotypes.csv \
         models/binary_morgan_only_xgb.pkl models/binary_morgan_clamp_xgb.pkl; do
  [ -f "$f" ] && echo "  ok: $f" || { echo "  MISSING: $f"; exit 1; }
done

cat <<'MSG'
Setup complete. Next steps:
  make figures   # regenerate all 5 paper figures from the included caches (minutes, no L1000 needed)
  make stats     # bootstrap AUROC CIs + feature importance (needs CLAMP checkpoint, downloaded above)
  make paper     # recompile the manuscript (needs a LaTeX toolchain, e.g. tectonic or TeX Live)
For the full pipeline from raw data (incl. the L1000 download), see README.md.
MSG
