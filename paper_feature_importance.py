#!/usr/bin/env python3
"""Supplementary feature-importance analysis for the paper.

Decodes the top Morgan fingerprint bits of the saved structure-only binary
classifier (binary_morgan_only_xgb.pkl, 2048-bit Morgan FP radius 2) into
substructure SMARTS and compares bit frequencies between HAMDB autophagy
modulators and a ChEMBL background sample.

Outputs:
  - fig5_feature_importance.png   (300 dpi, ~6.5 in wide)
  - paper_feature_importance.csv
"""

from pathlib import Path
import pickle
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import FindAtomEnvironmentOfRadiusN, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

REPO = str(Path(__file__).resolve().parent)
MODEL_PATH = f"{REPO}/models/binary_morgan_only_xgb.pkl"
HAMDB_PATH = f"{REPO}/data/hamdb_autophagy_directions.csv"
BACKGROUND_PATH = f"{REPO}/data/chembl_neutral_candidates.csv"
N_BITS = 2048
RADIUS = 2
TOP_N = 15
N_BACKGROUND_SAMPLE = 2000
RANDOM_SEED = 42


def load_model(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"pkl object type: {type(obj)}")
    model = None
    if hasattr(obj, "get_booster"):
        model = obj
    elif isinstance(obj, dict):
        print(f"pkl dict keys: {list(obj.keys())}")
        for key in ("model", "clf", "classifier", "xgb", "estimator"):
            if key in obj and hasattr(obj[key], "get_booster"):
                model = obj[key]
                print(f"  -> using model stored under key '{key}'")
                break
        if model is None:
            for v in obj.values():
                if hasattr(v, "get_booster"):
                    model = v
                    print(f"  -> using model stored under value of type {type(v)}")
                    break
    if model is None:
        raise TypeError(f"Could not locate an XGBoost model inside {path}")
    return model, obj


def bit_env_smiles(mol, bit_info, bit):
    """Collect SMARTS of atom environments that set the given bit."""
    envs = []
    if bit not in bit_info:
        return envs
    for atom_idx, radius in bit_info[bit]:
        if radius == 0:
            envs.append(f"[#{mol.GetAtomWithIdx(atom_idx).GetAtomicNum()}]")
            continue
        env = FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        if not env:
            continue
        envs.append(Chem.MolToSmarts(Chem.PathToSubmol(mol, env)))
    return envs


def fingerprint_with_info(mol):
    bit_info = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, RADIUS, nBits=N_BITS, bitInfo=bit_info
    )
    return fp, bit_info


def main():
    model, raw = load_model(MODEL_PATH)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k in ("model", "clf", "classifier", "xgb", "estimator"):
                continue
            print(f"metadata[{k!r}]: {type(v)} = {str(v)[:200]}")

    # --- feature importances -------------------------------------------------
    scores = model.get_booster().get_score(importance_type="gain")
    print(f"\nfeatures with nonzero gain: {len(scores)} / {N_BITS}")
    imp = pd.Series(scores).sort_values(ascending=False)
    imp.index = imp.index.map(lambda s: int(str(s).lstrip("f")))
    top = imp.head(TOP_N)
    ranks = imp.rank(method="min", ascending=False).astype(int)

    # --- molecule sets -------------------------------------------------------
    hamdb = pd.read_csv(HAMDB_PATH)
    print(f"\nHAMDB columns: {list(hamdb.columns)}")
    print(f"HAMDB autophagy_direction counts:\n{hamdb['autophagy_direction'].value_counts()}")

    modulators = hamdb[hamdb["autophagy_direction"].isin(["activator", "inhibitor"])][
        "Canonical_SMILES"
    ].dropna()
    print(f"modulators (Activator+Inhibitor): {len(modulators)}")

    bg = pd.read_csv(BACKGROUND_PATH)
    bg_smiles = bg["smiles"].dropna().sample(
        n=min(N_BACKGROUND_SAMPLE, len(bg)), random_state=RANDOM_SEED
    )

    # --- bit frequencies -----------------------------------------------------
    def bit_counts(smiles_series, label):
        counts = Counter()
        n_valid = 0
        for smi in smiles_series:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            n_valid += 1
            fp, _ = fingerprint_with_info(mol)
            on = set(fp.GetOnBits())
            for b in on:
                counts[b] += 1
        print(f"{label}: {n_valid} valid molecules")
        return counts, n_valid

    mod_counts, n_mod = bit_counts(modulators, "HAMDB modulators")
    bg_counts, n_bg = bit_counts(bg_smiles, "ChEMBL background sample")

    # --- decode top bits to SMARTS -------------------------------------------
    top_bits = list(top.index)
    bit_smarts = {b: Counter() for b in top_bits}
    for smi in modulators:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        _, bit_info = fingerprint_with_info(mol)
        for b in top_bits:
            for env in bit_env_smiles(mol, bit_info, b):
                bit_smarts[b][env] += 1

    rows = []
    for b in top_bits:
        smarts, n_mol_with_env = (
            bit_smarts[b].most_common(1)[0] if bit_smarts[b] else ("<no env found>", 0)
        )
        f_mod = mod_counts.get(b, 0) / n_mod
        f_bg = bg_counts.get(b, 0) / n_bg
        fold = (f_mod / f_bg) if f_bg > 0 else np.inf
        rows.append(
            {
                "bit": b,
                "gain": round(float(top[b]), 3),
                "rank": int(ranks[b]),
                "smarts": smarts,
                "freq_modulators": round(f_mod, 4),
                "freq_background": round(f_bg, 4),
                "fold_enrichment": round(fold, 2) if np.isfinite(fold) else "inf",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(f"{REPO}/paper/paper_feature_importance.csv", index=False)
    print("\nTop bits:")
    print(df.to_string(index=False))

    # --- figure ---------------------------------------------------------------
    def short_label(row):
        s = row["smarts"]
        s = s if len(s) <= 28 else s[:25] + "..."
        return f"bit {row['bit']} | {s}"

    labels = [short_label(r) for _, r in df.iloc[::-1].iterrows()]
    values = df["gain"].values[::-1]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(range(len(values)), values, color="#4878A8", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("XGBoost mean gain", fontsize=9)
    ax.set_title(
        "Top 15 Morgan fingerprint bits by gain\n(binary structure-only classifier)",
        fontsize=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{REPO}/paper/fig5_feature_importance.png", dpi=300)
    plt.close(fig)

    print("\nSummary:")
    print(f"  background sample size: {n_bg} ChEMBL molecules")
    print(f"  modulator set: {n_mod} HAMDB compounds (Activator+Inhibitor)")
    print("  wrote fig5_feature_importance.png and paper_feature_importance.csv")


if __name__ == "__main__":
    main()
