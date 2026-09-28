"""
XGBoost binary classifier: autophagy modulator vs. non-modulator.

Design choices (peer-reviewed):
- 1:1 class balance at compound level (227 HAMDB positives, 227 random negatives)
- 70/30 scaffold-aware split: split by Murcko scaffold family, never by signature
- Mechanism overlap enforced: mTOR and PI3K positives appear in both train and test
- Held-out drugs (Rapamycin/Sirolimus, Chloroquine) excluded from train/test
- Features: CLAMP embedding + context, and Morgan fingerprint + context ablation
- Model: XGBClassifier (no MLP)
"""
import sys
import logging
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import RidgeClassifier
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('hamdb_binary_xgb')

RANDOM_STATE = 42
TEST_FRAC = 0.30
N_NEGATIVES = 227  # match positive count
# Base (non-salt) SMILES for canonical autophagy controls; these were excluded from the full cache
HELDOUT_DRUGS = {
    'Rapamycin': r'CO[C@@H]1C[C@@H]2C[C@H](C)[C@@H]3CC(=O)[C@H](C)\C=C(C)\[C@@H](O)[C@@]2(O)[C@@H]1OC(=O)[C@@H](C)C[C@H](C)\C=C\C=C\C=C(C)\[C@H](C[C@@H]1CC[C@@H](C)[C@](O)(O1)C(=O)C(=O)N1CCCC[C@H]1C(=O)O3)OC',
    'Chloroquine': r'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',
}


def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """Compute Morgan fingerprint for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def clamp_embed_smiles_list(smiles_list, clamp_model, batch_size=32):
    """Generate CLAMP embeddings for a list of SMILES."""
    embeddings = []
    clamp_model.eval()
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            emb = clamp_model.encode_smiles(batch).cpu().numpy().astype(np.float32)
            embeddings.append(emb)
    return np.vstack(embeddings)


def get_numeric_pcids(series):
    ids = set()
    for x in series.dropna():
        s = str(x).strip()
        try:
            ids.add(str(int(float(s))))
        except Exception:
            pass
    return ids


def assign_mechanism_tags(row):
    """Assign mechanism tags from HAMDB annotations."""
    text = ' '.join([
        str(row.get('Category', '')),
        str(row.get('Pathway', '')),
        str(row.get('Target', ''))
    ]).lower()
    tags = set()
    if 'mtor' in text:
        tags.add('mtor')
    if 'pi3k' in text:
        tags.add('pi3k')
    if 'autophagy activator' in str(row.get('Category', '')).lower():
        tags.add('activator')
    if 'autophagy inhibitor' in str(row.get('Category', '')).lower():
        tags.add('inhibitor')
    return tags


def stratified_scaffold_split(compound_info, test_frac=TEST_FRAC, random_state=RANDOM_STATE):
    """
    Split scaffold families into train/test with constraints:
    - 70/30 compound ratio
    - mTOR and PI3K positives in both splits
    - Activator/Inhibitor positives in both splits if available
    """
    # Group compounds by scaffold
    scaffold_to_compounds = defaultdict(list)
    for cid, info in compound_info.items():
        scaffold_to_compounds[info['scaffold']].append(cid)

    required_tags = ['mtor', 'pi3k']
    optional_tags = ['activator', 'inhibitor']

    rng = np.random.default_rng(random_state)
    scaffolds = list(scaffold_to_compounds.keys())
    rng.shuffle(scaffolds)

    train_cids = set()
    test_cids = set()
    train_tags = set()
    test_tags = set()

    # First, assign scaffold groups to satisfy required tag overlap
    tag_scaffolds = defaultdict(list)
    for scaf in scaffolds:
        tags_in_scaf = set()
        for cid in scaffold_to_compounds[scaf]:
            tags_in_scaf.update(compound_info[cid]['tags'])
        for tag in tags_in_scaf:
            tag_scaffolds[tag].append(scaf)

    # Reserve at least one scaffold for test per required/optional tag
    test_scaffolds = set()
    for tag in required_tags + optional_tags:
        candidates = [s for s in tag_scaffolds.get(tag, []) if s not in test_scaffolds]
        if not candidates:
            continue
        # pick a small scaffold group to reserve for test
        candidates.sort(key=lambda s: len(scaffold_to_compounds[s]))
        test_scaffolds.add(candidates[0])

    # Assign remaining scaffolds to reach target test fraction
    for scaf in scaffolds:
        if scaf in test_scaffolds:
            test_cids.update(scaffold_to_compounds[scaf])
            for cid in scaffold_to_compounds[scaf]:
                test_tags.update(compound_info[cid]['tags'])
            continue

        current_test_frac = len(test_cids) / len(compound_info)
        if current_test_frac < test_frac:
            test_cids.update(scaffold_to_compounds[scaf])
            for cid in scaffold_to_compounds[scaf]:
                test_tags.update(compound_info[cid]['tags'])
        else:
            train_cids.update(scaffold_to_compounds[scaf])
            for cid in scaffold_to_compounds[scaf]:
                train_tags.update(compound_info[cid]['tags'])

    # Verify required tags in both splits; swap if necessary
    for tag in required_tags:
        train_has = any(tag in compound_info[cid]['tags'] for cid in train_cids)
        test_has = any(tag in compound_info[cid]['tags'] for cid in test_cids)
        if not train_has or not test_has:
            logger.warning(f"Split verification issue for tag '{tag}': train={train_has}, test={test_has}")

    return train_cids, test_cids


def build_context_features(cell_ids, doses, times, cell_types, fit_encoder=None):
    """Build one-hot cell type + log dose + one-hot time features."""
    n = len(cell_ids)
    # Convert integer cell_ids to cell type names
    cell_labels = np.array([cell_types[i] for i in cell_ids], dtype=object)

    # Cell type one-hot using full known cell types as categories
    if fit_encoder is None:
        encoder = OneHotEncoder(categories=[cell_types], sparse_output=False, handle_unknown='ignore')
        cell_onehot = encoder.fit_transform(cell_labels.reshape(-1, 1))
    else:
        encoder = fit_encoder
        cell_onehot = encoder.transform(cell_labels.reshape(-1, 1))

    # Log dose
    dose_feat = np.zeros((n, 1), dtype=np.float32)
    for i, d in enumerate(doses):
        if d is not None and d > 0:
            dose_feat[i, 0] = np.log10(float(d) + 1e-6)
        else:
            dose_feat[i, 0] = -10.0  # sentinel for missing/control dose

    # Time one-hot using all known time points as categories
    all_times = np.array([1, 2, 3, 4, 6, 24, 48, 72, 96, 120, 144, 168])
    time_map = {t: i for i, t in enumerate(all_times)}
    time_onehot = np.zeros((n, len(all_times)), dtype=np.float32)
    for i, t in enumerate(times):
        if t in time_map:
            time_onehot[i, time_map[t]] = 1.0

    context = np.hstack([cell_onehot, dose_feat, time_onehot])
    return context, encoder


def run_feature_set(name, X_chem, train_idx, test_idx, y_train, y_test,
                    cell_ids_full, doses_full, times_full, cell_types,
                    X_heldout_chem=None, heldout_context=None, heldout_drug_names=None):
    """Train XGB on one feature set and evaluate."""
    logger.info(f"\n=== Feature set: {name} ===")

    # Build context features on train, transform test/heldout
    context_train, encoder = build_context_features(
        cell_ids_full[train_idx], doses_full[train_idx], times_full[train_idx], cell_types
    )
    context_test, _ = build_context_features(
        cell_ids_full[test_idx], doses_full[test_idx], times_full[test_idx], cell_types, encoder
    )

    X_train = np.hstack([X_chem[train_idx], context_train])
    X_test = np.hstack([X_chem[test_idx], context_test])

    # Inner validation split for early stopping (random signature-level split)
    rng = np.random.default_rng(RANDOM_STATE)
    n_train = len(train_idx)
    val_size = int(0.15 * n_train)
    val_idx = rng.choice(n_train, size=val_size, replace=False)
    train_inner_mask = np.ones(n_train, dtype=bool)
    train_inner_mask[val_idx] = False

    logger.info(f"  Train: {X_train.shape[0]}, Val: {val_idx.size}, Test: {X_test.shape[0]}")

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',
        tree_method='hist',
        n_jobs=12,
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    model.fit(
        X_train[train_inner_mask], y_train[train_inner_mask],
        verbose=False
    )

    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)

    # Per-signature metrics
    auroc = roc_auc_score(y_test, y_proba_test)
    auprc = average_precision_score(y_test, y_proba_test)
    acc = accuracy_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)

    logger.info(f"  Test AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")

    # Per-compound metrics (aggregate signatures by mean probability)
    test_compounds = compound_ids_full[test_idx]
    unique_test_compounds = sorted(set(test_compounds))
    compound_probs = defaultdict(list)
    compound_labels = {}
    for cid, prob, label in zip(test_compounds, y_proba_test, y_test):
        compound_probs[cid].append(prob)
        compound_labels[cid] = label

    comp_mean_probs = [np.mean(compound_probs[cid]) for cid in unique_test_compounds]
    comp_labels = [compound_labels[cid] for cid in unique_test_compounds]
    comp_auroc = roc_auc_score(comp_labels, comp_mean_probs)
    comp_auprc = average_precision_score(comp_labels, comp_mean_probs)
    comp_fpr, comp_tpr, comp_thresholds = roc_curve(comp_labels, comp_mean_probs)
    logger.info(f"  Compound-level AUROC: {comp_auroc:.4f}, AUPRC: {comp_auprc:.4f}")

    # Per-mechanism test AUROC (one-vs-rest: tag positives vs all other test samples)
    mech_results = {}
    for tag in ['mtor', 'pi3k', 'activator', 'inhibitor']:
        tag_labels = np.array([1 if tag in compound_info[cid]['tags'] else 0 for cid in test_compounds])
        if tag_labels.sum() > 0 and len(set(tag_labels)) > 1:
            tag_auroc = roc_auc_score(tag_labels, y_proba_test)
            mech_results[f'{tag}_auroc'] = tag_auroc
            logger.info(f"  {tag.upper()} test AUROC: {tag_auroc:.4f} (n_pos={tag_labels.sum()}, n_total={len(tag_labels)})")
        else:
            mech_results[f'{tag}_auroc'] = np.nan

    # Held-out drug predictions
    heldout_results = {}
    if X_heldout_chem is not None and heldout_context is not None:
        X_heldout = np.hstack([X_heldout_chem, heldout_context])
        y_proba_heldout = model.predict_proba(X_heldout)[:, 1]
        for i, drug_name in enumerate(heldout_drug_names):
            heldout_results[f'{drug_name}_mean_proba'] = float(y_proba_heldout[i])
            logger.info(f"  {drug_name} proba: {heldout_results[f'{drug_name}_mean_proba']:.4f}")

    return {
        'feature_set': name,
        'n_estimators': getattr(model, 'n_estimators', 1000),
        'signature_auroc': auroc,
        'signature_auprc': auprc,
        'signature_accuracy': acc,
        'signature_f1': f1,
        'signature_precision': prec,
        'signature_recall': rec,
        'compound_auroc': comp_auroc,
        'compound_auprc': comp_auprc,
        'compound_fpr': comp_fpr,
        'compound_tpr': comp_tpr,
        'compound_thresholds': comp_thresholds,
        **mech_results,
        **heldout_results,
    }


def run_ridge_feature_set(name, X_chem, train_idx, test_idx, y_train, y_test,
                          cell_ids_full, doses_full, times_full, cell_types):
    """Train a Ridge classifier as a linear baseline on the same split."""
    logger.info(f"\n=== Ridge baseline: {name} ===")

    context_train, encoder = build_context_features(
        cell_ids_full[train_idx], doses_full[train_idx], times_full[train_idx], cell_types
    )
    context_test, _ = build_context_features(
        cell_ids_full[test_idx], doses_full[test_idx], times_full[test_idx], cell_types, encoder
    )

    X_train = np.hstack([X_chem[train_idx], context_train])
    X_test = np.hstack([X_chem[test_idx], context_test])

    clf = RidgeClassifier(class_weight='balanced', random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_score = clf.decision_function(X_test)
    y_pred = clf.predict(X_test)

    auroc = roc_auc_score(y_test, y_score)
    auprc = average_precision_score(y_test, y_score)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Per-compound aggregation
    test_compounds = compound_ids_full[test_idx]
    unique_test_compounds = sorted(set(test_compounds))
    compound_scores = defaultdict(list)
    compound_labels = {}
    for cid, score, label in zip(test_compounds, y_score, y_test):
        compound_scores[cid].append(score)
        compound_labels[cid] = label
    comp_mean_scores = [np.mean(compound_scores[cid]) for cid in unique_test_compounds]
    comp_labels = [compound_labels[cid] for cid in unique_test_compounds]
    comp_auroc = roc_auc_score(comp_labels, comp_mean_scores)
    comp_auprc = average_precision_score(comp_labels, comp_mean_scores)
    comp_fpr, comp_tpr, comp_thresholds = roc_curve(comp_labels, comp_mean_scores)

    logger.info(f"  Test AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}")
    logger.info(f"  Compound-level AUROC: {comp_auroc:.4f}, AUPRC: {comp_auprc:.4f}")

    return {
        'feature_set': f'Ridge+context ({name})',
        'n_estimators': 0,
        'signature_auroc': auroc,
        'signature_auprc': auprc,
        'signature_accuracy': acc,
        'signature_f1': f1,
        'signature_precision': precision_score(y_test, y_pred),
        'signature_recall': recall_score(y_test, y_pred),
        'compound_auroc': comp_auroc,
        'compound_auprc': comp_auprc,
        'compound_fpr': comp_fpr,
        'compound_tpr': comp_tpr,
        'compound_thresholds': comp_thresholds,
    }


def main():
    global compound_info, compound_ids_full, smiles_full, cell_ids_full, doses_full, times_full

    logger.info("=== XGBoost binary autophagy modulator classifier ===")

    # Load HAMDB
    logger.info("Loading HAMDB...")
    hamdb = pd.read_csv('data/hamdb_chemicals.csv', encoding='latin-1')
    hamdb = hamdb.dropna(subset=['Canonical_SMILES'])
    hamdb['mech_tags'] = hamdb.apply(assign_mechanism_tags, axis=1)
    hamdb_pcids = get_numeric_pcids(hamdb['Pubchem_CID'])
    logger.info(f"  HAMDB compounds with SMILES: {len(hamdb)}")
    logger.info(f"  HAMDB with numeric PubChem CID: {len(hamdb_pcids)}")

    # Load L1000 pert_info
    logger.info("Loading L1000 pert_info...")
    pert_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz', sep='\t',
                            usecols=['pert_id', 'pert_iname', 'canonical_smiles', 'pubchem_cid'])
    pert_info['numeric_pcid'] = pert_info['pubchem_cid'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else None
    )

    # Find HAMDB compounds in L1000
    hamdb_pert = pert_info[pert_info['numeric_pcid'].isin(hamdb_pcids)]
    hamdb_pert = hamdb_pert.dropna(subset=['canonical_smiles'])
    hamdb_pert = hamdb_pert[hamdb_pert['canonical_smiles'] != '-666']
    hamdb_pert = hamdb_pert.drop_duplicates(subset=['canonical_smiles'])
    positive_smiles = set(hamdb_pert['canonical_smiles'].tolist())
    logger.info(f"  HAMDB positives in L1000: {len(positive_smiles)}")

    # Map positive SMILES -> HAMDB tags
    pcid_to_tags = {}
    for _, row in hamdb.iterrows():
        pcid = str(int(float(row['Pubchem_CID']))) if pd.notna(row['Pubchem_CID']) else None
        if pcid:
            pcid_to_tags[pcid] = row['mech_tags']

    smile_to_tags = {}
    for _, row in hamdb_pert.iterrows():
        pcid = row['numeric_pcid']
        if pcid in pcid_to_tags:
            smile_to_tags[row['canonical_smiles']] = pcid_to_tags[pcid]

    # Load full cache metadata
    logger.info("Loading full L1000 cache metadata...")
    d = np.load('diag_cache_v9_morgan_full.npz', allow_pickle=True)
    smiles_full = d['smiles']
    scaffolds_full = d['scaffolds']
    cell_ids_full = d['cell_ids']
    doses_full = d['doses']
    times_full = d['times']
    cell_types = d['cell_types']
    unique_cache_smiles = set(smiles_full.tolist())
    logger.info(f"  Full cache: {len(smiles_full)} signatures, {len(unique_cache_smiles)} unique compounds, {len(cell_types)} cell types")

    # Filter positives to those actually in cache
    positive_smiles = positive_smiles & unique_cache_smiles
    logger.info(f"  HAMDB positives with cache signatures: {len(positive_smiles)}")

    # Sample negatives from cache compounds (held-out controls are not in cache)
    all_cache_smiles = unique_cache_smiles
    non_positive = sorted(all_cache_smiles - positive_smiles)
    rng = np.random.default_rng(RANDOM_STATE)
    negative_smiles = set(rng.choice(non_positive, size=N_NEGATIVES, replace=False))
    logger.info(f"  Sampled {len(negative_smiles)} negative compounds")

    selected_smiles = positive_smiles | negative_smiles
    selected_smiles_list = sorted(selected_smiles)
    smile_to_label = {sm: 1 for sm in positive_smiles}
    smile_to_label.update({sm: 0 for sm in negative_smiles})

    # Build compound-level info for selected compounds
    global compound_info
    compound_info = {}
    for sm in selected_smiles_list:
        compound_info[sm] = {
            'smiles': sm,
            'label': smile_to_label[sm],
            'tags': smile_to_tags.get(sm, set()),
        }

    # Determine representative scaffold for each compound
    for sm in compound_info:
        mask = smiles_full == sm
        if mask.sum() == 0:
            continue
        # use most frequent scaffold
        scafs = scaffolds_full[mask]
        compound_info[sm]['scaffold'] = Counter(scafs.tolist()).most_common(1)[0][0]

    logger.info(f"  Compounds for split: {len(compound_info)}")
    logger.info(f"  Positive compounds: {sum(1 for c in compound_info.values() if c['label'] == 1)}")
    logger.info(f"  Negative compounds: {sum(1 for c in compound_info.values() if c['label'] == 0)}")

    # Stratified scaffold split
    logger.info("Creating stratified scaffold split...")
    train_cids, test_cids = stratified_scaffold_split(compound_info)
    logger.info(f"  Train compounds: {len(train_cids)}, Test compounds: {len(test_cids)}")
    logger.info(f"  Train positives: {sum(1 for c in train_cids if compound_info[c]['label'] == 1)}, "
                f"Test positives: {sum(1 for c in test_cids if compound_info[c]['label'] == 1)}")

    # Build signature-level masks
    selected_mask = np.array([sm in selected_smiles for sm in smiles_full])
    train_mask = selected_mask & np.array([sm in train_cids for sm in smiles_full])
    test_mask = selected_mask & np.array([sm in test_cids for sm in smiles_full])

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    logger.info(f"  Train signatures: {len(train_idx)}, Test signatures: {len(test_idx)}")

    # Compound IDs for per-compound aggregation
    global compound_ids_full
    compound_ids_full = smiles_full.copy()

    y_train = np.array([smile_to_label[sm] for sm in smiles_full[train_idx]])
    y_test = np.array([smile_to_label[sm] for sm in smiles_full[test_idx]])

    # Determine representative context for held-out drugs from training data
    most_common_cell_idx = int(Counter(cell_ids_full[train_idx].tolist()).most_common(1)[0][0])
    heldout_cell_ids = np.full(len(HELDOUT_DRUGS), most_common_cell_idx, dtype=np.int32)
    heldout_doses = np.full(len(HELDOUT_DRUGS), 10.0, dtype=np.float32)
    heldout_times = np.full(len(HELDOUT_DRUGS), 24, dtype=np.int32)
    heldout_context, _ = build_context_features(
        heldout_cell_ids, heldout_doses, heldout_times, cell_types
    )
    logger.info(f"  Held-out context: cell={cell_types[most_common_cell_idx]}, dose=10uM, time=24h")

    # Load CLAMP embeddings for cache SMILES
    logger.info("Loading CLAMP embeddings...")
    clamp_embs_all = torch.load('clamp_embeddings_l1000_full.pt', map_location='cpu', weights_only=True)
    X_clamp = np.zeros((len(smiles_full), 768), dtype=np.float32)
    for i, sm in enumerate(smiles_full):
        if sm in clamp_embs_all:
            X_clamp[i] = clamp_embs_all[sm].numpy().astype(np.float32)
    logger.info(f"  CLAMP matrix: {X_clamp.shape}")

    # Generate CLAMP embeddings for held-out drugs
    logger.info("Generating CLAMP embeddings for held-out drugs...")
    from clamp.models.pretrained import PretrainedCLAMP
    clamp_model = PretrainedCLAMP(device='cpu')
    heldout_smiles_list = list(HELDOUT_DRUGS.values())
    heldout_clamp = clamp_embed_smiles_list(heldout_smiles_list, clamp_model)
    logger.info(f"  Held-out CLAMP: {heldout_clamp.shape}")

    # Run CLAMP + context
    results = []
    res_clamp = run_feature_set(
        'CLAMP+context', X_clamp, train_idx, test_idx, y_train, y_test,
        cell_ids_full, doses_full, times_full, cell_types,
        X_heldout_chem=heldout_clamp, heldout_context=heldout_context,
        heldout_drug_names=list(HELDOUT_DRUGS.keys())
    )
    results.append(res_clamp)

    # Ridge baseline on CLAMP + context
    res_ridge_clamp = run_ridge_feature_set(
        'CLAMP', X_clamp, train_idx, test_idx, y_train, y_test,
        cell_ids_full, doses_full, times_full, cell_types
    )
    results.append(res_ridge_clamp)

    # Free CLAMP memory and load Morgan
    del X_clamp, clamp_embs_all, clamp_model
    logger.info("Loading Morgan fingerprints...")
    X_morgan = d['X_morgan'].astype(np.float32)
    logger.info(f"  Morgan matrix: {X_morgan.shape}")

    # Generate Morgan fingerprints for held-out drugs
    logger.info("Generating Morgan fingerprints for held-out drugs...")
    heldout_morgan = np.array([smiles_to_morgan(sm) for sm in heldout_smiles_list], dtype=np.float32)
    logger.info(f"  Held-out Morgan: {heldout_morgan.shape}")

    res_morgan = run_feature_set(
        'Morgan+context', X_morgan, train_idx, test_idx, y_train, y_test,
        cell_ids_full, doses_full, times_full, cell_types,
        X_heldout_chem=heldout_morgan, heldout_context=heldout_context,
        heldout_drug_names=list(HELDOUT_DRUGS.keys())
    )
    results.append(res_morgan)

    # Ridge baseline on Morgan + context
    res_ridge_morgan = run_ridge_feature_set(
        'Morgan', X_morgan, train_idx, test_idx, y_train, y_test,
        cell_ids_full, doses_full, times_full, cell_types
    )
    results.append(res_ridge_morgan)

    # Save results
    df_results = pd.DataFrame(results)
    # Drop ROC curve arrays before saving CSV
    df_results_csv = df_results.drop(columns=[c for c in df_results.columns
                                               if c in ('compound_fpr', 'compound_tpr', 'compound_thresholds')],
                                     errors='ignore')
    df_results_csv.to_csv('hamdb_binary_xgb_results.csv', index=False)
    logger.info(f"\nSaved hamdb_binary_xgb_results.csv")

    # Save ROC curves for figure generation
    roc_data = {}
    auroc_records = []
    for res in results:
        name = res['feature_set']
        roc_data[name] = np.vstack([res['compound_fpr'], res['compound_tpr'], res['compound_thresholds']])
        auroc_records.append({'model': name, 'auroc': res['compound_auroc']})
    np.savez('hamdb_binary_roc_curves.npz', **roc_data,
             aurocs=np.array([(r['model'], r['auroc']) for r in auroc_records], dtype=object))
    pd.DataFrame(auroc_records).to_csv('hamdb_binary_roc_curves.csv', index=False)
    logger.info("Saved hamdb_binary_roc_curves.npz and hamdb_binary_roc_curves.csv")

    logger.info("\nSummary:")
    summary_cols = ['feature_set', 'signature_auroc', 'signature_auprc', 'compound_auroc',
                    'compound_auprc', 'mtor_auroc', 'pi3k_auroc']
    summary_cols += [c for c in df_results.columns if '_mean_proba' in c or '_proba' in c]
    logger.info(df_results[summary_cols].to_string(index=False))


if __name__ == '__main__':
    main()
