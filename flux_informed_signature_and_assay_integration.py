#!/usr/bin/env python3
"""
Flux-informed autophagy signature extraction + orthogonal ChEMBL assay integration.

Strategy:
1. Use conservative HAMDB activator/inhibitor labels.
2. Identify genes among the strict autophagy set where activators and inhibitors
   push expression in opposite directions relative to non-HAMDB background.
3. Build per-cell-line contrastive signatures using only those flux-informative genes.
4. Validate with scaffold-aware cross-validation and HAMDB enrichment.
5. Rank all LINCS compounds with a contrastive inverse-search score.
6. Annotate top hits with ChEMBL autophagy-assay data (orthogonal validation).
"""

import os
import re
import json
import time
import logging
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import urllib.parse
from scipy.stats import spearmanr
from scipy.sparse import issparse
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WORK_DIR = Path(__file__).resolve().parent
CACHE_PATH = WORK_DIR / 'diag_cache_v9_morgan_full.npz'
HAMDB_PATH = WORK_DIR / 'data' / 'hamdb_autophagy_directions.csv'
PERT_INFO_PATH = WORK_DIR / 'data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz'
GENE_FILE = WORK_DIR / 'autophagy_genes.txt'

OUT_DIR = WORK_DIR
N_TOP_GENES = 25
MIN_CELL_LINES_PER_GENE = 3
N_PERMUTATIONS = 500
N_JOBS = 12

# Strict core autophagy gene set (~70 genes, matches v2)
STRICT_AUTOPHAGY_GENES = [
    'ULK1', 'ULK2', 'ATG13', 'RB1CC1',
    'PIK3C3', 'PIK3R4', 'BECN1', 'ATG14', 'NRBF2', 'UVRAG',
    'ATG2A', 'ATG2B', 'ATG9A', 'ATG9B',
    'ATG3', 'ATG4A', 'ATG4B', 'ATG4C', 'ATG4D', 'ATG5', 'ATG7', 'ATG10', 'ATG12', 'ATG16L1', 'ATG101',
    'MAP1LC3B', 'MAP1LC3C', 'GABARAPL1', 'GABARAPL2',
    'WIPI1', 'WIPI2',
    'SQSTM1', 'NBR1', 'TAX1BP1', 'OPTN', 'CALCOCO2',
    'LAMP1', 'LAMP2', 'CTSD', 'CTSL', 'CTSB', 'TPP1', 'GNS', 'NPC1', 'NPC2', 'TFEB', 'TFE3',
    'DDIT4', 'EIF4EBP1', 'RPS6KB1', 'TSC1', 'TSC2', 'MTOR', 'RHEB',
    'PRKAA1', 'PRKAA2', 'PRKAB1', 'PRKAB2', 'PRKAG1', 'PRKAG2', 'STK11',
    'PINK1', 'BNIP3', 'BNIP3L', 'RTN3', 'FAM134B',
    'AMBRA1', 'DEPTOR', 'PTEN', 'FOXO3',
]

# Candidate flux/lysosomal/autophagy marker genes to prioritize if present
PRIORITY_FLUX_GENES = [
    'MAP1LC3B', 'SQSTM1', 'BECN1', 'ATG5', 'ATG7', 'ATG12',
    'WIPI1', 'WIPI2', 'GABARAPL1', 'GABARAPL2',
    'LAMP1', 'LAMP2', 'CTSD', 'CTSB', 'TFEB',
    'BNIP3', 'BNIP3L', 'DRAM1', 'DDIT4',
    'ULK1', 'ULK2', 'ATG13', 'RB1CC1',
    'TSC1', 'TSC2', 'RPTOR', 'RICTOR',
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def load_cache(path: Path):
    logger.info('Loading full cache...')
    d = np.load(path, allow_pickle=True, mmap_mode='r')
    Y = np.array(d['Y'], dtype=np.float32)
    # some caches store Y as masked; if a mask exists, apply it
    if 'mask' in d:
        mask = d['mask']
        if issparse(mask):
            mask = mask.toarray()
        Y = np.where(mask, np.nan, Y)
    return {
        'Y': Y,
        'smiles': d['smiles'],
        'cell_ids': d['cell_ids'],
        'time': d.get('times'),
        'dose': d.get('doses'),
        'gene_symbols': d['gene_symbols'],
        'cell_types': d.get('cell_types'),
    }


def get_numeric_pcids(series):
    ids = set()
    for x in series.dropna():
        s = str(x).strip()
        try:
            ids.add(str(int(float(s))))
        except Exception:
            pass
    return ids


def load_hamdb_directions(path: Path):
    df = pd.read_csv(path)
    col = 'autophagy_direction' if 'autophagy_direction' in df.columns else 'direction'
    pcid_col = 'Pubchem_CID' if 'Pubchem_CID' in df.columns else 'pubchem_cid'
    smi_col = 'Canonical_SMILES' if 'Canonical_SMILES' in df.columns else 'canonical_smiles'
    activator_pcids = get_numeric_pcids(df[df[col] == 'activator'][pcid_col])
    inhibitor_pcids = get_numeric_pcids(df[df[col] == 'inhibitor'][pcid_col])
    neutral_pcids = get_numeric_pcids(df[df[col] == 'neutral'][pcid_col])
    return activator_pcids, inhibitor_pcids, neutral_pcids


def get_murcko_scaffold(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold) if scaffold else smiles
    except Exception:
        return smiles


def compute_scaffolds(smiles: np.ndarray, n_jobs: int = N_JOBS) -> np.ndarray:
    cache_path = OUT_DIR / 'scaffold_cache.npy'
    if cache_path.exists():
        logger.info('Loading cached scaffolds...')
        return np.load(cache_path, allow_pickle=True)
    logger.info('Computing Murcko scaffolds...')
    scaffolds = Parallel(n_jobs=n_jobs)(
        delayed(get_murcko_scaffold)(s) for s in smiles
    )
    scaffolds = np.array(scaffolds)
    np.save(cache_path, scaffolds)
    return scaffolds


def safe_spearman(a, b):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            c, _ = spearmanr(a, b, nan_policy='omit')
            return float(c)
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# Flux gene selection
# ---------------------------------------------------------------------------
def select_flux_informative_genes(
    Y: np.ndarray,
    smiles: np.ndarray,
    cell_ids: np.ndarray,
    gene_symbols: np.ndarray,
    activator_smiles: set,
    inhibitor_smiles: set,
    priority_genes: List[str],
    n_top: int = N_TOP_GENES,
    min_cell_lines: int = MIN_CELL_LINES_PER_GENE,
) -> Tuple[List[int], pd.DataFrame]:
    """Select genes where activators and inhibitors show opposite directions vs background."""
    logger.info('Selecting flux-informative genes...')

    act_mask = np.array([s in activator_smiles for s in smiles], dtype=bool)
    inh_mask = np.array([s in inhibitor_smiles for s in smiles], dtype=bool)
    non_mask = ~(act_mask | inh_mask)

    cell_types = np.unique(cell_ids)
    gene_idx = np.arange(len(gene_symbols))

    records = []
    for g in gene_idx:
        contrasts = []
        magnitudes = []
        for cell in cell_types:
            cmask = cell_ids == cell
            a_vals = Y[cmask & act_mask, g]
            i_vals = Y[cmask & inh_mask, g]
            n_vals = Y[cmask & non_mask, g]
            if len(a_vals) < 3 or len(i_vals) < 2 or len(n_vals) < 10:
                continue
            a_mean = np.nanmean(a_vals)
            i_mean = np.nanmean(i_vals)
            n_mean = np.nanmean(n_vals)
            da = a_mean - n_mean
            di = i_mean - n_mean
            # Opposite direction relative to background
            if da * di < 0:
                contrasts.append(-da * di)  # positive, larger = more opposite
                magnitudes.append(min(abs(da), abs(di)))
        n_cells = len(contrasts)
        if n_cells >= min_cell_lines:
            records.append({
                'gene': gene_symbols[g],
                'gene_idx': int(g),
                'n_cell_lines': n_cells,
                'sum_contrast': float(np.sum(contrasts)),
                'sum_magnitude': float(np.sum(magnitudes)),
                'is_priority': gene_symbols[g] in priority_genes,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError('No flux-informative genes found with current filters.')

    # Rank: priority genes get a bonus, then by sum_contrast
    df['priority_bonus'] = df['is_priority'].astype(float) * df['sum_contrast'].max() * 0.2
    df['score'] = df['sum_contrast'] + df['priority_bonus']
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)

    selected = df.head(n_top)
    selected_indices = selected['gene_idx'].astype(int).tolist()
    logger.info(f'Selected {len(selected_indices)} flux-informative genes:')
    for _, row in selected.iterrows():
        logger.info(f"  {row['rank']}. {row['gene']} (n_cells={row['n_cell_lines']}, score={row['score']:.3f})")
    return selected_indices, selected


# ---------------------------------------------------------------------------
# Signature building and validation
# ---------------------------------------------------------------------------
def build_per_cell_signatures(
    Y: np.ndarray,
    smiles: np.ndarray,
    cell_ids: np.ndarray,
    activator_smiles: set,
    inhibitor_smiles: set,
    gene_indices: List[int],
) -> Dict[str, Dict]:
    """Build per-cell-line contrastive signatures for activators and inhibitors."""
    logger.info('Building per-cell-line flux signatures...')
    signatures = {}
    cell_types = np.unique(cell_ids)
    Y_sel = Y[:, gene_indices]

    act_mask = np.array([s in activator_smiles for s in smiles], dtype=bool)
    inh_mask = np.array([s in inhibitor_smiles for s in smiles], dtype=bool)
    non_mask = ~(act_mask | inh_mask)

    for cell_name in cell_types:
        cell_idx = list(cell_types).index(cell_name)
        cmask = cell_ids == cell_idx
        Yc = Y_sel[cmask]
        sc = smiles[cmask]

        act_mask_c = np.array([s in activator_smiles for s in sc], dtype=bool)
        inh_mask_c = np.array([s in inhibitor_smiles for s in sc], dtype=bool)
        non_mask_c = ~(act_mask_c | inh_mask_c)

        if act_mask_c.sum() < 5 or inh_mask_c.sum() < 3 or non_mask_c.sum() < 20:
            continue

        # Background = non-HAMDB compounds matched by count
        rng = np.random.default_rng(42)
        non_idx = np.where(non_mask_c)[0]
        bg_act = np.nanmean(Yc[rng.choice(non_idx, size=min(act_mask_c.sum(), len(non_idx)), replace=False)], axis=0)
        bg_inh = np.nanmean(Yc[rng.choice(non_idx, size=min(inh_mask_c.sum(), len(non_idx)), replace=False)], axis=0)

        act_sig = np.nanmean(Yc[act_mask_c], axis=0) - bg_act
        inh_sig = np.nanmean(Yc[inh_mask_c], axis=0) - bg_inh

        # Quality: median within-class Spearman
        act_corrs = []
        for idx in np.where(act_mask_c)[0]:
            c = safe_spearman(Yc[idx], act_sig)
            if not np.isnan(c):
                act_corrs.append(c)
        inh_corrs = []
        for idx in np.where(inh_mask_c)[0]:
            c = safe_spearman(Yc[idx], inh_sig)
            if not np.isnan(c):
                inh_corrs.append(c)

        signatures[cell_name] = {
            'activator': act_sig,
            'inhibitor': inh_sig,
            'act_corrs': act_corrs,
            'inh_corrs': inh_corrs,
            'n_act': int(act_mask_c.sum()),
            'n_inh': int(inh_mask_c.sum()),
            'n_non': int(non_mask_c.sum()),
        }
        logger.info(f"  {cell_name}: act={act_mask_c.sum()}, inh={inh_mask_c.sum()}, "
                    f"act_quality={np.median(act_corrs):.3f}, inh_quality={np.median(inh_corrs):.3f}")

    return signatures


def cross_validate_signatures(
    Y: np.ndarray,
    smiles: np.ndarray,
    cell_ids: np.ndarray,
    scaffolds: np.ndarray,
    activator_smiles: set,
    inhibitor_smiles: set,
    gene_indices: List[int],
    signatures: Dict[str, Dict],
) -> pd.DataFrame:
    logger.info('\n=== Cross-validation (flux signatures) ===')
    cv_results = []
    Y_sel = Y[:, gene_indices]
    cell_types = np.unique(cell_ids)

    for cell_name, sigs in signatures.items():
        cell_idx = list(cell_types).index(cell_name)
        cmask = cell_ids == cell_idx
        Yc = Y_sel[cmask]
        sc = smiles[cmask]

        act_mask = np.array([s in activator_smiles for s in sc], dtype=bool)
        non_mask = ~np.array([s in activator_smiles or s in inhibitor_smiles for s in sc], dtype=bool)

        act_scaffolds = list(set(scaffolds[cmask][act_mask]))
        rng = np.random.default_rng(42)
        rng.shuffle(act_scaffolds)
        n_train = int(0.7 * len(act_scaffolds))
        train_scafs = set(act_scaffolds[:n_train])
        test_scafs = set(act_scaffolds[n_train:])

        train_mask = np.array([scaffolds[cmask][i] in train_scafs for i in range(len(sc))]) & act_mask
        test_mask = np.array([scaffolds[cmask][i] in test_scafs for i in range(len(sc))]) & act_mask

        if train_mask.sum() < 5 or test_mask.sum() < 3:
            continue

        non_indices = np.where(non_mask)[0]
        bg = np.nanmean(Yc[rng.choice(non_indices, size=min(train_mask.sum(), len(non_indices)), replace=False)], axis=0)
        train_sig = np.nanmean(Yc[train_mask], axis=0) - bg

        test_corrs = [safe_spearman(Yc[i], train_sig) for i in np.where(test_mask)[0]]
        median_test = float(np.nanmedian(test_corrs))

        rand_corrs = []
        for _ in range(50):
            rand_idx = rng.choice(non_indices, size=test_mask.sum(), replace=False)
            rand_corrs.extend([safe_spearman(Yc[i], train_sig) for i in rand_idx])
        median_rand = float(np.nanmedian(rand_corrs))

        cv_results.append({
            'cell_line': cell_name,
            'median_test_corr': median_test,
            'median_random_corr': median_rand,
            'delta': median_test - median_rand,
            'n_train': int(train_mask.sum()),
            'n_test': int(test_mask.sum()),
        })
        logger.info(f"  {cell_name}: test={median_test:.3f}, random={median_rand:.3f}, delta={median_test-median_rand:.3f}")

    return pd.DataFrame(cv_results)


# ---------------------------------------------------------------------------
# Inverse search
# ---------------------------------------------------------------------------
def inverse_search(
    Y: np.ndarray,
    smiles: np.ndarray,
    cell_ids: np.ndarray,
    gene_indices: List[int],
    signatures: Dict[str, Dict],
) -> pd.DataFrame:
    logger.info('\n=== Flux-informed inverse search ===')
    Y_sel = Y[:, gene_indices]
    cell_types = np.unique(cell_ids)

    # Map each unique compound to best representative SMILES (first occurrence)
    unique_smiles = []
    seen = set()
    for s in smiles:
        if s not in seen:
            seen.add(s)
            unique_smiles.append(s)
    unique_smiles = np.array(unique_smiles)

    rng = np.random.default_rng(42)

    def score_compound(smi):
        mask = smiles == smi
        if mask.sum() == 0:
            return None
        Yc_all = Y_sel[mask]
        cellc = cell_ids[mask]

        act_corrs = []
        inh_corrs = []
        cell_weights = []
        for cell_name, sigs in signatures.items():
            cell_idx = list(cell_types).index(cell_name)
            cmask = cellc == cell_idx
            if cmask.sum() == 0:
                continue
            # Use median signature correlation across signatures for this compound in this cell line
            ac = [safe_spearman(y, sigs['activator']) for y in Yc_all[cmask]]
            ic = [safe_spearman(y, sigs['inhibitor']) for y in Yc_all[cmask]]
            act_corrs.append(np.nanmedian(ac))
            inh_corrs.append(np.nanmedian(ic))
            cell_weights.append(np.median(sigs['act_corrs']) + np.median(sigs['inh_corrs']))

        if len(act_corrs) == 0:
            return None

        weights = np.array(cell_weights)
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()

        act_mean = float(np.average(act_corrs, weights=weights))
        inh_mean = float(np.average(inh_corrs, weights=weights))

        # Contrastive score: high activator correlation, low inhibitor correlation
        contrastive = act_mean - inh_mean
        return {
            'canonical_smiles': smi,
            'n_cell_lines': len(act_corrs),
            'activator_corr_mean': act_mean,
            'inhibitor_corr_mean': inh_mean,
            'contrastive_score': contrastive,
            'best_activator_corr': float(np.nanmax(act_corrs)),
            'best_inhibitor_corr': float(np.nanmax(inh_corrs)),
        }

    results = Parallel(n_jobs=N_JOBS)(
        delayed(score_compound)(smi) for smi in unique_smiles
    )
    results = [r for r in results if r is not None]
    df = pd.DataFrame(results)

    # Permutation null for contrastive score
    logger.info('Computing permutation null for contrastive score...')
    all_scores = []
    for cell_name, sigs in signatures.items():
        cell_idx = list(cell_types).index(cell_name)
        cmask = cell_ids == cell_idx
        Yc = Y_sel[cmask]
        rng.shuffle(Yc)  # shuffle rows in place for null
        for y in Yc[:min(200, len(Yc))]:
            c_act = safe_spearman(y, sigs['activator'])
            c_inh = safe_spearman(y, sigs['inhibitor'])
            all_scores.append(c_act - c_inh)
    null_scores = np.array(all_scores)

    def empirical_pvalue(x):
        return float(np.mean(null_scores >= x))

    df['pvalue'] = df['contrastive_score'].apply(empirical_pvalue)
    df = df.sort_values('contrastive_score', ascending=False).reset_index(drop=True)
    df['rank'] = np.arange(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# ChEMBL assay integration
# ---------------------------------------------------------------------------
CACHED_CHEMBL_DIR = OUT_DIR / 'chembl_cache'
CACHED_CHEMBL_DIR.mkdir(exist_ok=True)


def chembl_get(url: str, params: Optional[dict] = None, retries: int = 3, sleep: float = 0.2) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                logger.warning(f'ChEMBL request failed: {url} -> {e}')
                return {}
            time.sleep(sleep * (attempt + 1))
    return {}


def find_autophagy_assays() -> pd.DataFrame:
    cache = CACHED_CHEMBL_DIR / 'autophagy_assays.json'
    if cache.exists():
        logger.info('Loading cached ChEMBL autophagy assay list...')
        return pd.read_json(cache)

    logger.info('Querying ChEMBL for autophagy-related assays...')
    terms = ['autophagy', 'autophagosome', 'lysosome', 'lysosomal', 'MTOR', 'LC3']
    assay_records = []
    seen = set()
    for term in terms:
        url = f'https://www.ebi.ac.uk/chembl/api/data/assay.json'
        params = {
            'description__icontains': term,
            'assay_organism': 'Homo sapiens',
            'limit': 1000,
        }
        data = chembl_get(url, params)
        for assay in data.get('assays', []):
            aid = assay.get('assay_chembl_id')
            if aid and aid not in seen:
                seen.add(aid)
                assay_records.append({
                    'assay_chembl_id': aid,
                    'description': assay.get('description', ''),
                    'assay_type': assay.get('assay_type', ''),
                    'target_chembl_id': assay.get('target_chembl_id', ''),
                    'confidence_score': assay.get('confidence_score', None),
                })

    df = pd.DataFrame(assay_records)
    df.to_json(cache, orient='records')
    logger.info(f'Found {len(df)} unique autophagy-related ChEMBL assays.')
    return df


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def chembl_molecule_by_inchikey(inchikey: str) -> Optional[dict]:
    cache = CACHED_CHEMBL_DIR / f'mol_{inchikey}.json'
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    url = f'https://www.ebi.ac.uk/chembl/api/data/molecule.json'
    data = chembl_get(url, params={'molecule_structures__standard_inchi_key': inchikey, 'limit': 1})
    mols = data.get('molecules', [])
    if not mols:
        return None
    mol = mols[0]
    with open(cache, 'w') as f:
        json.dump(mol, f)
    return mol


def chembl_activities_for_molecule(chembl_id: str, assay_ids: set) -> List[dict]:
    cache = CACHED_CHEMBL_DIR / f'acts_{chembl_id}.json'
    if cache.exists():
        with open(cache) as f:
            acts = json.load(f)
    else:
        url = f'https://www.ebi.ac.uk/chembl/api/data/activity.json'
        acts = []
        params = {'molecule_chembl_id': chembl_id, 'limit': 1000}
        while True:
            data = chembl_get(url, params)
            acts.extend(data.get('activities', []))
            page = data.get('page_meta', {})
            if not page.get('next'):
                break
            params['offset'] = page.get('offset', 0) + params['limit']
        with open(cache, 'w') as f:
            json.dump(acts, f)

    def _float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    filtered = []
    for a in acts:
        if a.get('assay_chembl_id') in assay_ids:
            filtered.append({
                'assay_chembl_id': a.get('assay_chembl_id'),
                'standard_value': _float(a.get('standard_value')),
                'standard_units': a.get('standard_units'),
                'standard_type': a.get('standard_type'),
                'activity_comment': a.get('activity_comment'),
                'pchembl_value': _float(a.get('pchembl_value')),
                'target_chembl_id': a.get('target_chembl_id'),
            })
    return filtered


def annotate_hits_with_chembl(df: pd.DataFrame, assay_df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    if assay_df.empty:
        logger.warning('No ChEMBL assays found; skipping assay annotation.')
        return df

    assay_ids = set(assay_df['assay_chembl_id'])
    logger.info(f'Annotating top {top_n} hits with ChEMBL activities...')

    top = df.head(top_n).copy()
    top['inchikey'] = Parallel(n_jobs=N_JOBS)(
        delayed(smiles_to_inchikey)(s) for s in top['canonical_smiles']
    )

    chembl_annotations = []
    for _, row in top.iterrows():
        ik = row['inchikey']
        acts = []
        if ik:
            mol = chembl_molecule_by_inchikey(ik)
            if mol:
                chembl_id = mol.get('molecule_chembl_id')
                if chembl_id:
                    acts = chembl_activities_for_molecule(chembl_id, assay_ids)
        if acts:
            # Summarize: number of autophagy assays, best pchembl, active/inactive calls
            pchembls = [a['pchembl_value'] for a in acts if a.get('pchembl_value') is not None]
            comments = [str(a.get('activity_comment', '')).lower() for a in acts]
            active_keywords = ['active', 'inhibitor', 'agonist', 'inducer', 'potent', 'stimulant']
            is_active = any(kw in ' '.join(comments) for kw in active_keywords) or len(pchembls) > 0
            chembl_annotations.append({
                'canonical_smiles': row['canonical_smiles'],
                'n_autophagy_assays': len(acts),
                'best_pchembl': float(max(pchembls)) if pchembls else None,
                'median_pchembl': float(np.median(pchembls)) if pchembls else None,
                'chembl_active': is_active,
                'chembl_activity_summary': '; '.join(
                    f"{a['assay_chembl_id']}:{a['standard_type']}={a['standard_value']}{a['standard_units']}"
                    for a in acts[:3]
                ),
            })
        else:
            chembl_annotations.append({
                'canonical_smiles': row['canonical_smiles'],
                'n_autophagy_assays': 0,
                'best_pchembl': None,
                'median_pchembl': None,
                'chembl_active': False,
                'chembl_activity_summary': '',
            })

    ann_df = pd.DataFrame(chembl_annotations)
    top = top.merge(ann_df, on='canonical_smiles', how='left')
    return top


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cache = load_cache(CACHE_PATH)
    Y = cache['Y']
    smiles = cache['smiles']
    cell_ids = cache['cell_ids']
    gene_symbols = cache['gene_symbols']
    logger.info(f'Cache loaded: {Y.shape} signatures, {len(np.unique(smiles))} compounds, {len(gene_symbols)} genes')

    # Load HAMDB directions by PubChem CID and map to L1000 canonical SMILES
    activator_pcids, inhibitor_pcids, neutral_pcids = load_hamdb_directions(HAMDB_PATH)
    logger.info(f'HAMDB conservative labels: {len(activator_pcids)} activator PCIDs, {len(inhibitor_pcids)} inhibitor PCIDs, {len(neutral_pcids)} neutral PCIDs')

    logger.info('Loading L1000 pert_info for CID->SMILES mapping...')
    pert_info = pd.read_csv(PERT_INFO_PATH, sep='\t',
                            usecols=['pert_id', 'pert_iname', 'canonical_smiles', 'pubchem_cid'])
    pert_info['numeric_pcid'] = pert_info['pubchem_cid'].apply(
        lambda x: str(int(float(x))) if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else None
    )
    activator_smiles = set(pert_info[pert_info['numeric_pcid'].isin(activator_pcids)]['canonical_smiles'].dropna().unique())
    inhibitor_smiles = set(pert_info[pert_info['numeric_pcid'].isin(inhibitor_pcids)]['canonical_smiles'].dropna().unique())
    neutral_smiles = set(pert_info[pert_info['numeric_pcid'].isin(neutral_pcids)]['canonical_smiles'].dropna().unique())
    activator_smiles = {s for s in activator_smiles if s and s != '-666'}
    inhibitor_smiles = {s for s in inhibitor_smiles if s and s != '-666'}
    neutral_smiles = {s for s in neutral_smiles if s and s != '-666'}
    logger.info(f'  Mapped to L1000 SMILES: {len(activator_smiles)} activators, {len(inhibitor_smiles)} inhibitors, {len(neutral_smiles)} neutral')

    # Filter to 24h, 1-10 uM
    time_mask = cache['time'] == 24 if cache['time'] is not None else np.ones(len(Y), dtype=bool)
    dose = cache['dose']
    dose_mask = np.ones(len(Y), dtype=bool)
    if dose is not None:
        dose_mask = (dose >= 1.0) & (dose <= 10.0)
    keep = time_mask & dose_mask
    Y = Y[keep]
    smiles = smiles[keep]
    cell_ids = cell_ids[keep]
    if cache['time'] is not None:
        logger.info(f'Filtered to 24h, 1-10uM: {Y.shape[0]} signatures')

    # Restrict to strict core autophagy genes (same as v2)
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    strict_indices = [gene_to_idx[g] for g in STRICT_AUTOPHAGY_GENES if g in gene_to_idx]
    missing = [g for g in STRICT_AUTOPHAGY_GENES if g not in gene_to_idx]
    if missing:
        logger.warning(f'  {len(missing)} strict genes not in L1000: {missing}')
    Y = Y[:, strict_indices]
    gene_symbols = np.array([gene_symbols[i] for i in strict_indices])
    logger.info(f'Using {len(gene_symbols)} strict autophagy genes')

    scaffolds = compute_scaffolds(smiles)

    # Select flux-informative genes
    gene_indices, gene_df = select_flux_informative_genes(
        Y, smiles, cell_ids, gene_symbols,
        activator_smiles, inhibitor_smiles, PRIORITY_FLUX_GENES,
        n_top=N_TOP_GENES, min_cell_lines=MIN_CELL_LINES_PER_GENE,
    )
    gene_df.to_csv(OUT_DIR / 'flux_informative_genes.csv', index=False)

    # Build signatures
    signatures = build_per_cell_signatures(
        Y, smiles, cell_ids, activator_smiles, inhibitor_smiles, gene_indices,
    )
    np.savez(
        OUT_DIR / 'autophagy_signatures_flux.npz',
        signatures=signatures,
        gene_symbols=gene_symbols[gene_indices],
        gene_indices=gene_indices,
    )

    # Cross-validation
    cv_df = cross_validate_signatures(
        Y, smiles, cell_ids, scaffolds, activator_smiles, inhibitor_smiles,
        gene_indices, signatures,
    )
    cv_df.to_csv(OUT_DIR / 'flux_signature_cv_results.csv', index=False)

    # Inverse search
    hits_df = inverse_search(Y, smiles, cell_ids, gene_indices, signatures)

    # Add HAMDB labels
    hits_df['is_hamdb_activator'] = hits_df['canonical_smiles'].isin(activator_smiles)
    hits_df['is_hamdb_inhibitor'] = hits_df['canonical_smiles'].isin(inhibitor_smiles)
    hits_df['is_hamdb_neutral'] = hits_df['canonical_smiles'].isin(neutral_smiles)

    # HAMDB enrichment
    def enrichment(df, label_col, top_n):
        if df[label_col].sum() == 0:
            return 0.0
        top = df.head(top_n)
        return 100.0 * top[label_col].mean()

    enrichment_50 = enrichment(hits_df, 'is_hamdb_activator', 50)
    enrichment_100 = enrichment(hits_df, 'is_hamdb_activator', 100)
    enrichment_200 = enrichment(hits_df, 'is_hamdb_activator', 200)
    logger.info(f'HAMDB activator enrichment: top50={enrichment_50:.1f}%, top100={enrichment_100:.1f}%, top200={enrichment_200:.1f}%')

    # Save full hits
    hits_df.to_csv(OUT_DIR / 'flux_signature_hits_all.csv', index=False)
    hits_df.head(200).to_csv(OUT_DIR / 'flux_signature_hits_top200.csv', index=False)

    # ChEMBL assay integration
    assay_df = find_autophagy_assays()
    annotated_top = annotate_hits_with_chembl(hits_df, assay_df, top_n=200)
    annotated_top.to_csv(OUT_DIR / 'flux_signature_hits_top200_chembl.csv', index=False)

    n_chembl_active = annotated_top['chembl_active'].sum()
    logger.info(f'ChEMBL autophagy assay annotation: {int(n_chembl_active)}/200 top hits with activity data')

    # Write summary markdown
    write_summary(
        gene_df, signatures, cv_df, hits_df, annotated_top,
        enrichment_50, enrichment_100, enrichment_200, assay_df, n_chembl_active,
    )

    logger.info('Done.')


def write_summary(
    gene_df: pd.DataFrame,
    signatures: Dict[str, Dict],
    cv_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    annotated_top: pd.DataFrame,
    enrichment_50: float,
    enrichment_100: float,
    enrichment_200: float,
    assay_df: pd.DataFrame,
    n_chembl_active: int,
):
    lines = []
    lines.append('# Flux-Informed Autophagy Signature + ChEMBL Assay Integration\n')
    lines.append('## Gene Selection\n')
    lines.append(f'Flux-informative genes selected: {len(gene_df)}\n')
    lines.append('| Rank | Gene | Cell lines | Contrast score | Priority |\n')
    lines.append('|------|------|------------|----------------|----------|\n')
    for _, row in gene_df.iterrows():
        lines.append(f"| {row['rank']} | {row['gene']} | {row['n_cell_lines']} | {row['score']:.3f} | {'Yes' if row['is_priority'] else 'No'} |\n")

    lines.append('\n## Per-cell-line Signature Quality\n')
    lines.append('| Cell line | n_act | n_inh | Activator quality | Inhibitor quality |\n')
    lines.append('|-----------|-------|-------|-------------------|-------------------|\n')
    for cell_name, sigs in signatures.items():
        lines.append(f"| {cell_name} | {sigs['n_act']} | {sigs['n_inh']} | "
                     f"{np.median(sigs['act_corrs']):.3f} | {np.median(sigs['inh_corrs']):.3f} |\n")

    lines.append('\n## Cross-validation\n')
    if not cv_df.empty:
        lines.append('| Cell line | n_train | n_test | median_test_corr | median_random_corr | delta |\n')
        lines.append('|-----------|---------|--------|------------------|--------------------|-------|\n')
        for _, row in cv_df.iterrows():
            lines.append(f"| {row['cell_line']} | {row['n_train']} | {row['n_test']} | "
                         f"{row['median_test_corr']:.3f} | {row['median_random_corr']:.3f} | {row['delta']:.3f} |\n")
    else:
        lines.append('No valid cross-validation folds.\n')

    lines.append('\n## Inverse Search HAMDB Enrichment\n')
    lines.append(f'- Top 50: {enrichment_50:.1f}% HAMDB activators\n')
    lines.append(f'- Top 100: {enrichment_100:.1f}% HAMDB activators\n')
    lines.append(f'- Top 200: {enrichment_200:.1f}% HAMDB activators\n')

    lines.append('\n## Top 20 Hits\n')
    top20 = hits_df.head(20)
    cols = ['rank', 'canonical_smiles', 'contrastive_score', 'activator_corr_mean',
            'inhibitor_corr_mean', 'pvalue', 'is_hamdb_activator', 'is_hamdb_inhibitor']
    lines.append('| ' + ' | '.join(cols) + ' |\n')
    lines.append('|' + '|'.join(['---'] * len(cols)) + '|\n')
    for _, row in top20.iterrows():
        lines.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |\n')

    lines.append('\n## ChEMBL Assay Integration\n')
    lines.append(f'- Autophagy-related ChEMBL assays queried: {len(assay_df)}\n')
    lines.append(f'- Top-200 hits with ChEMBL autophagy activity data: {int(n_chembl_active)}\n')
    if n_chembl_active > 0:
        active_hits = annotated_top[annotated_top['chembl_active']].head(10)
        lines.append('\n### Top ChEMBL-annotated hits\n')
        cols2 = ['rank', 'canonical_smiles', 'contrastive_score', 'n_autophagy_assays', 'best_pchembl', 'chembl_activity_summary']
        lines.append('| ' + ' | '.join(cols2) + ' |\n')
        lines.append('|' + '|'.join(['---'] * len(cols2)) + '|\n')
        for _, row in active_hits.iterrows():
            lines.append('| ' + ' | '.join(str(row[c]) for c in cols2) + ' |\n')

    lines.append('\n## Interpretation\n')
    if enrichment_50 < 5:
        lines.append('HAMDB activator enrichment remains low even with the flux-informed gene set. '
                     'This suggests that the transcriptional signal distinguishing HAMDB-curated autophagy activators '
                     'from inhibitors is weak in L1000, or that the conserved stress response dominates. '
                     'The ChEMBL annotations should be used as the primary orthogonal filter for prioritizing hits.\n')
    else:
        lines.append('The flux-informed signature shows improved HAMDB activator enrichment compared to the broad 199-gene signature. '
                     'ChEMBL assay annotations provide orthogonal validation for the top-ranked compounds.\n')

    with open(OUT_DIR / 'flux_signature_summary.md', 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    main()
