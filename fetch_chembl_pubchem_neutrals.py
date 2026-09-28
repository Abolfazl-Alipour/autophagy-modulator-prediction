#!/usr/bin/env python3
"""
Fetch ChEMBL and PubChem compounds for use as neutral negatives in the
3-class autophagy classifier.

Outputs:
  - chembl_neutral_candidates.csv
  - pubchem_neutral_candidates.csv
"""

import os
import re
import json
import time
import logging
import random
import requests
import urllib.parse
import pandas as pd
from rdkit import Chem
from pathlib import Path
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(exist_ok=True)

CHEMBL_OUT = OUT_DIR / 'chembl_neutral_candidates.csv'
PUBCHEM_OUT = OUT_DIR / 'pubchem_neutral_candidates.csv'

# ---------------------------------------------------------------------------
# ChEMBL fetching
# ---------------------------------------------------------------------------
def chembl_get(url, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning(f'ChEMBL request failed ({attempt+1}/{retries}): {url} -> {e}')
            time.sleep(2 * (attempt + 1))
    return None


def fetch_chembl_small_molecules(max_compounds=100_000, mw_max=800):
    """Fetch ChEMBL small molecules with molecular weight <= mw_max."""
    if CHEMBL_OUT.exists():
        logger.info(f'Loading cached ChEMBL candidates from {CHEMBL_OUT}')
        return pd.read_csv(CHEMBL_OUT)

    logger.info(f'Fetching up to {max_compounds} ChEMBL small molecules (MW <= {mw_max})...')
    records = []
    base_url = 'https://www.ebi.ac.uk/chembl/api/data/molecule.json'
    params = {
        'molecule_type': 'Small molecule',
        'mw_freebase__lte': mw_max,
        'limit': 1000,
    }
    offset = 0
    page = 0
    while offset < max_compounds:
        params['offset'] = offset
        data = chembl_get(base_url, params)
        if data is None:
            break
        molecules = data.get('molecules', [])
        if not molecules:
            break
        for m in molecules:
            structs = m.get('molecule_structures')
            if not structs:
                continue
            smi = structs.get('canonical_smiles')
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            records.append({
                'chembl_id': m.get('molecule_chembl_id'),
                'smiles': smi,
                'inchi_key': structs.get('standard_inchi_key'),
                'mw': m.get('molecule_properties', {}).get('mw_freebase'),
            })
        offset += len(molecules)
        page += 1
        if page % 10 == 0:
            logger.info(f'  ChEMBL fetched: {len(records)}')
        time.sleep(0.2)
        if len(records) >= max_compounds:
            break

    df = pd.DataFrame(records).drop_duplicates(subset=['inchi_key', 'smiles'])
    df.to_csv(CHEMBL_OUT, index=False)
    logger.info(f'Saved {len(df)} ChEMBL candidates to {CHEMBL_OUT}')
    return df


# ---------------------------------------------------------------------------
# PubChem fetching
# ---------------------------------------------------------------------------
def fetch_pubchem_random_sample(target=50_000, mw_max=800, cid_max=140_000_000):
    """Fetch a random sample of PubChem compounds by random CID probing."""
    if PUBCHEM_OUT.exists():
        logger.info(f'Loading cached PubChem candidates from {PUBCHEM_OUT}')
        return pd.read_csv(PUBCHEM_OUT)

    logger.info(f'Fetching random PubChem sample of ~{target} compounds...')
    records = []
    batch_size = 100
    attempts = 0
    max_attempts = target * 20  # probe many CIDs to get enough valid ones

    while len(records) < target and attempts < max_attempts:
        cids = [random.randint(1, cid_max) for _ in range(batch_size)]
        cid_str = ','.join(map(str, cids))
        url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/IsomericSMILES,MolecularWeight/JSON'
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            props = data.get('PropertyTable', {}).get('Properties', [])
            for p in props:
                smi = p.get('IsomericSMILES')
                mw = p.get('MolecularWeight')
                if not smi or not mw or mw > mw_max:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                records.append({
                    'pubchem_cid': p.get('CID'),
                    'smiles': smi,
                    'mw': mw,
                })
        except Exception as e:
            logger.warning(f'PubChem batch failed: {e}')
        attempts += batch_size
        if (attempts // batch_size) % 50 == 0:
            logger.info(f'  PubChem attempts: {attempts}, valid: {len(records)}')
        time.sleep(0.1)

    df = pd.DataFrame(records).drop_duplicates(subset=['smiles'])
    df.to_csv(PUBCHEM_OUT, index=False)
    logger.info(f'Saved {len(df)} PubChem candidates to {PUBCHEM_OUT}')
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    random.seed(42)
    fetch_chembl_small_molecules(max_compounds=100_000, mw_max=800)
    # PubChem is unreachable from this environment; ChEMBL will be the sole neutral source.
    logger.warning('PubChem fetching skipped: network unreachable in this environment.')
    logger.info('Done fetching neutral candidates.')


if __name__ == '__main__':
    main()
