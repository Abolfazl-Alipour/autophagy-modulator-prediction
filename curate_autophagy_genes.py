"""
Curate comprehensive autophagy gene set from KEGG and GO,
then map to L1000 gene symbols.
"""
import requests
import time
import pandas as pd
import numpy as np


def fetch_kegg_pathway_genes(pathway_id):
    """Fetch gene list from KEGG pathway."""
    url = f"https://rest.kegg.jp/link/hsa/{pathway_id}"
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.strip().split('\n')
    hsa_ids = []
    for line in lines:
        if '\t' in line:
            _, hsa = line.split('\t')
            hsa_ids.append(hsa.replace('hsa:', ''))
    return hsa_ids


def hsa_to_symbols(hsa_ids):
    """Map KEGG hsa IDs to gene symbols via KEGG API."""
    symbols = {}
    # KEGG API accepts up to 10 IDs per request
    for i in range(0, len(hsa_ids), 10):
        batch = hsa_ids[i:i+10]
        url = "https://rest.kegg.jp/list/" + "+".join([f"hsa:{x}" for x in batch])
        r = requests.get(url)
        if r.status_code != 200:
            continue
        for line in r.text.strip().split('\n'):
            parts = line.split('\t')
            if len(parts) >= 2:
                hsa = parts[0].replace('hsa:', '')
                # Symbol is first word in description
                desc = parts[1]
                sym = desc.split(';')[0].split(',')[0].strip()
                symbols[hsa] = sym
        time.sleep(0.1)
    return symbols


def fetch_go_genes(go_id):
    """Fetch gene symbols for a GO term via QuickGO."""
    url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?goId={go_id}&taxonId=9606&limit=10000"
    r = requests.get(url, headers={"Accept": "application/json"})
    if r.status_code != 200:
        return []
    data = r.json()
    genes = set()
    for row in data.get('results', []):
        sym = row.get('symbol')
        if sym:
            genes.add(sym)
    return list(genes)


def main():
    # Load L1000 gene symbols
    d = np.load('diag_cache_v9_morgan_full.npz', allow_pickle=True)
    l1000_genes = set(d['gene_symbols'])
    print(f"L1000 total genes: {len(l1000_genes)}")

    # KEGG autophagy pathway
    print("Fetching KEGG autophagy pathway (hsa04140)...")
    hsa_ids = fetch_kegg_pathway_genes('hsa04140')
    print(f"  KEGG hsa IDs: {len(hsa_ids)}")
    kegg_symbols = hsa_to_symbols(hsa_ids)
    kegg_genes = set(kegg_symbols.values())
    print(f"  KEGG gene symbols: {len(kegg_genes)}")

    # GO autophagy terms
    go_terms = {
        'GO:0006914': 'autophagy',
        'GO:0016236': 'macroautophagy',
        'GO:0000422': 'autophagy of mitochondrion',
        'GO:0000423': 'mitophagy',
        'GO:0000421': 'autophagy of nucleus',
        'GO:0034045': 'phagophore assembly site',
        'GO:0000420': 'autophagosome maturation',
        'GO:0005764': 'lysosome',
        'GO:0007040': 'lysosome organization',
    }
    go_genes = set()
    for go_id in go_terms:
        if go_id.startswith('GO:'):
            print(f"Fetching {go_id}...")
            genes = fetch_go_genes(go_id)
            print(f"  {len(genes)} genes")
            go_genes.update(genes)
            time.sleep(0.2)

    # Literature core autophagy genes (manual curation)
    core_genes = {
        'ATG1', 'ULK1', 'ULK2', 'ULK3', 'ATG2A', 'ATG2B', 'ATG3', 'ATG4A', 'ATG4B', 'ATG4C', 'ATG4D',
        'ATG5', 'ATG7', 'ATG9A', 'ATG9B', 'ATG10', 'ATG12', 'ATG13', 'ATG14', 'ATG16L1', 'ATG16L2',
        'ATG101', 'BECN1', 'BECN2', 'PIK3C3', 'PIK3R4', 'NRBF2', 'AMBRA1', 'WIPI1', 'WIPI2', 'WIPI3', 'WIPI4',
        'MAP1LC3A', 'MAP1LC3B', 'MAP1LC3C', 'GABARAP', 'GABARAPL1', 'GABARAPL2', 'SQSTM1', 'NBR1',
        'OPTN', 'CALCOCO2', 'TAX1BP1', 'NDP52', 'TBK1', 'RPTOR', 'RICTOR', 'MTOR', 'DEPTOR', 'MLST8',
        'PRKAA1', 'PRKAA2', 'PRKAB1', 'PRKAB2', 'PRKAG1', 'PRKAG2', 'PRKAG3',
        'TFEB', 'TFE3', 'FOXO3', 'FOXO1', 'DDIT4', 'TSC1', 'TSC2', 'RHEB', 'EIF4EBP1', 'RPS6KB1',
        'LAMP1', 'LAMP2', 'CTSD', 'CTSL', 'CTSB', 'CTSS', 'CTSZ', 'HEXA', 'HEXB', 'GLA', 'GBA', 'GBA1',
        'NPC1', 'NPC2', 'SGSH', 'NAGLU', 'GNS', 'IDS', 'ARSA', 'GALNS', 'GLB1', 'FUCA1', 'PPT1', 'TPP1',
        'PINK1', 'PRKN', 'BNIP3', 'BNIP3L', 'FUNDC1', 'FKBP8', 'NIX', 'OPTN', 'TOMM20', 'TOMM22',
        'RTN3', 'FAM134B', 'ATL3', 'SEC62', 'CCPG1', 'TEX264', 'CALCOCO2', 'ATL3',
        'BCL2', 'BCL2L1', 'BAX', 'BAK1', 'CASP3', 'BECN1', 'HMGB1',
        'HSPA8', 'HSP90AA1', 'HSP90AB1', 'STUB1', 'BAG3', 'BAG1',
        'TRIM21', 'TRIM5', 'TRIM16', 'TRIM17', 'TRIM32', 'TRIM50',
        'RB1CC1', 'SEC23A', 'SEC24A', 'SEC24B', 'SEC24C', 'SEC24D',
        'TMEM173', 'CGAS', 'STING1',
        'KEAP1', 'NFE2L2', 'SQSTM1',
    }

    all_genes = kegg_genes | go_genes | core_genes
    # Normalize: uppercase
    all_genes = {g.upper().strip() for g in all_genes if g and isinstance(g, str)}

    # Filter to L1000 genes
    in_l1000 = sorted(all_genes & l1000_genes)
    not_in_l1000 = sorted(all_genes - l1000_genes)

    print(f"\nTotal curated autophagy genes: {len(all_genes)}")
    print(f"Present in L1000: {len(in_l1000)}")
    print(f"Missing from L1000: {len(not_in_l1000)}")

    # Save
    with open('autophagy_genes.txt', 'w') as f:
        for g in in_l1000:
            f.write(g + '\n')
    print(f"\nSaved {len(in_l1000)} autophagy genes to autophagy_genes.txt")

    # Save detailed report
    with open('autophagy_genes_report.txt', 'w') as f:
        f.write(f"KEGG hsa04140 genes: {len(kegg_genes)}\n")
        f.write(f"GO autophagy genes: {len(go_genes)}\n")
        f.write(f"Core literature genes: {len(core_genes)}\n")
        f.write(f"Total unique: {len(all_genes)}\n")
        f.write(f"In L1000: {len(in_l1000)}\n")
        f.write(f"Missing from L1000: {len(not_in_l1000)}\n")
        f.write("\nGenes in L1000:\n")
        f.write('\n'.join(in_l1000))


if __name__ == '__main__':
    main()