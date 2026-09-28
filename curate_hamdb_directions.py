"""
Curate HAMDB compounds into autophagy effect directions:
  activator, inhibitor, neutral

Rules applied in order of priority:
1. Explicit HAMDB labels (Category contains 'Autophagy activator' / 'Autophagy inhibitor')
2. Plain Category 'Activator' / 'Inhibitor' (these are autophagy modulators in HAMDB context)
3. Target/mechanism-based rules for known autophagy regulators
4. Biological_Description text mining
5. Manual review flags for ambiguous cases

Output: hamdb_autophagy_directions.csv with direction and rationale.
"""
import re
import pandas as pd
import numpy as np
from collections import Counter


def classify_direction(row):
    """Classify one HAMDB row. Returns (direction, rationale)."""
    name = str(row.get('Chemcial_Name', '')).strip().lower()
    category = str(row.get('Category', '')).strip()
    pathway = str(row.get('Pathway', '')).strip()
    target = str(row.get('Target', '')).strip()
    desc = str(row.get('Biological_Description', '')).strip()

    text = ' | '.join([category, pathway, target, desc]).lower()

    # Manual overrides for well-known autophagy modulators that lack explicit HAMDB category
    manual_overrides = {
        # Activators
        'bafilomycin a1': ('inhibitor', 'manual: V-ATPase inhibitor, blocks autophagic flux'),
        'chloroquine': ('inhibitor', 'manual: lysosomal inhibitor, blocks autophagic degradation'),
        'hydroxychloroquine': ('inhibitor', 'manual: lysosomal inhibitor, blocks autophagic degradation'),
        'everolimus': ('activator', 'manual: mTOR inhibitor, induces autophagy'),
        'rapamycin (sirolimus)': ('activator', 'manual: mTOR inhibitor, induces autophagy'),
        'sirolimus': ('activator', 'manual: mTOR inhibitor, induces autophagy'),
        'metformin': ('activator', 'manual: AMPK activator, induces autophagy'),
        'lithium': ('activator', 'manual: inositol monophosphatase/GSK-3 inhibitor, induces autophagy'),
        'spermidine': ('activator', 'manual: polyamine, induces autophagy'),
        'y-27632': ('activator', 'manual: ROCK inhibitor, can induce autophagy'),
        'y 27632': ('activator', 'manual: ROCK inhibitor, can induce autophagy'),
        '5-fluorouracil': ('activator', 'manual: chemotherapeutic, commonly induces autophagy'),
        'arsenic trioxide': ('activator', 'manual: induces autophagy in cancer cells'),
        'atorvastatin': ('activator', 'manual: statin/HMG-CoA reductase inhibitor, induces autophagy'),
        'camptothecin': ('activator', 'manual: topoisomerase inhibitor, can induce autophagy'),
        'cisplatin': ('activator', 'manual: DNA damaging agent, commonly induces autophagy'),
        'cyclosporin a': ('activator', 'manual: calcineurin inhibitor, can induce autophagy'),
        'bleomycin': ('activator', 'manual: DNA damaging agent, can induce autophagy'),
        'melphalan': ('activator', 'manual: DNA alkylator, can induce autophagy'),
        'staurosporine': ('activator', 'manual: broad kinase inhibitor, induces autophagy/apoptosis'),
        'tretinoin': ('activator', 'manual: retinoid, can induce autophagy'),
        'oleic acid': ('activator', 'manual: fatty acid, can induce lipophagy'),
        'palmitic acid': ('activator', 'manual: fatty acid, can induce lipophagy'),
        'thioridazine': ('activator', 'manual: phenothiazine, reported to induce autophagy'),
        'promazine': ('activator', 'manual: phenothiazine, reported to induce autophagy'),
        'promethazine': ('activator', 'manual: phenothiazine, reported to induce autophagy'),
        'mesoridazine': ('activator', 'manual: phenothiazine, reported to induce autophagy'),
        # Inhibitors
        'epothilone b': ('inhibitor', 'manual: microtubule stabilizer, blocks autophagosome trafficking'),
        'vinblastine': ('inhibitor', 'manual: microtubule inhibitor, blocks autophagy'),
        'vincristine': ('inhibitor', 'manual: microtubule inhibitor, blocks autophagy'),
        'podophyllotoxin': ('inhibitor', 'manual: microtubule inhibitor, blocks autophagy'),
        'quinacrine': ('inhibitor', 'manual: lysosomotropic agent, inhibits autophagy'),
        'mefloquine': ('inhibitor', 'manual: lysosomotropic antimalarial, inhibits autophagy'),
    }
    if name in manual_overrides:
        return manual_overrides[name]

    # 1. Explicit autophagy labels
    if re.search(r'\bautophagy activator\b', category, re.I):
        return 'activator', 'explicit: Autophagy activator'
    if re.search(r'\bautophagy inhibitor\b', category, re.I):
        return 'inhibitor', 'explicit: Autophagy inhibitor'

    # 2. Plain Activator / Inhibitor in HAMDB context mean autophagy modulator
    if category.lower() == 'activator':
        return 'activator', 'HAMDB plain Activator (Autophagy in Pathway/Target)'
    if category.lower() == 'inhibitor':
        return 'inhibitor', 'HAMDB plain Inhibitor (Autophagy in Pathway/Target)'

    # 3. Description mentions autophagy explicitly
    if re.search(r'(activat|induc|promot|enhanc).*autophagy', desc, re.I):
        return 'activator', 'description: activates/induces autophagy'
    if re.search(r'(inhibit|block|suppress|impair).*autophagy', desc, re.I):
        return 'inhibitor', 'description: inhibits/blocks autophagy'

    # 4. Target/mechanism rules
    # Well-established autophagy activators
    activator_patterns = [
        (r'\bmtor inhibitor\b', 'mTOR inhibitor induces autophagy'),
        (r'\bampk activator\b', 'AMPK activation induces autophagy'),
        (r'\bpi3k inhibitor\b', 'PI3K inhibition induces autophagy'),
        (r'\bproteasome inhibitor\b', 'Proteasome inhibition induces ER stress/autophagy'),
        (r'\bhdac inhibitor\b', 'HDAC inhibition commonly induces autophagy'),
        (r'\bsirtuin inhibitor\b', 'Sirtuin inhibition (SIRT1/2) induces autophagy'),
        (r'\bmek inhibitor\b', 'MEK inhibition induces autophagy'),
        (r'\bbraf inhibitor\b', 'BRAF inhibition induces autophagy'),
        (r'\braf inhibitor\b', 'Raf inhibition induces autophagy'),
        (r'\begfr inhibitor\b', 'EGFR inhibition induces autophagy'),
        (r'\bher2 inhibitor\b', 'HER2 inhibition induces autophagy'),
        (r'\balk inhibitor\b', 'ALK inhibition induces autophagy'),
        (r'\bhsp90 inhibitor\b', 'HSP90 inhibition induces autophagy'),
        (r'\bp53 activator\b', 'p53 activation can induce autophagy'),
        (r'\bcamp activator\b', 'cAMP activation can induce autophagy'),
        (r'\batpase inhibitor\b', 'ATPase inhibition can induce autophagy'),
        (r'\bhmg-coa reductase inhibitor\b', 'Statin-induced autophagy'),
        (r'\bhif inhibitor\b', 'HIF stabilizer/PHD inhibitor induces hypoxia-like autophagy'),
        (r'\bpim inhibitor\b', 'Pim kinase inhibition can induce autophagy'),
        (r'\bros chemical\b', 'ROS induction triggers autophagy'),
        (r'\bcrm1 inhibitor\b', 'CRM1/nuclear export inhibition can induce autophagy'),
        (r'\bmdm2 antagonist\b', 'MDM2 antagonism activates p53 → autophagy'),
        (r'\bthioredoxin reductase inhibitor\b', 'Redox stress induces autophagy'),
        (r'\bgsk-3 inhibitor\b', 'GSK-3 inhibition can induce autophagy'),
        (r'\bpkci inhibitor\b', 'PKCι inhibition can induce autophagy'),
        (r'\bc-raf inhibitor\b', 'C-Raf inhibition induces autophagy'),
    ]
    for pattern, rationale in activator_patterns:
        if re.search(pattern, text):
            return 'activator', f'target rule: {rationale}'

    # Well-established autophagy inhibitors
    inhibitor_patterns = [
        (r'\bulk1 inhibitor\b', 'ULK1 inhibition blocks autophagy initiation'),
        (r'\bvps34 inhibitor\b', 'VPS34 inhibition blocks autophagy initiation'),
        (r'\bvps34\b', 'VPS34 (class III PI3K) inhibition blocks autophagy'),
        (r'\bcdc14b inhibitor\b', 'CDC14B inhibition blocks autophagy'),
        (r'\bp97 inhibitor\b', 'p97/VCP inhibition blocks autophagy'),
        (r'\bdub inhibitor\b', 'DUB inhibition blocks autophagy flux'),
        (r'\bautophag.*inhibit', 'Autophagosome inhibition'),
        (r'\blysosomal inhibitor\b', 'Lysosomal inhibition blocks autophagic degradation'),
        (r'\bautophagic flux inhibitor\b', 'Blocks autophagic flux'),
        (r'\bautophagic flux.*inhibit', 'Blocks autophagic flux'),
        (r'\baurora kinase inhibitor\b', 'Aurora kinase inhibition blocks autophagosome formation'),
        (r'\bcdk inhibitor\b', 'CDK inhibition blocks autophagy in many contexts'),
        (r'\blrrk2 inhibitor\b', 'LRRK2 inhibition impairs autophagy'),
        (r'\bdopamine receptor antagonist\b', 'Dopamine receptor antagonism commonly inhibits autophagy'),
        (r'\bdopamine receptor inhibitor\b', 'Dopamine receptor inhibition commonly inhibits autophagy'),
    ]
    for pattern, rationale in inhibitor_patterns:
        if re.search(pattern, text):
            return 'inhibitor', f'target rule: {rationale}'

    # 5. Ambiguous modulators that are in HAMDB but direction unclear
    # These affect autophagy-related targets but direction is context-dependent
    ambiguous_patterns = [
        r'\bcalcium channel inhibitor\b',
        r'\bcalcium channel antagonist\b',
        r'\bserine protease inhibitor\b',
        r'\bsodium channel inhibitor\b',
        r'\btnf-alpha',
        r'\be1 activating inhibitor\b',
        r'\bear domain inhibitor\b',
    ]
    for pattern in ambiguous_patterns:
        if re.search(pattern, text):
            return 'neutral', f'ambiguous target: {pattern.strip("\\\\b")} — direction not clear from annotation'

    # 6. Default: if in HAMDB but no clear direction, mark neutral
    return 'neutral', 'no clear autophagy direction in Category/Target/Description'


def main():
    hamdb = pd.read_csv('data/hamdb_chemicals.csv', encoding='latin-1')
    hamdb = hamdb.dropna(subset=['Canonical_SMILES']).copy()

    directions = []
    rationales = []
    for _, row in hamdb.iterrows():
        d, r = classify_direction(row)
        directions.append(d)
        rationales.append(r)

    hamdb['autophagy_direction'] = directions
    hamdb['rationale'] = rationales

    print('Curation summary:')
    print(Counter(directions))
    print()

    # Save curated output
    out_cols = ['Chemcial_Name', 'HAMDB_ID', 'CAS_Number', 'Pubchem_CID',
                'Canonical_SMILES', 'Category', 'Pathway', 'Target',
                'Biological_Description', 'autophagy_direction', 'rationale']
    out_cols = [c for c in out_cols if c in hamdb.columns]
    hamdb[out_cols].to_csv('hamdb_autophagy_directions.csv', index=False)
    print('Saved hamdb_autophagy_directions.csv')

    # Print breakdown by category for manual review
    print('\nBreakdown by direction within non-obvious categories:')
    sub = hamdb[~hamdb['Category'].isin(['Activator', 'Inhibitor', 'Autophagy activator', 'Autophagy inhibitor'])]
    print(pd.crosstab(sub['Category'], sub['autophagy_direction']).to_string())


if __name__ == '__main__':
    main()
