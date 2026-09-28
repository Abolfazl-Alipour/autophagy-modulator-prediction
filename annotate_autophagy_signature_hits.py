"""
Annotate all L1000 compounds by their match to the activator and inhibitor autophagy signatures,
and generate a clean report listing every compound.

Inputs:
  - autophagy_signature_hits_all.csv
  - L1000 pert_info

Outputs:
  - autophagy_signature_hits_all_annotated.csv
  - all_compounds_autophagy_signature_match.md
"""
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('annotate_signature_hits')


def main():
    logger.info("Loading signature hits...")
    df = pd.read_csv('autophagy_signature_hits_all.csv')
    logger.info(f"  {len(df)} rows")

    # Exclude control/unannotated SMILES
    df = df[df['canonical_smiles'] != '-666'].copy()
    logger.info(f"  {len(df)} compounds after excluding -666 controls")

    logger.info("Loading L1000 pert_info...")
    pert_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz', sep='\t',
                            usecols=['pert_id', 'pert_iname', 'canonical_smiles', 'pubchem_cid'])

    # Build SMILES -> metadata mapping (first occurrence)
    smiles_to_name = {}
    smiles_to_pcid = {}
    for _, row in pert_info.iterrows():
        sm = row['canonical_smiles']
        if pd.isna(sm) or sm == '-666':
            continue
        if sm not in smiles_to_name:
            smiles_to_name[sm] = row['pert_iname']
            smiles_to_pcid[sm] = row['pubchem_cid']

    # Add pubchem_cid if not present; fill missing pert_iname
    if 'pubchem_cid' not in df.columns:
        df.insert(0, 'pubchem_cid', df['canonical_smiles'].map(smiles_to_pcid))
    if 'pert_iname' not in df.columns:
        df.insert(0, 'pert_iname', df['canonical_smiles'].map(smiles_to_name))
    else:
        # Fill NaN pert_inames
        missing_mask = df['pert_iname'].isna()
        df.loc[missing_mask, 'pert_iname'] = df.loc[missing_mask, 'canonical_smiles'].map(smiles_to_name)

    # Sort by activator correlation
    df = df.sort_values('activator_corr_mean', ascending=False).reset_index(drop=True)

    # Save annotated full CSV
    df.to_csv('autophagy_signature_hits_all_annotated.csv', index=False)
    logger.info("Saved autophagy_signature_hits_all_annotated.csv")

    # Generate threshold summary
    thresholds = [0.5, 0.45, 0.4, 0.35, 0.3]
    summary_lines = []
    summary_lines.append("# All LINCS L1000 Compounds Ranked by Autophagy Signature Match\n")
    summary_lines.append("**Method:** Spearman correlation between each compound's 24h transcriptomic profile (199 autophagy genes) and the mean HAMDB-curated activator/inhibitor signatures.\n")
    summary_lines.append(f"**Total compounds ranked:** {len(df)}\n")

    summary_lines.append("## Compounds above correlation thresholds (activator signature)\n")
    summary_lines.append("| Threshold | Number of compounds |")
    summary_lines.append("|-----------|---------------------|")
    for t in thresholds:
        n = (df['activator_corr_mean'] >= t).sum()
        summary_lines.append(f"| {t:.2f} | {n} |")
    summary_lines.append("")

    summary_lines.append("## Compounds above correlation thresholds (inhibitor signature)\n")
    summary_lines.append("| Threshold | Number of compounds |")
    summary_lines.append("|-----------|---------------------|")
    for t in thresholds:
        n = (df['inhibitor_corr_mean'] >= t).sum()
        summary_lines.append(f"| {t:.2f} | {n} |")
    summary_lines.append("")

    # Top 100 activator-like compounds
    summary_lines.append("## Top 100 activator-signature matches\n")
    summary_lines.append("| Rank | pert_iname | pubchem_cid | n_24h_sigs | activator_corr_mean | inhibitor_corr_mean | is_HAMDB_activator | is_HAMDB_inhibitor |")
    summary_lines.append("|------|------------|-------------|------------|---------------------|---------------------|--------------------|--------------------|")
    for i, row in df.head(100).iterrows():
        pcid = str(row['pubchem_cid']) if pd.notna(row['pubchem_cid']) else ''
        summary_lines.append(
            f"| {i+1} | {row['pert_iname']} | {pcid} | {int(row['n_24h_sigs'])} | "
            f"{row['activator_corr_mean']:.4f} | {row['inhibitor_corr_mean']:.4f} | "
            f"{bool(row['is_hamdb_activator'])} | {bool(row['is_hamdb_inhibitor'])} |"
        )
    summary_lines.append("")

    # Top 100 inhibitor-like compounds
    df_inh = df.sort_values('inhibitor_corr_mean', ascending=False).reset_index(drop=True)
    summary_lines.append("## Top 100 inhibitor-signature matches\n")
    summary_lines.append("| Rank | pert_iname | pubchem_cid | n_24h_sigs | inhibitor_corr_mean | activator_corr_mean | is_HAMDB_activator | is_HAMDB_inhibitor |")
    summary_lines.append("|------|------------|-------------|------------|---------------------|---------------------|--------------------|--------------------|")
    for i, row in df_inh.head(100).iterrows():
        pcid = str(row['pubchem_cid']) if pd.notna(row['pubchem_cid']) else ''
        summary_lines.append(
            f"| {i+1} | {row['pert_iname']} | {pcid} | {int(row['n_24h_sigs'])} | "
            f"{row['inhibitor_corr_mean']:.4f} | {row['activator_corr_mean']:.4f} | "
            f"{bool(row['is_hamdb_activator'])} | {bool(row['is_hamdb_inhibitor'])} |"
        )
    summary_lines.append("")

    # Add note about full CSV
    summary_lines.append("## Full compound list\n")
    summary_lines.append("Every compound in LINCS L1000 is ranked in `autophagy_signature_hits_all_annotated.csv`. "
                         "This file contains all 20k+ compounds with their activator and inhibitor Spearman correlations, "
                         "HAMDB labels, and perturbagen names.\n")

    with open('all_compounds_autophagy_signature_match.md', 'w') as f:
        f.write('\n'.join(summary_lines))
    logger.info("Saved all_compounds_autophagy_signature_match.md")

    # Summary stats
    logger.info("\nSummary:")
    logger.info(f"  Total compounds: {len(df)}")
    logger.info(f"  Activator corr > 0.5: {(df['activator_corr_mean'] >= 0.5).sum()}")
    logger.info(f"  Activator corr > 0.4: {(df['activator_corr_mean'] >= 0.4).sum()}")
    logger.info(f"  Inhibitor corr > 0.5: {(df['inhibitor_corr_mean'] >= 0.5).sum()}")
    logger.info(f"  Inhibitor corr > 0.4: {(df['inhibitor_corr_mean'] >= 0.4).sum()}")


if __name__ == '__main__':
    main()
