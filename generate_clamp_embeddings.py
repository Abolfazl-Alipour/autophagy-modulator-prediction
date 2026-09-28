"""
Generate CLAMP embeddings for all unique compounds in L1000 full cache.
Outputs: clamp_embeddings_l1000_full.pt (dict: smiles -> 768-dim tensor)
"""
import sys
import logging
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('clamp_embed')


def main():
    logger.info("Loading full cache to get unique SMILES...")
    d = np.load('diag_cache_v9_morgan_full.npz', allow_pickle=True)
    smiles = d['smiles']
    unique_smiles = sorted(list(set(smiles.tolist())))
    logger.info(f"  {len(unique_smiles)} unique SMILES")

    logger.info("Loading CLAMP model on CPU...")
    from clamp.models.pretrained import PretrainedCLAMP
    model = PretrainedCLAMP(device='cpu')
    model.eval()

    logger.info("Generating CLAMP embeddings in batches...")
    embeddings = {}
    batch_size = 2048
    for i in tqdm(range(0, len(unique_smiles), batch_size)):
        batch = unique_smiles[i:i+batch_size]
        with torch.no_grad():
            emb = model.encode_smiles(batch).cpu()
        for sm, e in zip(batch, emb):
            embeddings[sm] = e

    logger.info(f"  Generated {len(embeddings)} embeddings")

    out_path = 'clamp_embeddings_l1000_full.pt'
    torch.save(embeddings, out_path)
    logger.info(f"Saved to {out_path}")


if __name__ == '__main__':
    main()