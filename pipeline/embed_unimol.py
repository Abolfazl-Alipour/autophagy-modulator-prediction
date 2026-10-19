import os
import torch
import numpy as np
import pandas as pd
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('UniMolEmbedder')

class UniMolEmbedder:
    """
    Generates 3D-aware molecular embeddings using Uni-Mol.
    Pre-computes for all SMILES to avoid repeated conformer generation during training.
    """
    
    def __init__(self, model_name='unimolv1', model_size='84m'):
        try:
            from unimol_tools import UniMolRepr
            self.repr = UniMolRepr(model_name=model_name, model_size=model_size)
            self.dim = 512  # 84M model outputs 512-dim
            logger.info(f"Uni-Mol loaded: {model_name}/{model_size}, dim={self.dim}")
        except ImportError:
            logger.error("unimol_tools not installed. Run: pip install unimol-tools")
            raise
    
    def embed(self, smiles_list: List[str], batch_size=32) -> dict:
        """
        Generate embeddings for a list of SMILES strings.
        Returns a dictionary mapping smiles to tensor embeddings.
        """
        embeddings_dict = {}
        failed = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            
            valid_batch = []
            for smi in batch:
                if smi and isinstance(smi, str) and len(smi) > 3:
                    valid_batch.append(smi)
            
            if not valid_batch:
                continue
            
            try:
                emb = self.repr.get_repr(valid_batch)
                
                # In unimol-tools 0.1.5, get_repr might return a dict or a tuple.
                # Usually it returns a dict with 'cls_repr' for molecular representation.
                if isinstance(emb, dict) and 'cls_repr' in emb:
                    emb = emb['cls_repr']
                elif isinstance(emb, tuple):
                    emb = emb[0]
                
                for smi, emb_vec in zip(valid_batch, emb):
                    embeddings_dict[smi] = torch.tensor(emb_vec, dtype=torch.float32)
                
            except Exception as e:
                logger.warning(f"Uni-Mol failed for batch {i}: {e}")
                failed.extend(valid_batch)
            
            if i % 1000 == 0:
                logger.info(f"Embedded {i}/{len(smiles_list)} molecules")
        
        if failed:
            logger.warning(f"Failed to embed {len(failed)} molecules")
        
        return embeddings_dict

def main():
    pert_info_path = "data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz"
    out_cache_path = "unimol_embeddings_l1000.pt"
    
    if not os.path.exists(pert_info_path):
        logger.error(f"{pert_info_path} not found.")
        return
        
    logger.info("Loading perturbagen metadata...")
    pert_info = pd.read_csv(pert_info_path, sep='\t')
    
    unique_smiles = pert_info['canonical_smiles'].dropna().unique().tolist()[:1000]
    logger.info(f"Using {len(unique_smiles)} unique SMILES strings in LINCS L1000 (Subset limit).")
    
    embedder = UniMolEmbedder()
    embeddings_dict = embedder.embed(unique_smiles)
    
    torch.save(embeddings_dict, out_cache_path)
    logger.info(f"Saved {len(embeddings_dict)} embeddings to {out_cache_path}")

if __name__ == "__main__":
    main()
