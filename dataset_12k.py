import torch
from torch.utils.data import Dataset, DataLoader
import anndata as ad
import numpy as np

class Unified12kDataset(Dataset):
    """
    Streams the harmonized 12k transcriptomic targets and pairs them with 
    precomputed Uni-Mol embeddings.
    """
    def __init__(self, adata_12k_path, unimol_embeddings_path, cell_type_map=None):
        super().__init__()
        print(f"Loading 12k dataset from {adata_12k_path}...")
        self.adata = ad.read_h5ad(adata_12k_path, backed='r')
        
        print(f"Loading embeddings from {unimol_embeddings_path}...")
        self.unimol = torch.load(unimol_embeddings_path, map_location='cpu')
        
        # Identify columns
        obs_cols = self.adata.obs.columns.str.lower()
        
        smiles_col = None
        for col in ['smiles', 'canonical_smiles', 'compound_smiles']:
            if col in obs_cols:
                smiles_col = self.adata.obs.columns[list(obs_cols).index(col)]
                break
        self.smiles_col = smiles_col
        
        cell_col = None
        for col in ['cell_type', 'cell_line', 'cell_id']:
            if col in obs_cols:
                cell_col = self.adata.obs.columns[list(obs_cols).index(col)]
                break
        self.cell_col = cell_col
        
        # Build cell type map if none provided
        if cell_type_map is None:
            if self.cell_col:
                unique_cells = self.adata.obs[self.cell_col].unique()
                self.cell_map = {c: i for i, c in enumerate(unique_cells)}
            else:
                self.cell_map = {"UNKNOWN": 0}
        else:
            self.cell_map = cell_type_map
            
        print(f"Dataset ready. {self.adata.n_obs} observations.")
        
    def get_num_cell_types(self):
        return len(self.cell_map)
        
    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        obs = self.adata.obs.iloc[idx]
        
        smiles = str(obs[self.smiles_col]) if self.smiles_col else ""
        if smiles in self.unimol:
            unimol_emb = self.unimol[smiles]
        else:
            # Fallback to zero vector if missing
            unimol_emb = torch.zeros(768)
            
        cell_name = str(obs[self.cell_col]) if self.cell_col else "UNKNOWN"
        cell_type_idx = self.cell_map.get(cell_name, 0)
        
        target = torch.tensor(self.adata.X[idx], dtype=torch.float32)
        
        # Load gene confidence from .var if available
        if 'inference_confidence' in self.adata.var:
            confidence = torch.tensor(self.adata.var['inference_confidence'].values, dtype=torch.float32)
        else:
            confidence = torch.ones(target.shape[0], dtype=torch.float32)
            
        return {
            'unimol': unimol_emb,
            'cell_type': torch.tensor(cell_type_idx, dtype=torch.long),
            'target': target,
            'gene_confidence': confidence
        }

def get_dataloader(h5ad_path, emb_path, batch_size=32, shuffle=True, **kwargs):
    dataset = Unified12kDataset(h5ad_path, emb_path, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), dataset.get_num_cell_types()
