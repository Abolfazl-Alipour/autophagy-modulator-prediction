import os
import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger('NAFS2')

def get_murcko_scaffold(smiles: str) -> str:
    """Extract the Murcko scaffold of a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold
    except Exception:
        return ""

# =============================================================================
# SECTION 2: GCTX Data Loading with cmapPy (Proper Format)
# =============================================================================

class L1000DataManager:
    """
    Manages LINCS L1000 Level 5 GCTX data.
    Uses cmapPy for correct parsing. Supports chunked loading for memory efficiency.
    Loads all 12,328 genes.
    """
    
    def __init__(self, gctx_path: str, sig_info_path: str, pert_info_path: str):
        self.gctx_path = Path(gctx_path)
        self.sig_info_path = Path(sig_info_path)
        self.pert_info_path = Path(pert_info_path)
        
        # Validate files
        for p in [self.gctx_path, self.sig_info_path, self.pert_info_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")
        
        self.gctoo = None
        self.sig_info = None
        self.pert_info = None
        self.meta = None
        self.gene_symbols = None
        
    def load_metadata(self):
        """Load signature and perturbagen metadata."""
        logger.info("Loading signature metadata...")
        self.sig_info = pd.read_csv(self.sig_info_path, sep='\t', low_memory=False)
        
        logger.info("Loading perturbagen metadata...")
        self.pert_info = pd.read_csv(self.pert_info_path, sep='\t', low_memory=False)
        
        # Merge to get SMILES for each signature
        logger.info("Merging metadata...")
        self.meta = self.sig_info.merge(
            self.pert_info[['pert_id', 'canonical_smiles', 'inchi_key']],
            on='pert_id',
            how='left'
        )
        
        # Set sig_id as index for fast lookup
        if 'sig_id' in self.meta.columns:
            self.meta.set_index('sig_id', inplace=True)
        
        logger.info(f"Total signatures: {len(self.meta)}")
        logger.info(f"With SMILES: {self.meta['canonical_smiles'].notna().sum()}")
        logger.info(f"Unique compounds: {self.meta['pert_iname'].nunique()}")
        logger.info(f"Cell types: {self.meta['cell_id'].nunique()}")
        
        # In LINCS Phase I, 'pert_time' might be named 'pert_time' or 'pert_time_unit'
        if 'pert_time' in self.meta.columns:
            logger.info(f"Time points: {sorted(self.meta['pert_time'].dropna().unique())}")
        
        return self.meta
    
    def load_gctx_chunked(self, chunk_size: int = 10000):
        """
        Load GCTX data in chunks to avoid 23GB RAM load.
        Uses cmapPy.parse with row/col subsetting.
        Loads all 12,328 genes.
        """
        try:
            from cmapPy.pandasGEXpress.parse import parse
        except ImportError:
            logger.error("cmapPy not installed. Run: pip install cmapPy")
            raise
        
        logger.info(f"Loading GCTX in chunks of {chunk_size} signatures...")
        
        # First, parse just metadata to get dimensions and indices
        row_df = parse(str(self.gctx_path), row_meta_only=True)
        col_df = parse(str(self.gctx_path), col_meta_only=True)
        n_genes = row_df.shape[0]
        n_sigs = col_df.shape[0]
        
        logger.info(f"GCTX dimensions: {n_genes} genes x {n_sigs} signatures")
        
        # Load gene symbols from gene_info.txt.gz
        gene_info = pd.read_csv('data/l1000/GSE92742_Broad_LINCS_gene_info.txt.gz', sep='\t')
        gene_map = dict(zip(gene_info['pr_gene_id'].astype(str), gene_info['pr_gene_symbol']))
        self.gene_symbols = [gene_map.get(str(i), str(i)) for i in row_df.index]
            
        logger.info(f"First 10 genes: {self.gene_symbols[:10]}")
        
        # Check presence of critical genes
        critical_genes = ['DDIT4', 'LAMP1', 'LAMP2', 'CTSD', 'TFEB', 'SQSTM1', 'MAP1LC3B', 'CASP3', 'BAX']
        present = [g for g in critical_genes if g in self.gene_symbols]
        missing = [g for g in critical_genes if g not in self.gene_symbols]
        logger.info(f"Critical autophagy genes present: {len(present)}/{len(critical_genes)}")
        if missing:
            logger.warning(f"Missing: {missing}")
        
        # Yield chunks
        for start_idx in range(0, n_sigs, chunk_size):
            end_idx = min(start_idx + chunk_size, n_sigs)
            
            logger.info(f"Loading chunk {start_idx}:{end_idx}...")
            
            # Parse subset of columns using cid
            gctoo_chunk = parse(
                str(self.gctx_path),
                cid=col_df.index[start_idx:end_idx].tolist()
            )
            
            yield gctoo_chunk, start_idx, end_idx
            
            # Force garbage collection
            del gctoo_chunk
            gc.collect()
    
    def get_smiles_for_chunk(self, gctoo_chunk) -> Dict[str, str]:
        """
        Get SMILES strings for signatures in this chunk.
        Returns dict: sig_id -> smiles
        """
        sig_ids = gctoo_chunk.col_metadata_df.index.tolist()
        
        smiles_map = {}
        for sig_id in sig_ids:
            if sig_id in self.meta.index:
                row = self.meta.loc[sig_id]
                # In case of duplicate sig_ids, meta.loc returns DataFrame instead of Series
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                smiles = row.get('canonical_smiles')
                if pd.notna(smiles) and isinstance(smiles, str):
                    smiles_map[sig_id] = smiles
        
        return smiles_map
    
    def get_cell_info_for_chunk(self, gctoo_chunk) -> Dict[str, dict]:
        """Get cell type, time, dose for each signature in chunk."""
        sig_ids = gctoo_chunk.col_metadata_df.index.tolist()
        
        info = {}
        for sig_id in sig_ids:
            if sig_id in self.meta.index:
                row = self.meta.loc[sig_id]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                info[sig_id] = {
                    'cell_type': row.get('cell_id', 'unknown'),
                    'time_h': row.get('pert_time', 6),
                    'dose': row.get('pert_dose', 0),
                    'compound': row.get('pert_iname', 'unknown')
                }
        
        return info


# =============================================================================
# SECTION 3: PyTorch Dataset with Chunked Loading
# =============================================================================

class L1000ChunkedDataset(Dataset):
    """
    PyTorch Dataset that loads L1000 data in chunks from disk.
    For performance, all valid profiles are cached in memory.
    """
    
    def __init__(
        self,
        data_manager: L1000DataManager,
        smiles_embeddings: Dict[str, torch.Tensor],  # smiles -> 512-dim tensor
        cell_type_map: Dict[str, int],
        chunk_size: int = 10000,
        cache_dir: str = "chunk_cache_12k",
        subset_limit: Optional[int] = None,
        baselines: Optional[Dict[str, np.ndarray]] = None,
        exclude_smiles: Optional[set] = None,
        exclude_scaffolds: Optional[set] = None
    ):
        self.dm = data_manager
        self.smiles_emb = smiles_embeddings
        self.cell_type_map = cell_type_map
        self.chunk_size = chunk_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.subset_limit = subset_limit
        self.baselines = baselines  # Per-cell-type DMSO mean profiles for batch correction
        self.exclude_smiles = exclude_smiles
        self.exclude_scaffolds = exclude_scaffolds
        
        # Pre-compute valid signatures (those with SMILES embeddings)
        self.valid_indices = []  # List of (chunk_idx, local_idx, sig_id, smiles)
        self.profiles = []  # List of numpy arrays, to be concatenated
        
        logger.info("Scanning for valid signatures with SMILES and caching in memory...")
        
        chunk_idx = 0
        for gctoo_chunk, start_idx, end_idx in self.dm.load_gctx_chunked(chunk_size):
            smiles_map = self.dm.get_smiles_for_chunk(gctoo_chunk)
            
            chunk_valid_local_indices = []
            chunk_valid_meta = []
            
            for local_idx, sig_id in enumerate(gctoo_chunk.col_metadata_df.index):
                if sig_id in smiles_map:
                    smiles = smiles_map[sig_id]
                    if smiles in self.smiles_emb:
                        if self.exclude_smiles is not None and smiles in self.exclude_smiles:
                            continue
                        if self.exclude_scaffolds is not None:
                            scaffold = get_murcko_scaffold(smiles)
                            if scaffold in self.exclude_scaffolds:
                                continue
                        chunk_valid_local_indices.append(local_idx)
                        chunk_valid_meta.append((chunk_idx, local_idx, sig_id, smiles))
            
            if chunk_valid_local_indices:
                # Extract profiles for these indices: shape [N_valid_in_chunk, n_genes]
                chunk_profiles = gctoo_chunk.data_df.values[:, chunk_valid_local_indices].T
                self.profiles.append(chunk_profiles)
                self.valid_indices.extend(chunk_valid_meta)
            
            chunk_idx += 1
            logger.info(f"Chunk {chunk_idx}: {len(smiles_map)} with SMILES, {len(chunk_valid_local_indices)} valid")
            
            # Save chunk data locally so we have it if needed (optional)
            chunk_path = self.cache_dir / f"chunk_{chunk_idx-1}.npz"
            if not chunk_path.exists():
                np.savez(chunk_path, data=gctoo_chunk.data_df.values, sig_ids=gctoo_chunk.col_metadata_df.index.values)
                
            if self.subset_limit and len(self.valid_indices) >= self.subset_limit:
                excess = len(self.valid_indices) - self.subset_limit
                if excess > 0:
                    self.valid_indices = self.valid_indices[:self.subset_limit]
                    self.profiles[-1] = self.profiles[-1][:-excess]
                logger.info(f"Subset limit reached. Using {self.subset_limit} signatures.")
                break
        
        if self.profiles:
            self.profiles = np.concatenate(self.profiles, axis=0)
        else:
            self.profiles = np.zeros((0, len(self.dm.gene_symbols)))
            
        logger.info(f"Total valid signatures loaded in memory: {len(self.valid_indices)}")
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        chunk_idx, local_idx, sig_id, smiles = self.valid_indices[idx]
        profile = self.profiles[idx].copy()  # [12328]
        
        # Get embedding
        chem_emb = self.smiles_emb[smiles]  # [512]
        
        # Get cell info
        info = self.dm.meta.loc[sig_id] if sig_id in self.dm.meta.index else {}
        if isinstance(info, pd.DataFrame):
            info = info.iloc[0]
            
        cell_type = str(info.get('cell_id', 'unknown'))
        try:
            time_h = float(info.get('pert_time', 6.0))
        except:
            time_h = 6.0
            
        cell_idx = self.cell_type_map.get(cell_type, 0)
        
        # Apply batch correction: subtract cell-type DMSO baseline
        if self.baselines is not None and cell_type in self.baselines:
            profile = profile - self.baselines[cell_type]
        
        return {
            'chemical_embedding': chem_emb,
            'cell_type': torch.tensor(cell_idx, dtype=torch.long),
            'transcriptome': torch.tensor(profile, dtype=torch.float32),
            'time_h': torch.tensor(time_h, dtype=torch.float32),
            'sig_id': sig_id,
            'smiles': smiles
        }
