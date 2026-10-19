import numpy as np
import torch
from torch.utils.data import Sampler

class CompoundAwareBatchSampler(Sampler):
    """
    Custom BatchSampler that groups multiple signatures of the same compound
    together in each batch, while mixing in negative compounds.
    Accepts a specific list of indices to support scaffold splits.
    """
    def __init__(self, dataset, indices, batch_size=256, sigs_per_compound=4):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.sigs_per_compound = sigs_per_compound
        
        # Group indices by compound (smiles)
        self.compound_groups = {}
        for idx in indices:
            # item format in valid_indices: (chunk_idx, local_idx, sig_id, smiles)
            item = dataset.valid_indices[idx]
            smiles = item[3]
            self.compound_groups.setdefault(smiles, []).append(idx)
            
        self.n_compounds_per_batch = batch_size // sigs_per_compound
        
    def __iter__(self):
        compounds = list(self.compound_groups.keys())
        np.random.shuffle(compounds)
        
        current_batch = []
        for smiles in compounds:
            indices = self.compound_groups[smiles]
            # Randomly select sigs_per_compound samples for this compound
            if len(indices) >= 2:
                sampled = np.random.choice(indices, min(self.sigs_per_compound, len(indices)), replace=False).tolist()
            else:
                sampled = indices
            current_batch.extend(sampled)
            
            # When we have enough samples for a batch, yield it
            if len(current_batch) >= self.batch_size:
                # Shuffle the batch indices to prevent identical compounds from always being adjacent
                np.random.shuffle(current_batch)
                yield current_batch[:self.batch_size]
                current_batch = current_batch[self.batch_size:]
                
        # Yield remainder if it's large enough (at least half batch size) to prevent small trailing batches
        if len(current_batch) >= self.batch_size // 2:
            yield current_batch
            
    def __len__(self):
        return len(self.indices) // self.batch_size
