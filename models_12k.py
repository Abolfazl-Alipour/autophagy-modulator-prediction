import torch
import torch.nn as nn
import torch.nn.functional as F

class NAFS2Model(nn.Module):
    def __init__(self, unimol_dim=512, n_genes=12328, n_cell_types=76, mechanism_dim=256, gene_symbols=None):
        super().__init__()
        
        # Chemical encoder: Uni-Mol → mechanism latent
        self.chem_encoder = nn.Sequential(
            nn.Linear(unimol_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, mechanism_dim)
        )
        
        # Cell type embedding
        self.cell_embedding = nn.Embedding(n_cell_types, 64)
        
        # Shared Latent mechanism space
        self.shared_latent = nn.Sequential(
            nn.Linear(mechanism_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Pathway-specific target genes
        self.autophagy_genes = ['LAMP1', 'LAMP2', 'CTSD', 'CTSB', 'CTSL', 'HEXA', 'GAA', 'ATP6V0A1',
                                'MAP1LC3B', 'MAP1LC3C', 'ATG5', 'ATG12', 'TFEB', 'TFE3', 'FOXO3',
                                'SQSTM1', 'NBR1', 'OPTN', 'CALCOCO2', 'DDIT4', 'TSC1', 'TSC2', 'DEPTOR']
        self.apoptosis_genes = ['CASP3', 'CASP7', 'PARP1', 'BAX', 'BAK1', 'BCL2L1']
        self.senescence_genes = ['CDKN2A', 'CDKN1A', 'GLB1', 'IL6', 'IL1A', 'IL1B', 'CCL2', 'CXCL8', 'LMNB1', 'SIRT1', 'SIRT6']
        
        # Prevent overlapping indices (assign FOXO3 solely to autophagy)
        self.apoptosis_genes = [g for g in self.apoptosis_genes if g not in self.autophagy_genes]
        self.senescence_genes = [g for g in self.senescence_genes if g not in self.autophagy_genes and g not in self.apoptosis_genes]
        
        self.n_genes = n_genes
        self.gene_symbols = gene_symbols
        
        # Map target genes to indices in the output space
        if gene_symbols is not None:
            self.gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
            self.autophagy_idx = [self.gene_to_idx[g] for g in self.autophagy_genes if g in self.gene_to_idx]
            self.apoptosis_idx = [self.gene_to_idx[g] for g in self.apoptosis_genes if g in self.gene_to_idx]
            self.senescence_idx = [self.gene_to_idx[g] for g in self.senescence_genes if g in self.gene_to_idx]
        else:
            self.autophagy_idx = []
            self.apoptosis_idx = []
            self.senescence_idx = []
            
        # 1. Autophagy Head
        n_auto = max(1, len(self.autophagy_idx))
        self.autophagy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_auto)
        )
        
        # 2. Apoptosis Head
        n_apop = max(1, len(self.apoptosis_idx))
        self.apoptosis_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_apop)
        )
        
        # 3. Senescence Head
        n_sen = max(1, len(self.senescence_idx))
        self.senescence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_sen)
        )
        
        # 4. Global Baseline Reconstruction Head
        self.recon_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_genes)
        )
        
        # Gene-wise confidence (learned)
        self.gene_confidence = nn.Parameter(torch.ones(n_genes) * 0.5)
        
    def forward(self, unimol_emb, cell_type_idx):
        z_chem = self.chem_encoder(unimol_emb)
        z_cell = self.cell_embedding(cell_type_idx)
        z_merged = torch.cat([z_chem, z_cell], dim=-1)
        
        latent = self.shared_latent(z_merged)
        
        # Baseline predicted logFC for all genes
        out = self.recon_head(latent)
        
        # Overwrite indices with outputs from pathway specialized heads
        # PyTorch clone avoids in-place operation gradient tracking issues
        if len(self.autophagy_idx) > 0:
            pred_auto = self.autophagy_head(latent)
            out = out.clone()
            out[:, self.autophagy_idx] = pred_auto
            
        if len(self.apoptosis_idx) > 0:
            pred_apop = self.apoptosis_head(latent)
            out = out.clone()
            out[:, self.apoptosis_idx] = pred_apop
            
        if len(self.senescence_idx) > 0:
            pred_sen = self.senescence_head(latent)
            out = out.clone()
            out[:, self.senescence_idx] = pred_sen
            
        return out, self.gene_confidence, z_chem
    
    def predict_autophagy(self, unimol_emb, cell_type_idx, scorer):
        predicted_logfc, _, _ = self.forward(unimol_emb, cell_type_idx)
        coherence, _ = scorer.score(predicted_logfc)
        return coherence


class UnifiedAutophagyScorer:
    """
    Scores autophagy induction from 12k-dim predicted transcriptome.
    Uses all target genes.
    """
    
    GENE_SETS = {
        'lysosomal_biogenesis': ['LAMP1', 'LAMP2', 'CTSD', 'CTSB', 'CTSL', 'HEXA', 'GAA', 'ATP6V0A1'],
        'autophagy_machinery': ['MAP1LC3B', 'MAP1LC3A', 'MAP1LC3C', 'GABARAP', 'GABARAPL1', 'GABARAPL2', 'ATG5', 'ATG12'],
        'master_regulators': ['TFEB', 'TFE3', 'FOXO3'],
        'cargo_recognition': ['SQSTM1', 'NBR1', 'OPTN', 'CALCOCO2'],
        'negative_apoptosis': ['CASP3', 'CASP7', 'PARP1', 'BAX', 'BAK1', 'BCL2L1'],
        'mTORC1_suppression': ['DDIT4', 'TSC1', 'TSC2', 'DEPTOR'],
    }
    
    def __init__(self, gene_names):
        self.gene_index = {name: i for i, name in enumerate(gene_names)}
        
    def score(self, predicted_logfc):
        """
        predicted_logfc: [batch, 12328]
        Returns: coherence [batch], diagnostics
        """
        # Extract gene set scores
        set_scores = {}
        for set_name, genes in self.GENE_SETS.items():
            indices = [self.gene_index[g] for g in genes if g in self.gene_index]
            if len(indices) == 0:
                set_scores[set_name] = torch.zeros(predicted_logfc.shape[0], device=predicted_logfc.device)
            else:
                set_scores[set_name] = predicted_logfc[:, indices].mean(dim=1)
        
        # Core autophagy score
        lysosomal = F.relu(set_scores.get('lysosomal_biogenesis', torch.zeros(predicted_logfc.shape[0], device=predicted_logfc.device)))
        machinery = F.relu(set_scores.get('autophagy_machinery', torch.zeros(predicted_logfc.shape[0], device=predicted_logfc.device)))
        regulators = F.relu(set_scores.get('master_regulators', torch.zeros(predicted_logfc.shape[0], device=predicted_logfc.device)))
        cargo = set_scores.get('cargo_recognition', torch.zeros(predicted_logfc.shape[0], device=predicted_logfc.device))  # p62 should be NEGATIVE (degraded)
        
        # Apoptosis safety
        apoptosis = set_scores.get('negative_apoptosis', torch.zeros(predicted_logfc.shape[0], device=predicted_logfc.device))
        apoptosis_penalty = F.relu(apoptosis - 0.3)  # Penalize if > 0.3
        
        # mTORC1 suppression (positive = good)
        mtor_supp = set_scores.get('mTORC1_suppression', torch.zeros(predicted_logfc.shape[0], device=predicted_logfc.device))
        
        # Combined coherence
        coherence = (
            0.35 * lysosomal +
            0.25 * machinery +
            0.20 * regulators +
            0.10 * F.relu(-cargo) +  # SQSTM1 degradation (decrease)
            0.10 * F.relu(mtor_supp)
        )
        
        # Subtract apoptosis penalty for post-eval/coherence scoring
        coherence = coherence - 0.5 * apoptosis_penalty
        coherence = torch.clamp(coherence, 0, 1)
        
        diagnostics = {
            'lysosomal': lysosomal,
            'machinery': machinery,
            'regulators': regulators,
            'cargo_p62': cargo,
            'apoptosis': apoptosis,
            'mtor_supp': mtor_supp
        }
        
        return coherence, diagnostics


class NAFS2SingleHeadModel(nn.Module):
    """
    Simplified single-head NAFS2 model with bottleneck layers.
    Maps Uni-Mol embeddings directly to the full 12,328-gene transcriptome profiles.
    """
    def __init__(self, unimol_dim=512, n_genes=12328, n_cell_types=76, mechanism_dim=128, gene_symbols=None):
        super().__init__()
        
        # Chemical encoder: Uni-Mol (512) -> 256 -> mechanism latent (128-dim)
        self.chem_encoder = nn.Sequential(
            nn.Linear(unimol_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, mechanism_dim)
        )
        
        # Cell type embedding (reduced to 16-dim)
        self.cell_embedding = nn.Embedding(n_cell_types, 16)
        
        # Shared Latent mechanism space (reduced to 128-dim)
        self.shared_latent = nn.Sequential(
            nn.Linear(mechanism_dim + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output head (reconstruction) mapping 128 -> 12328 directly
        self.recon_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, n_genes)
        )
        
        # Gene-wise confidence (learned)
        self.gene_confidence = nn.Parameter(torch.ones(n_genes) * 0.5)
        self.n_genes = n_genes
        self.gene_symbols = gene_symbols
        
    def forward(self, unimol_emb, cell_type_idx, mask_cell_types=False, p_mask=0.5):
        z_chem = self.chem_encoder(unimol_emb)
        z_cell = self.cell_embedding(cell_type_idx)
        
        if mask_cell_types and self.training:
            # Create a boolean mask of shape [batch_size, 1] with probability p_mask
            mask = (torch.rand(z_cell.size(0), 1, device=z_cell.device) < p_mask)
            z_cell = z_cell.masked_fill(mask, 0.0)
            
        z_merged = torch.cat([z_chem, z_cell], dim=-1)
        latent = self.shared_latent(z_merged)
        out = self.recon_head(latent)
        
        return out, self.gene_confidence, z_chem
