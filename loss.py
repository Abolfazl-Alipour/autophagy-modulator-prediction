import torch
import torch.nn as nn
import torch.nn.functional as F

class NAFS2Loss(nn.Module):
    """
    Weighted MSE + InfoNCE Contrastive Loss for NAFS2 V6.
    Applies high weights to direct L1000 landmarks in key pathways
    and medium weights to inferred pathway members.
    InfoNCE contrastive loss pushes predicted profiles of different compounds apart.
    """
    def __init__(self, gene_symbols, landmark_genes, inferred_genes, landmark_weight=10.0, inferred_weight=2.0, tau=0.1):
        super().__init__()
        self.tau = tau
        
        # Initialize gene weights tensor of shape [n_genes]
        self.gene_weights = torch.ones(len(gene_symbols))
        landmark_set = set(landmark_genes)
        inferred_set = set(inferred_genes)
        
        for idx, symbol in enumerate(gene_symbols):
            if symbol in landmark_set:
                self.gene_weights[idx] = landmark_weight
            elif symbol in inferred_set:
                self.gene_weights[idx] = inferred_weight
                
    def forward(self, pred, target, smiles_list, lambda_weight=1.0):
        # 1. Weighted MSE Loss
        weights = self.gene_weights.to(pred.device)
        mse_loss = torch.mean(weights * (pred - target) ** 2)
        
        # 2. Batch-wise InfoNCE Contrastive Loss
        # Normalize profiles to unit sphere
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)
        
        # Cosine similarity matrix scaled by temperature: shape [B, B]
        sim_matrix = torch.matmul(pred_norm, target_norm.t()) / self.tau
        
        # InfoNCE Target Labels: [0, 1, 2, ..., B-1] (diagonal index)
        batch_size = pred.size(0)
        labels = torch.arange(batch_size, device=pred.device)
        
        # Construct same-compound mask: 0.0 where smiles are identical but indices are different
        mask = torch.ones_like(sim_matrix)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j and smiles_list[i] == smiles_list[j]:
                    mask[i, j] = 0.0
                    
        # Apply mask by filling with large negative value (equivalent to softmax probability = 0)
        sim_matrix = sim_matrix.masked_fill(mask == 0.0, -1e9)
        
        # InfoNCE loss
        contrastive_loss = F.cross_entropy(sim_matrix, labels)
        
        # Combined objective
        total_loss = mse_loss + lambda_weight * contrastive_loss
        return total_loss, mse_loss, contrastive_loss
