from pathlib import Path
import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from dataset_l1000_fixed import L1000DataManager, L1000ChunkedDataset, get_murcko_scaffold
from models_12k import NAFS2SingleHeadModel
from batch_correction import load_baselines
from loss import NAFS2Loss
from sampler import CompoundAwareBatchSampler

# Setup logging
log_path = 'train_v6_contrastive.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('NAFS2_V6')

def get_lambda(epoch):
    """Annealing schedule for contrastive loss weight lambda."""
    if epoch < 3:
        # Linear warmup to 0.5
        return 0.5 * (epoch + 1) / 3.0
    elif epoch < 10:
        # Plateau at 0.5
        return 0.5
    elif epoch < 30:
        # Linear decay from 0.5 to 0.1
        return 0.5 - 0.4 * (epoch - 10) / 20.0
    else:
        # Constant 0.1
        return 0.1

def main():
    SUBSET_LIMIT = 20000
    EPOCHS = 50
    BATCH_SIZE = 256
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("NAFS2 Training V6 — InfoNCE & Pathway-Weighted MSE")
    logger.info(f"Device: {device} | Subset Limit: {SUBSET_LIMIT} | Epochs: {EPOCHS}")
    
    # 1. Load Uni-Mol embeddings
    logger.info("Loading Uni-Mol embeddings...")
    unimol_embs = torch.load("unimol_embeddings_l1000_10k.pt", map_location='cpu', weights_only=True)
    
    # 2. Extract control compounds & scaffolds
    meta_df = pd.read_csv("embedded_compounds_metadata_10k.csv")
    control_smiles_list = meta_df[meta_df['is_control'] == True]['smiles'].tolist()
    control_scaffolds = set()
    for s in control_smiles_list:
        scaff = get_murcko_scaffold(s)
        if scaff:
            control_scaffolds.add(scaff)
            
    # 3. Load DataManager & Baselines
    logger.info("Setting up DataManager...")
    dm = L1000DataManager(
        gctx_path="data/l1000/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        sig_info_path="data/l1000/GSE92742_Broad_LINCS_sig_info.txt.gz",
        pert_info_path="data/l1000/GSE92742_Broad_LINCS_pert_info.txt.gz"
    )
    dm.load_metadata()
    cell_types = dm.meta['cell_id'].dropna().unique().tolist()
    cell_type_map = {ct: i for i, ct in enumerate(cell_types)}
    baselines = load_baselines('cell_type_baselines.npz')
    
    # 4. Set up Dataset (strict scaffold-exclusion for OOD controls)
    logger.info("Setting up Dataset with batch correction and Murcko scaffold exclusion...")
    dataset = L1000ChunkedDataset(
        dm, unimol_embs, cell_type_map,
        chunk_size=10000, cache_dir="chunk_cache_12k_v3",
        subset_limit=SUBSET_LIMIT,
        baselines=baselines,
        exclude_smiles=set(control_smiles_list),
        exclude_scaffolds=control_scaffolds
    )
    
    # 5. Split training set by Murcko Scaffolds (OOD validation set)
    logger.info("Splitting dataset by Murcko scaffolds for OOD validation...")
    scaffold_groups = {}
    for idx, item in enumerate(dataset.valid_indices):
        smiles = item[3]
        scaff = get_murcko_scaffold(smiles)
        scaffold_groups.setdefault(scaff, []).append(idx)
        
    unique_scaffolds = list(scaffold_groups.keys())
    np.random.seed(42)
    np.random.shuffle(unique_scaffolds)
    split_idx = int(len(unique_scaffolds) * 0.95)
    
    train_scaffolds = set(unique_scaffolds[:split_idx])
    val_scaffolds = set(unique_scaffolds[split_idx:])
    
    train_indices = []
    val_indices = []
    for scaff, idxs in scaffold_groups.items():
        if scaff in train_scaffolds:
            train_indices.extend(idxs)
        else:
            val_indices.extend(idxs)
            
    logger.info(f"Split: {len(train_indices)} train samples, {len(val_indices)} validation samples.")
    
    # 6. Setup Samplers and Dataloaders
    train_sampler = CompoundAwareBatchSampler(dataset, train_indices, batch_size=BATCH_SIZE, sigs_per_compound=4)
    train_loader = DataLoader(dataset, batch_sampler=train_sampler)
    
    val_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        sampler=torch.utils.data.SubsetRandomSampler(val_indices)
    )
    
    # 7. Setup Model and Loss
    landmark_genes = ['CTSD', 'SQSTM1', 'CTSL', 'CDKN1A', 'CASP3', 'GAA', 'DDIT4', 'ATG5', 'IL1B', 'CDKN2A', 'CASP7', 'BAX', 'PARP1', 'CCL2']
    pathway_genes_all = [
        'LAMP1', 'LAMP2', 'CTSD', 'CTSB', 'CTSL', 'HEXA', 'GAA', 'ATP6V0A1',
        'MAP1LC3B', 'MAP1LC3C', 'ATG5', 'ATG12', 'TFEB', 'TFE3', 'FOXO3',
        'SQSTM1', 'NBR1', 'OPTN', 'CALCOCO2', 'DDIT4', 'TSC1', 'TSC2', 'DEPTOR',
        'CASP3', 'CASP7', 'PARP1', 'BAX', 'BAK1', 'BCL2L1',
        'CDKN2A', 'CDKN1A', 'GLB1', 'IL6', 'IL1A', 'IL1B', 'CCL2', 'CXCL8', 'LMNB1', 'SIRT1', 'SIRT6'
    ]
    inferred_genes = [g for g in pathway_genes_all if g not in landmark_genes]
    
    loss_fn = NAFS2Loss(
        gene_symbols=dm.gene_symbols,
        landmark_genes=landmark_genes,
        inferred_genes=inferred_genes,
        landmark_weight=10.0,
        inferred_weight=2.0,
        tau=0.1
    )
    
    model = NAFS2SingleHeadModel(
        unimol_dim=512,
        n_genes=len(dm.gene_symbols),
        n_cell_types=len(cell_types),
        mechanism_dim=128,
        gene_symbols=dm.gene_symbols
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 8. Training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    logger.info("Beginning Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_contra = 0.0
        
        lambda_val = get_lambda(epoch)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} | Lambda = {lambda_val:.4f}")
        
        for batch_idx, batch in enumerate(train_loader):
            chem_emb = batch['chemical_embedding'].to(device)
            cell_type = batch['cell_type'].to(device)
            target = batch['transcriptome'].to(device)
            smiles_list = batch['smiles']
            
            optimizer.zero_grad()
            
            # Forward pass (with cell-type masking enabled)
            pred, _, _ = model(chem_emb, cell_type, mask_cell_types=True, p_mask=0.5)
            
            loss, mse, contra = loss_fn(pred, target, smiles_list, lambda_weight=lambda_val)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse.item()
            total_contra += contra.item()
            
            if batch_idx % 30 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)} | Total: {loss.item():.4f} | MSE: {mse.item():.4f} | Contra: {contra.item():.4f}")
                
        # Step learning rate scheduler
        scheduler.step()
        
        # Calculate training epoch metrics
        n_batches = len(train_loader)
        avg_train_loss = total_loss / n_batches
        avg_train_mse = total_mse / n_batches
        avg_train_contra = total_contra / n_batches
        
        # 9. Validation Pass
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for batch in val_loader:
                chem_emb = batch['chemical_embedding'].to(device)
                cell_type = batch['cell_type'].to(device)
                target = batch['transcriptome'].to(device)
                
                # Forward pass without masking or dropout for evaluation
                pred, _, _ = model(chem_emb, cell_type, mask_cell_types=False)
                loss = F.mse_loss(pred, target)
                val_mse += loss.item()
                
        avg_val_mse = val_mse / len(val_loader)
        logger.info(f"Epoch {epoch+1} Done | Train Total Loss: {avg_train_loss:.4f} | Train MSE: {avg_train_mse:.4f} | Train Contra: {avg_train_contra:.4f} | Val MSE: {avg_val_mse:.4f}")
        
        # Save checkpoints and monitor early stopping
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), "nafs2_best_v6.pt")
            logger.info(f"  → New best validation model saved (val_loss={avg_val_mse:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                logger.info(f"  → Early stopping triggered! No improvement in validation MSE for 10 epochs.")
                break
                
    # Save final checkpoint
    torch.save(model.state_dict(), "nafs2_v6_final.pt")
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
