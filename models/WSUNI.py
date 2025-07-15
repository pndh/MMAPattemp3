import os
from argparse import ArgumentParser
import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from einops import rearrange
import timm
import loralib as lora


class SimilarityContrastiveLoss(nn.Module):
    """
    Similarity Contrastive Loss:
    - Encourages positive pairs (same index in batch) to have high similarity.
    - Encourages negative pairs (different indices) to have low similarity.
    """

    def __init__(self, margin=0.5):
        """
        :param margin: Margin for negative pairs (default: 0.5)
        """
        super(SimilarityContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2):
        """
        Compute Similarity Contrastive Loss.

        :param x1: Tensor of shape (B, D) - First batch of vectors
        :param x2: Tensor of shape (B, D) - Second batch of vectors
        :return: Similarity contrastive loss
        """
        B, D = x1.shape

        # Compute cosine similarity matrix between all pairs
        cos_sim_matrix = F.cosine_similarity(x1.unsqueeze(1), x2.unsqueeze(0), dim=-1)  # (B, B)

        # Create labels: Positive pairs are diagonal (same index), others are negatives
        labels = torch.eye(B, device=x1.device)  # Identity matrix: 1 on diagonal, 0 elsewhere

        # Loss for positive pairs (diagonal) -> encourage similarity close to 1
        positive_loss = (1 - cos_sim_matrix) * labels  # Only for positive pairs

        # Loss for negative pairs (off-diagonal) -> encourage similarity < margin
        negative_loss = torch.clamp(cos_sim_matrix - self.margin, min=0) * (1 - labels)

        # Compute total loss (average over all pairs)
        loss = positive_loss.sum() + negative_loss.sum()
        loss /= B  # Normalize by batch size

        return loss
    
class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        
        self.pre_norm = nn.LayerNorm(dim)
        
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        x_qkv = self.pre_norm(x_qkv)
        
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out
    

class WSUNI(pl.LightningModule):
    def __init__(self, n_genes=1000, learning_rate=1e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_genes = n_genes
        
        self.simCL = SimilarityContrastiveLoss(0.2)
        
        self.attn_01 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.2)
        # self.attn_02 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.1)
        
        self.gene_head = nn.Linear(1024, n_genes)

    def forward(self, out_uni, cls_uni, center, cls_neighbors, cls_neighbors_loc):
        fused_all = torch.cat([cls_uni.unsqueeze(1), cls_neighbors], dim=1)  # use feat 20x attend to 5x and 10x
        # cont_cell = torch.cat([cls_cont.unsqueeze(1), feat_cell], dim=1)
        
        fused_cls = self.attn_01(fused_all)[:, 0, :]
        # fused_cls = fused_all.mean(dim=1)
        
        out = self.gene_head(fused_cls)
        
        out = (out_uni + out) * 0.5
        
        sim_loss = self.simCL(fused_cls, cls_uni)
        
        # the below return is for the best
        return out, fused_cls, sim_loss

# class WSUNI(pl.LightningModule):
#     def __init__(self, n_genes=1000, learning_rate=1e-4, max_epochs=100):
#         super().__init__()
#         self.save_hyperparameters()

#         self.learning_rate = learning_rate
#         self.max_epochs = max_epochs
#         self.n_genes = n_genes

#         self.simCL = SimilarityContrastiveLoss(0.2)

#         self.attn_01 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.2)

#         self.pos_mlp = nn.Sequential(  # learnable embedding from relative distance
#             nn.Linear(2, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1024)  # same dim as token embedding
#         )

#         self.gene_head = nn.Linear(1024, n_genes)

#     def forward(self, out_uni, cls_uni, center, cls_neighbors, cls_neighbors_loc):
#         """
#         center: [B, 2]         — tọa độ patch hiện tại
#         cls_neighbors: [B, L, 1024]
#         cls_neighbors_loc: [B, L, 2] — tọa độ các neighbor
#         """
#         # 1. Tính relative position
#         delta = cls_neighbors_loc - center.unsqueeze(1)  # [B, L, 2]

#         # 2. Học position embedding từ relative position
#         pos_emb = self.pos_mlp(delta)  # [B, L, 1024]

#         # 3. Cộng position embedding vào neighbor token
#         cls_neighbors = cls_neighbors + pos_emb  # [B, L, 1024]

#         # 4. Tạo fused_all = center token + neighbors
#         fused_all = torch.cat([cls_uni.unsqueeze(1), cls_neighbors], dim=1)  # [B, L+1, 1024]

#         # 5. Attention: lấy output của token đầu (center)
#         fused_cls = self.attn_01(fused_all)[:, 0, :]  # [B, 1024]

#         # 6. Predict gene expression
#         out = self.gene_head(fused_cls)
#         out = (out_uni + out) * 0.5

#         # 7. Similarity loss
#         sim_loss = self.simCL(fused_cls, cls_uni)

#         return out, fused_cls, sim_loss

    def training_step(self, batch, batch_idx):
        out_uni, cls_uni, cls_neighbors, cls_neighbors_loc, center, exp = batch
        pred, fused_cls, sim_loss = self(out_uni, cls_uni, center, cls_neighbors, cls_neighbors_loc)
        mse_loss = F.mse_loss(pred, exp)
        # if self.current_epoch < 5:
        #     loss = mse_loss
        # else:
        #     loss = mse_loss + sim_loss * 0.1 
        loss = mse_loss + sim_loss * 0.1   
        self.log('train_mse_loss', mse_loss)
        self.log('train_sim_loss', sim_loss)
        self.log('train_total_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out_uni, cls_uni, cls_neighbors, cls_neighbors_loc, center, exp = batch
        pred, fused_cls, sim_loss = self(out_uni, cls_uni, cls_neighbors, cls_neighbors_loc)
        mse_loss = F.mse_loss(pred, exp)
        # if self.current_epoch < 5:
        #     loss = mse_loss
        # else:
        #     loss = mse_loss + sim_loss * 0.1 
        loss = mse_loss + sim_loss * 0.1   
        self.log('val_mse_loss', mse_loss)
        self.log('val_sim_loss', sim_loss)
        self.log('val_total_loss', loss)
        
    def test_step(self, batch, batch_idx):
        out_uni, cls_uni, cls_neighbors, cls_neighbors_loc, center, exp = batch
        pred, fused_cls, sim_loss = self(out_uni, cls_uni, cls_neighbors, cls_neighbors_loc)
        mse_loss = F.mse_loss(pred, exp)
        # if self.current_epoch < 5:
        #     loss = mse_loss
        # else:
        #     loss = mse_loss + sim_loss * 0.1  
        loss = mse_loss + sim_loss * 0.1   
        self.log('test_mse_loss', mse_loss)
        self.log('test_sim_loss', sim_loss)
        self.log('test_total_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,  
            eta_min=1e-6 
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--max_epochs', type=int, default=100) 
        return parser
    



