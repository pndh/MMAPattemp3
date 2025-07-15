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
    

class UNI(pl.LightningModule):
    def __init__(self, n_genes=1000, learning_rate=1e-4, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_genes = n_genes
        
        self.simCL = SimilarityContrastiveLoss(0.2)
        # login()
        self.enc1 = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.enc2 = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        self.enc0 = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
        
        self.attn_01 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.2)
        # self.attn_02 = CrossAttention(dim=1024, heads=8, dim_head=64, dropout=0.1)
        
        self.apply_lora_to_vit(16, 32)
        
        self.gene_head1 = nn.Linear(1024, n_genes)
        self.gene_head2 = nn.Linear(1024, n_genes)

    def forward(self, x0, x1, x2):
        """
        return out of x2
        """
        feat_0 = self.enc0.forward_features(x0)
        feat_1 = self.enc1.forward_features(x1)
        feat_2 = self.enc2.forward_features(x2)
        
        cls_0, feat_0 = feat_0[:, 0, :], feat_0[:, 1:, :]
        cls_1, feat_1 = feat_1[:, 0, :], feat_1[:, 1:, :]
        cls_2, feat_2 = feat_2[:, 0, :], feat_2[:, 1:, :]
        
        fused_all = torch.cat([cls_2.unsqueeze(1), feat_0, feat_1], dim=1)  # use feat 20x attend to 5x and 10x
        # cont_cell = torch.cat([cls_cont.unsqueeze(1), feat_cell], dim=1)
        
        fused_cls_2 = self.attn_01(fused_all)[:, 0, :]
        
        out1 = self.gene_head1(fused_cls_2)
        out2 = self.gene_head2(cls_2)
        
        out = (out1 + out2) * 0.5
        
        sim_loss = self.simCL(fused_cls_2, cls_2)
        
        # the below return is for the best
        return out, fused_cls_2, sim_loss
    
    def apply_lora_to_vit(self, lora_r, lora_alpha, first_layer_start=15):
        """
        Apply LoRA to all the Linear layers in the Vision Transformer model.
        """
        for enc in [self.enc0, self.enc1, self.enc2]:
            # Step 1: Collect the names of layers to replace
            layers_to_replace = []
            
            for name, module in enc.named_modules():
                if isinstance(module, nn.Linear) :
                    if ('qkv' in name or 'proj' in name) and (int(name.split('.')[1]) >= first_layer_start):
                        # Collect layers for replacement (store name and module)
                        layers_to_replace.append((name, module))
            
            # Step 2: Replace the layers outside of the iteration
            for name, module in layers_to_replace:
                # Create the LoRA-augmented layer
                lora_layer = lora.Linear(module.in_features, module.out_features, r=lora_r, lora_alpha=lora_alpha)
                # Copy weights and bias
                lora_layer.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_layer.bias.data = module.bias.data.clone()

                # Replace the layer in the model
                parent_name, layer_name = name.rsplit('.', 1)
                parent_module = dict(enc.named_modules())[parent_name]
                setattr(parent_module, layer_name, lora_layer)

    # Additional helper to enable LoRA fine-tuning
    def enable_lora_training(self):
        # LoRA for enc 1
        for param in self.enc1.parameters():
            param.requires_grad = False
        for name, param in self.enc1.named_parameters():
            if "lora" in name:
                param.requires_grad = True
           
        # LoRA for enc 2     
        for param in self.enc2.parameters():
            param.requires_grad = False
        for name, param in self.enc2.named_parameters():
            if "lora" in name:
                param.requires_grad = True
         
        # LoRA for enc 0       
        for param in self.enc0.parameters():
            param.requires_grad = False
        for name, param in self.enc0.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Enable gradients for the regression head
        for param in self.gene_head1.parameters():
            param.requires_grad = True
        for param in self.gene_head2.parameters():
            param.requires_grad = True

    def training_step(self, batch, batch_idx):
        patch_0, patch_1, patch_2, center, exp = batch
        pred, cls_smallest, sim_loss = self(patch_0, patch_1, patch_2)
        mse_loss = F.mse_loss(pred, exp)
        if self.current_epoch < 5:
            loss = mse_loss
        else:
            loss = mse_loss + sim_loss * 0.1 
        # loss = mse_loss + sim_loss * 0.1   
        self.log('train_mse_loss', mse_loss)
        self.log('train_sim_loss', sim_loss)
        self.log('train_total_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch_0, patch_1, patch_2, center, exp = batch
        pred, cls_smallest, sim_loss = self(patch_0, patch_1, patch_2)
        mse_loss = F.mse_loss(pred, exp)
        if self.current_epoch < 5:
            loss = mse_loss
        else:
            loss = mse_loss + sim_loss * 0.1 
        # loss = mse_loss + sim_loss * 0.1   
        self.log('val_mse_loss', mse_loss)
        self.log('val_sim_loss', sim_loss)
        self.log('val_total_loss', loss)
        
    def test_step(self, batch, batch_idx):
        patch_0, patch_1, patch_2, center, exp, mask, label = batch
        pred, cls_smallest, sim_loss = self(patch_0, patch_1, patch_2)
        mse_loss = F.mse_loss(pred, exp)
        if self.current_epoch < 5:
            loss = mse_loss
        else:
            loss = mse_loss + sim_loss * 0.1  
        # loss = mse_loss + sim_loss * 0.1   
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
    



