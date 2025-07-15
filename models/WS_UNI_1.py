from models.UNI import UNI
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
import torch.nn.functional as F

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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        # self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x): #change
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        # attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        return x
    
# adaptation of histogene
class WS_UNI(pl.LightningModule):
    def __init__(self, uni_ckpt_path, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, max_epochs=100, dropout=0.1, n_pos=64):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patch_embedding = UNI.load_from_checkpoint(uni_ckpt_path, n_genes=n_genes, learning_rate=learning_rate, max_epochs=max_epochs)
        self.patch_embedding.eval()
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
            
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim, dropout = dropout, emb_dropout = dropout)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )
        
        self.simCL = SimilarityContrastiveLoss(0.2)

    def forward(self, patches_0, patches_1, patches_2, centers):
        B, P, C, H, W = patches_0.shape
        
        # Gộp toàn bộ patches trong batch B với P patches thành (B*P, C, H, W)
        patches0_reshaped = patches_0.view(B * P, C, H, W)
        patches1_reshaped = patches_1.view(B * P, C, H, W)
        patches2_reshaped = patches_2.view(B * P, C, H, W)
        
        with torch.no_grad():
            out1, emb, _ = self.patch_embedding(patches0_reshaped, patches1_reshaped, patches2_reshaped)
            
        # (B*P, D) → (B, P, D)
        out1 = out1.view(B, P, -1)
        patch_embeddings = emb.view(B, P, -1)
        
        # Positional encoding
        centers_x = self.x_embed(centers[:, :, 0])  # (B, P, dim)
        centers_y = self.y_embed(centers[:, :, 1])  # (B, P, dim)
        x = patch_embeddings + centers_x + centers_y
        
        h = self.vit(x)
        out2 = self.gene_head(h)
        
        out = (out1 + out2) * 0.5
        
        sim_loss = self.simCL(patch_embeddings.view(B * P, -1), h.view(B * P, -1))
        
        return out, sim_loss

    def training_step(self, batch, batch_idx):        
        patches_0, patches_1, patches_2, center, exp = batch
        pred, sim = self(patches_0, patches_1, patches_2, center)
        mse = F.mse_loss(pred.view_as(exp), exp) 
        loss = mse + 0.1 * sim
        self.log('test_total_loss', loss)
        self.log('test_mse_loss', mse)
        self.log('train_sim_loss', sim)
        return loss

    def validation_step(self, batch, batch_idx):
        patches_0, patches_1, patches_2, center, exp = batch
        pred, sim = self(patches_0, patches_1, patches_2, center)
        mse = F.mse_loss(pred.view_as(exp), exp) 
        loss = mse + 0.1 * sim
        self.log('test_total_loss', loss)
        self.log('test_mse_loss', mse)
        self.log('train_sim_loss', sim)
        return loss

    def test_step(self, batch, batch_idx):
        patches_0, patches_1, patches_2, center, exp = batch
        pred, sim = self(patches_0, patches_1, patches_2, center)
        mse = F.mse_loss(pred.view_as(exp), exp) 
        loss = mse + 0.1 * sim
        self.log('test_total_loss', loss)
        self.log('test_mse_loss', mse)
        self.log('train_sim_loss', sim)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
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