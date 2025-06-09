import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, classification_report, precision_score
import pandas as pd
import ast
from sklearn.preprocessing import LabelEncoder
import SimpleITK as sitk
import time as t
import numpy as np
import matplotlib.pyplot as plt
import json, glob, re, os, pydicom
from skimage.draw import polygon
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes,label,binary_dilation,binary_erosion, binary_closing
from ipywidgets import interact
import cv2
import shutil
import tensorflow as tf
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
from einops import repeat
  
# --------------------------- PATCH EMBEDDING
class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings.

    Args:
        dropout (float): Dropout probability.
        patch_size (int): Size of each patch (W=H).
        emb_size (int): Dimension of the embedding.
        in_channels (int): Number of input channels.
    """
    def __init__(self, dropout, patch_size: int, emb_size: int, in_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size,
                      p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class Attention(nn.Module):
    """
    Multi-head self-attention block with LayerNorm and residual connection.

    Args:
        dim (int): Input and output dimension.
        n_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """    
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads

        self.qkv = nn.Linear(dim, dim * 3)  # Combined QKV projection
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim)  # Optional but recommended

    def forward(self, x):
        # x: [batch, seq, dim]
        residual = x  # For the skip connection

        x = x.transpose(0, 1)  # [seq, batch, dim]

        qkv = self.qkv(x)  # [seq, batch, 3*dim]
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # Split into Q, K, V

        attn_output, _ = self.att(q, k, v)
        attn_output = attn_output.transpose(0, 1)  # Back to [batch, seq, dim]

        output = self.proj(attn_output)
        output = self.proj_drop(output)

        output = self.norm(output + residual)  # Add & Norm

        return output



class PreNorm(nn.Module):
    """Applies RMSNorm before the given function."""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.RMSNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (1.0 / (x.shape[-1] ** 0.5))
        return self.scale * x / (norm + self.eps)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ResidualAdd(nn.Module):
    """Applies residual connection to a function output."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) for image classification.

    Args:
        patch_size (int): Size of the image patches.
        emb_dim (int): Embedding dimension.
        heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        ch (int): Number of input channels.
        img_size (int): Size of input image (assumed square).
        n_layers (int): Number of transformer layers.
        out_dim (int): Output dimension (number of classes).
    """
    def __init__(self,
                 patch_size: int,
                 emb_dim: int,
                 heads: int,
                 dropout: float,
                 ch=1,
                 img_size=224,
                 n_layers=4,
                 out_dim=16):
        super(ViT, self).__init__()

        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        self.patch_embedding = PatchEmbedding(dropout=dropout,
                                              in_channels=ch,
                                              patch_size=patch_size,
                                              emb_size=emb_dim)

        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(0.02 * torch.rand(1, 1, emb_dim))

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads=heads, dropout=dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, dropout=dropout)))
            )
            self.layers.append(transformer_block)

        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))

    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        for i in range(self.n_layers):
            x = self.layers[i](x)

        return self.head(x[:, 0, :])

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Args:
        gamma (float): Focusing parameter for modulating factor.
        alpha (Tensor or None): Class weighting factor.
        reduction (str): Specifies reduction to apply to the output.
    """
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none',label_smoothing=0.1)  # shape [B]
        pt = torch.exp(-ce)  # pt = softmax probability of the true class
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)  # shape [B]
            loss = at * (1 - pt) ** self.gamma * ce
        else:
            loss = (1 - pt) ** self.gamma * ce
        return loss.mean() if self.reduction=='mean' else loss.sum()
    