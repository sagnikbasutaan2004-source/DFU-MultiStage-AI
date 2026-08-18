"""
Stage 2: Cross-Modal Multi-Head Attention Fusion.
Mathematical formulation:
    Z_fused = Softmax( Q K^T / sqrt(d_k) ) V
    Q = W_q * H_RGB
    K = W_k * [H_Thermal; H_Insole]
    V = W_v * H_Clinical
"""

import math
import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Attention bottleneck layer (d=128) fusing:
    - RGB visual tissue embedding
    - LWIR thermal feature embedding
    - Insole gait dynamics embedding
    - Clinical labs tabular embedding
    """

    def __init__(self, feature_dim: int = 128, num_heads: int = 4):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert self.head_dim * num_heads == feature_dim, "feature_dim must be divisible by num_heads"

        # Linear projections for Query, Key, Value
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_v = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )

    def forward(
        self,
        h_rgb: torch.Tensor,
        h_thermal: torch.Tensor,
        h_insole: torch.Tensor,
        h_clinical: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_rgb: (B, 128) RGB visual tissue embedding.
            h_thermal: (B, 128) Thermal spatial embedding.
            h_insole: (B, 128) Insole gait dynamic embedding.
            h_clinical: (B, 128) Clinical lab tabular embedding.

        Returns:
            torch.Tensor: (B, 128) Fused multi-modal context representation Z_fused.
        """
        batch_size = h_rgb.size(0)

        # 1. Query from RGB visual features: Q = W_q * H_RGB -> (B, 1, d)
        Q = self.W_q(h_rgb).unsqueeze(1)

        # 2. Key from concatenated Thermal & Insole features -> (B, 2, d)
        K_raw = torch.stack([h_thermal, h_insole], dim=1)
        K = self.W_k(K_raw)

        # 3. Value from Clinical lab features -> (B, 1, d)
        V = self.W_v(h_clinical).unsqueeze(1)

        # Reshape for multi-head attention: (B, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)

        # Context output: Attn * V
        # Note: if V has length 1, broadcast/expand V across Key sequence length
        if V.size(2) < K.size(2):
            V = V.repeat(1, 1, K.size(2), 1)

        context = torch.matmul(attn_weights, V)  # (B, num_heads, 1, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, self.feature_dim)

        z_fused = self.out_proj(context)
        return z_fused
