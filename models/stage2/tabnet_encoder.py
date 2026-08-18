"""
Stage 2: Clinical Labs Tabular Embedder (TabNet Sparse Attention).
Encodes non-linear glycemic, renal, and inflammatory markers (HbA1c, eGFR, Creatinine, WBC, CRP).
"""

import torch
import torch.nn as nn


class TabularClinicalEmbedder(nn.Module):
    """
    TabNet-style sparse attention tabular embedder for clinical lab values.
    Transforms raw tabular features -> 128-dim clinical embedding vector.
    """

    def __init__(self, num_features: int = 10, feature_dim: int = 128):
        super().__init__()

        self.num_features = num_features

        # Batch normalization for tabular features
        self.input_bn = nn.BatchNorm1d(num_features)

        # Sparse attention mask generator
        self.attn_mask_gen = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.Softmax(dim=-1)
        )

        # Tabular representation layers
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tabular tensor (B, num_features)
               Features: [HbA1c, eGFR, Creatinine, Fasting_Glucose, WBC, CRP, ESR, Age, BMI, Duration_Diabetes]

        Returns:
            dict containing:
                - 'clinical_embedding': (B, feature_dim) clinical feature vector (dim=128)
                - 'sparse_attn_weights': (B, num_features) feature importance weights
        """
        norm_x = self.input_bn(x)
        sparse_attn_weights = self.attn_mask_gen(norm_x)

        # Apply sparse feature selection
        masked_x = norm_x * sparse_attn_weights
        clinical_embedding = self.encoder(masked_x)

        return {
            'clinical_embedding': clinical_embedding,
            'sparse_attn_weights': sparse_attn_weights
        }
