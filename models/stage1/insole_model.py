"""
Stage 1: Pre-Ulcerative Early Warning Model Assembly.
Integrates 1D-CNN + Temporal Transformer to predict pre-ulcer hazard scores
and estimate MFI / SII indices from continuous insole telemetry.
"""

import torch
import torch.nn as nn

from models.stage1.conv1d_encoder import Conv1DTemporalEncoder
from models.stage1.temporal_transformer import TemporalTransformerEncoder


class InsoleEarlyWarningModel(nn.Module):
    """
    End-to-End Stage 1 Model:
    18-Ch Telemetry -> 1D-CNN -> Temporal Transformer -> Hazard Score & Domain Indices.
    """

    def __init__(
        self,
        in_channels: int = 18,
        conv_channels: list = [32, 64, 128],
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.conv_encoder = Conv1DTemporalEncoder(
            in_channels=in_channels,
            channels=conv_channels,
            dropout=dropout
        )

        self.d_model = d_model
        if conv_channels[-1] != d_model:
            self.proj = nn.Conv1d(conv_channels[-1], d_model, kernel_size=1)
        else:
            self.proj = nn.Identity()

        self.transformer = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Global temporal pooling & heads
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # Output Heads
        self.hazard_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.mfi_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 16)  # 16 sensor channels
        )

        self.sii_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)   # Scalar SII
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Telemetry tensor (B, 18, T)

        Returns:
            dict containing:
                - 'hazard_score': (B, 1) tensor in range [0, 1]
                - 'mfi_pred': (B, 16) per-channel fatigue index
                - 'sii_pred': (B, 1) sudomotor impairment score
                - 'latent_embedding': (B, d_model) latent context vector
        """
        # 1. 1D-CNN Feature Extraction: (B, 18, T) -> (B, C_out, T)
        cnn_feat = self.conv_encoder(x)
        cnn_feat = self.proj(cnn_feat)

        # 2. Transpose for Transformer: (B, C_out, T) -> (B, T, d_model)
        seq_feat = cnn_feat.transpose(1, 2)

        # 3. Temporal Transformer Encoding
        trans_out = self.transformer(seq_feat)  # (B, T, d_model)

        # 4. Temporal Pooling -> Latent Embedding: (B, d_model)
        pooled = self.pooling(trans_out.transpose(1, 2)).squeeze(-1)

        # 5. Output Heads
        hazard = self.hazard_head(pooled)
        mfi = self.mfi_head(pooled)
        sii = self.sii_head(pooled)

        return {
            'hazard_score': hazard,
            'mfi_pred': mfi,
            'sii_pred': sii,
            'latent_embedding': pooled
        }
