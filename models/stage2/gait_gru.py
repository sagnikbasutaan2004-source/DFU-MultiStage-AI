"""
Stage 2: Insole Gait Dynamics Bidirectional GRU Encoder.
Encodes temporal stance phase gait pressure sequences into a gait dynamics embedding.
"""

import torch
import torch.nn as nn


class BiGRUGaitEncoder(nn.Module):
    """
    Bidirectional GRU for processing 16-channel insole gait pressure dynamics.
    """

    def __init__(self, in_channels: int = 16, hidden_dim: int = 64, feature_dim: int = 128):
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.proj = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Gait temporal sequence (B, T, 16) or (B, 16, T)

        Returns:
            dict containing:
                - 'gait_embedding': (B, feature_dim) gait feature vector (dim=128)
        """
        if x.size(1) >= 16 and x.size(2) != 16:
            x = x[:, :16, :]  # Extract 16 FSR channels
            x = x.transpose(1, 2)  # (B, 16, T) -> (B, T, 16)

        gru_out, hidden = self.gru(x)

        # Take last forward and backward hidden states
        # hidden shape: (2 * num_layers, B, hidden_dim)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        gait_embedding = self.proj(last_hidden)

        return {
            'gait_embedding': gait_embedding
        }
