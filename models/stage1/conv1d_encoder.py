"""
Stage 1: 1D-CNN Temporal Feature Extractor.
Ingests 18-channel insole telemetry streams (16 FSR, 1 RH, 1 ΔT)
and extracts multi-scale temporal representation vectors.
"""

import torch
import torch.nn as nn


class Conv1DTemporalEncoder(nn.Module):
    """
    1D-CNN Encoder for multi-channel temporal sensor streams.
    """

    def __init__(
        self,
        in_channels: int = 18,
        channels: list = [32, 64, 128],
        kernel_size: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        curr_ch = in_channels
        for out_ch in channels:
            layers.extend([
                nn.Conv1d(curr_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            curr_ch = out_ch

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C_in, T) e.g., (B, 18, 100)

        Returns:
            torch.Tensor: Feature map of shape (B, C_out, T) e.g., (B, 128, 100)
        """
        return self.encoder(x)
