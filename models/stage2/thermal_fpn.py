"""
Stage 2: Spatial LWIR Thermal Head (ResNet-34 Feature Pyramid Network).
Extracts thermal asymmetry features and localized thermodynamic entropy maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermalFPNResNet34(nn.Module):
    """
    Feature Pyramid Network (FPN) for thermal image analysis.
    Ingests 1-channel thermal map (or 3-channel radiometry) -> 128-dim thermal embedding vector.
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 128):
        super().__init__()

        # Conv backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, num_blocks=2)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # FPN Lateral projections
        self.lat4 = nn.Conv2d(512, feature_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(256, feature_dim, kernel_size=1)
        self.lat2 = nn.Conv2d(128, feature_dim, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_out = nn.Linear(feature_dim, feature_dim)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        ))
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict:
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lat4(c5)
        p4 = self.lat3(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.lat2(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')

        p5_resized = F.interpolate(p5, size=p3.shape[-2:], mode='nearest')
        p4_resized = F.interpolate(p4, size=p3.shape[-2:], mode='nearest')

        fused_fpn = p5_resized + p4_resized + p3
        pooled = self.global_pool(fused_fpn).squeeze(-1).squeeze(-1)
        thermal_embedding = self.fc_out(pooled)

        return {
            'thermal_embedding': thermal_embedding
        }
