"""
Stage 2: SegFormer-B4 Tissue Segmentation & RGB Vision Head.
Multi-class tissue composition mask (Granulation, Slough, Necrotic, Background)
fused with CIE L*a*b* Erythema Index (EI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(x)


class TissueSegFormerB4(nn.Module):
    """
    SegFormer-B4 Architecture for multi-class tissue composition segmentation.
    Ingests 5-channel input (RGB + a* redness + Erythema Index map).
    """

    def __init__(self, in_channels: int = 5, num_classes: int = 4, feature_dim: int = 128):
        super().__init__()

        # Encoder stages
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(256, 512)

        # Decoder stages
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Output Segmentation Mask Head
        self.mask_head = nn.Conv2d(64, num_classes, kernel_size=1)

        # Feature Bottleneck projection for Multi-Modal Fusion (dim=128)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(512, feature_dim)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input tensor (B, 5, H, W) where channels are [R, G, B, a*, EI]

        Returns:
            dict containing:
                - 'mask_logits': (B, num_classes, H, W) raw segmentation logits
                - 'visual_embedding': (B, feature_dim) bottleneck feature vector (dim=128)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck visual embedding for multi-modal cross-attention fusion
        bottleneck_vec = self.global_pool(e4).squeeze(-1).squeeze(-1)
        visual_embedding = self.feature_proj(bottleneck_vec)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        mask_logits = self.mask_head(d1)

        return {
            'mask_logits': mask_logits,
            'visual_embedding': visual_embedding
        }
