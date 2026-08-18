"""
Stage 3: Siamese ResNet-50 Granulation & Healing Tracker.
Quantifies longitudinal wound tissue changes and computes embedding distance between baseline and follow-up images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBackbone(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        feat = self.features(x).squeeze(-1).squeeze(-1)
        emb = F.normalize(self.fc(feat), p=2, dim=1)  # L2 normalize
        return emb


class SiameseWoundTracker(nn.Module):
    """
    Siamese ResNet-style architecture for longitudinal wound comparison.
    Ingests baseline image I_t0 and follow-up image I_t1 -> output similarity distance and healing rate.
    """

    def __init__(self, feature_dim: int = 128):
        super().__init__()

        self.backbone = ConvBackbone(feature_dim=feature_dim)

        self.healing_regressor = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # Predicts area contraction ratio (Area_t1 / Area_t0)
        )

    def forward(self, img_baseline: torch.Tensor, img_followup: torch.Tensor) -> dict:
        """
        Args:
            img_baseline: Baseline wound image tensor (B, 3, H, W)
            img_followup: Follow-up wound image tensor (B, 3, H, W)

        Returns:
            dict containing:
                - 'emb_baseline': L2-normalized embedding (B, feature_dim)
                - 'emb_followup': L2-normalized embedding (B, feature_dim)
                - 'euclidean_distance': Pairwise distance vector (B, 1)
                - 'predicted_area_ratio': Estimated contraction ratio (B, 1)
        """
        emb_0 = self.backbone(img_baseline)
        emb_1 = self.backbone(img_followup)

        # L2 Euclidean distance in embedding space
        dist = torch.norm(emb_0 - emb_1, p=2, dim=1, keepdim=True)

        # Pairwise feature concatenation for healing trajectory regression
        concat_emb = torch.cat([emb_0, emb_1], dim=1)
        area_ratio = torch.sigmoid(self.healing_regressor(concat_emb))

        return {
            'emb_baseline': emb_0,
            'emb_followup': emb_1,
            'euclidean_distance': dist,
            'predicted_area_ratio': area_ratio
        }
