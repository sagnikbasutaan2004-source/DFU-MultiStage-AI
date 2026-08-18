"""
Stage 2: End-to-End Clinical Formation & Differential Diagnostic Pipeline.
Assembles SegFormer Vision Head, Thermal FPN, TabNet Clinical Embedder, BiGRU Gait Encoder,
Cross-Modal Attention Fusion, and TSK Neuro-Fuzzy Layer into a unified model.
"""

import torch
import torch.nn as nn

from models.stage2.segformer_head import TissueSegFormerB4
from models.stage2.thermal_fpn import ThermalFPNResNet34
from models.stage2.tabnet_encoder import TabularClinicalEmbedder
from models.stage2.gait_gru import BiGRUGaitEncoder
from models.stage2.cross_attention_fusion import CrossModalAttentionFusion
from models.stage2.tsk_fuzzy_layer import DifferentiableTSKFuzzyLayer


class Stage2DifferentialDiagnosticEngine(nn.Module):
    """
    Unified Stage 2 Pipeline:
    Ingests Multi-Modal Inputs -> Extracts Modality Embeddings -> Cross-Attention Fusion -> TSK Neuro-Fuzzy Reasoning.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_classes_seg: int = 4,
        num_classes_diag: int = 6,
        num_rules: int = 24
    ):
        super().__init__()

        # Modality Heads
        self.vision_head = TissueSegFormerB4(in_channels=5, num_classes=num_classes_seg, feature_dim=feature_dim)
        self.thermal_head = ThermalFPNResNet34(in_channels=1, feature_dim=feature_dim)
        self.clinical_head = TabularClinicalEmbedder(num_features=10, feature_dim=feature_dim)
        self.gait_head = BiGRUGaitEncoder(in_channels=16, feature_dim=feature_dim)

        # Cross-Modal Multi-Head Attention Fusion
        self.fusion = CrossModalAttentionFusion(feature_dim=feature_dim, num_heads=4)

        # Neuro-Fuzzy Reasoning Layer
        self.fuzzy_reasoner = DifferentiableTSKFuzzyLayer(in_features=feature_dim, num_rules=num_rules, num_classes=num_classes_diag)

    def forward(
        self,
        rgb_img: torch.Tensor,
        thermal_map: torch.Tensor,
        insole_gait: torch.Tensor,
        clinical_labs: torch.Tensor
    ) -> dict:
        """
        Args:
            rgb_img: (B, 5, H, W) RGB + a* + EI image.
            thermal_map: (B, 1, H, W) LWIR thermal map.
            insole_gait: (B, 16, T) or (B, T, 16) gait dynamics.
            clinical_labs: (B, 10) metabolic & inflammatory labs.

        Returns:
            dict containing:
                - 'tissue_mask_logits': (B, 4, H, W) segmentation logits
                - 'diagnosis_logits': (B, 6) differential diagnosis probabilities
                - 'rule_activations': (B, 24) normalized TSK rule activations
                - 'z_fused': (B, 128) multi-modal context vector
        """
        # 1. Modality-specific feature extraction
        vis_out = self.vision_head(rgb_img)
        therm_out = self.thermal_head(thermal_map)
        clin_out = self.clinical_head(clinical_labs)
        gait_out = self.gait_head(insole_gait)

        # 2. Cross-Modal Attention Bottleneck Fusion
        z_fused = self.fusion(
            h_rgb=vis_out['visual_embedding'],
            h_thermal=therm_out['thermal_embedding'],
            h_insole=gait_out['gait_embedding'],
            h_clinical=clin_out['clinical_embedding']
        )

        # 3. Differentiable TSK Fuzzy Reasoning
        fuzzy_out = self.fuzzy_reasoner(z_fused)

        return {
            'tissue_mask_logits': vis_out['mask_logits'],
            'diagnosis_logits': fuzzy_out['diagnosis_logits'],
            'rule_activations': fuzzy_out['rule_activations'],
            'sparse_lab_weights': clin_out['sparse_attn_weights'],
            'z_fused': z_fused
        }
