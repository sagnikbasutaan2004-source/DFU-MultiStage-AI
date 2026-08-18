"""
Unit tests for Stage 2 Clinical Formation & Differential Diagnostic Engine.
Tests Vision, Thermal, Clinical, Gait heads, Cross-Attention Fusion, TSK Fuzzy, and Unified Pipeline.
"""

import torch
import pytest

from models.stage2.segformer_head import TissueSegFormerB4
from models.stage2.thermal_fpn import ThermalFPNResNet34
from models.stage2.tabnet_encoder import TabularClinicalEmbedder
from models.stage2.gait_gru import BiGRUGaitEncoder
from models.stage2.cross_attention_fusion import CrossModalAttentionFusion
from models.stage2.tsk_fuzzy_layer import DifferentiableTSKFuzzyLayer
from models.stage2.stage2_pipeline import Stage2DifferentialDiagnosticEngine


def test_segformer_head_forward():
    x = torch.randn(2, 5, 128, 128)  # Batch 2, 5 channels (RGB + a* + EI), 128x128
    model = TissueSegFormerB4(in_channels=5, num_classes=4, feature_dim=128)
    out = model(x)
    assert out['mask_logits'].shape == (2, 4, 128, 128)
    assert out['visual_embedding'].shape == (2, 128)


def test_thermal_fpn_forward():
    x = torch.randn(2, 1, 128, 128)  # Batch 2, 1-ch LWIR thermal map
    model = ThermalFPNResNet34(in_channels=1, feature_dim=128)
    out = model(x)
    assert out['thermal_embedding'].shape == (2, 128)


def test_tabnet_clinical_embedder_forward():
    x = torch.randn(2, 10)  # Batch 2, 10 lab features
    model = TabularClinicalEmbedder(num_features=10, feature_dim=128)
    out = model(x)
    assert out['clinical_embedding'].shape == (2, 128)
    assert out['sparse_attn_weights'].shape == (2, 10)


def test_bigru_gait_encoder_forward():
    x = torch.randn(2, 16, 50)  # Batch 2, 16 FSR channels, 50 stance steps
    model = BiGRUGaitEncoder(in_channels=16, feature_dim=128)
    out = model(x)
    assert out['gait_embedding'].shape == (2, 128)


def test_cross_attention_fusion():
    h_rgb = torch.randn(2, 128)
    h_thermal = torch.randn(2, 128)
    h_insole = torch.randn(2, 128)
    h_clinical = torch.randn(2, 128)

    fusion = CrossModalAttentionFusion(feature_dim=128, num_heads=4)
    z_fused = fusion(h_rgb, h_thermal, h_insole, h_clinical)
    assert z_fused.shape == (2, 128)


def test_tsk_fuzzy_layer():
    z_fused = torch.randn(2, 128)
    fuzzy = DifferentiableTSKFuzzyLayer(in_features=128, num_rules=24, num_classes=6)
    out = fuzzy(z_fused)

    assert out['diagnosis_logits'].shape == (2, 6)
    assert out['rule_activations'].shape == (2, 24)

    # Rule activations must sum to 1 per sample
    rule_sums = torch.sum(out['rule_activations'], dim=-1)
    torch.testing.assert_close(rule_sums, torch.ones(2), rtol=1e-3, atol=1e-3)


def test_stage2_pipeline_end_to_end():
    rgb = torch.randn(2, 5, 128, 128)
    thermal = torch.randn(2, 1, 128, 128)
    gait = torch.randn(2, 16, 50)
    labs = torch.randn(2, 10)

    engine = Stage2DifferentialDiagnosticEngine(feature_dim=128)
    outputs = engine(rgb, thermal, gait, labs)

    assert outputs['tissue_mask_logits'].shape == (2, 4, 128, 128)
    assert outputs['diagnosis_logits'].shape == (2, 6)
    assert outputs['rule_activations'].shape == (2, 24)
    assert outputs['z_fused'].shape == (2, 128)
