"""
Unit tests for Stage 1 Pre-Ulcerative Early Warning Engine.
Tests Conv1D encoder, Temporal Transformer, InsoleEarlyWarningModel, and Trigger Logic.
"""

import torch
import pytest

from models.stage1.conv1d_encoder import Conv1DTemporalEncoder
from models.stage1.temporal_transformer import TemporalTransformerEncoder
from models.stage1.insole_model import InsoleEarlyWarningModel
from models.stage1.trigger_logic import Stage1TriggerGate


def test_conv1d_encoder_forward():
    x = torch.randn(4, 18, 100)  # Batch 4, 18 channels, 100 time steps
    encoder = Conv1DTemporalEncoder(in_channels=18, channels=[32, 64, 128])
    out = encoder(x)
    assert out.shape == (4, 128, 100)


def test_temporal_transformer_forward():
    x = torch.randn(4, 100, 128)  # Batch 4, 100 time steps, d_model 128
    transformer = TemporalTransformerEncoder(d_model=128, nhead=4, num_layers=2)
    out = transformer(x)
    assert out.shape == (4, 100, 128)


def test_insole_model_end_to_end():
    x = torch.randn(2, 18, 100)  # Batch 2, 18 telemetry channels, 100 samples
    model = InsoleEarlyWarningModel(in_channels=18, d_model=128)
    outputs = model(x)

    assert "hazard_score" in outputs
    assert "mfi_pred" in outputs
    assert "sii_pred" in outputs
    assert "latent_embedding" in outputs

    assert outputs["hazard_score"].shape == (2, 1)
    assert outputs["mfi_pred"].shape == (2, 16)
    assert outputs["sii_pred"].shape == (2, 1)
    assert outputs["latent_embedding"].shape == (2, 128)

    # Hazard score must be bounded in [0, 1] due to sigmoid
    assert (outputs["hazard_score"] >= 0.0).all() and (outputs["hazard_score"] <= 1.0).all()


def test_trigger_logic_gate():
    gate = Stage1TriggerGate(delta_t_threshold=2.2, hazard_threshold=0.65)

    # Normal case: ΔT=1.0, hazard=0.3 -> No trigger
    res1 = gate.evaluate(delta_t_celsius=1.0, hazard_score=0.3)
    assert not res1["triggered"]

    # Thermal trigger: ΔT=2.5, hazard=0.3 -> Triggered
    res2 = gate.evaluate(delta_t_celsius=2.5, hazard_score=0.3)
    assert res2["triggered"]
    assert res2["delta_t_triggered"]

    # Hazard trigger: ΔT=1.0, hazard=0.8 -> Triggered
    res3 = gate.evaluate(delta_t_celsius=1.0, hazard_score=0.8)
    assert res3["triggered"]
    assert res3["hazard_triggered"]
