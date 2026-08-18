"""
Unit tests for Phase 1 Foundation & Data Infrastructure.
Validates color utilities, signal processing, synthetic telemetry generator, and datasets.
"""

import os
import numpy as np
import torch
import pytest

from data.utils.lab_color import rgb_to_lab, compute_erythema_index
from data.utils.signal_processing import compute_pressure_time_integral, compute_mfi, compute_sii
from data.insole_dataset import InsoleTelemetryDataset
from data.rgb_dataset import RGBWoundDataset


def test_rgb_to_lab_shape_and_ranges():
    rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    lab = rgb_to_lab(rgb)
    assert lab.shape == (128, 128, 3)
    # L* channel in [0, 100]
    assert lab[:, :, 0].min() >= 0.0
    assert lab[:, :, 0].max() <= 100.0
    # a* channel in [-128, 127]
    assert lab[:, :, 1].min() >= -128.0
    assert lab[:, :, 1].max() <= 127.0


def test_erythema_index_computation():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200  # High Red
    rgb[:, :, 1] = 50   # Low Green

    ei_map = compute_erythema_index(rgb)
    assert ei_map.shape == (64, 64)
    # Erythema index should be positive when Red > Green
    assert np.mean(ei_map) > 0.0


def test_pressure_time_integral():
    fsr = np.ones((100, 16)) * 50.0  # 50 kPa constant
    pti = compute_pressure_time_integral(fsr, sampling_rate=20.0)
    assert pti.shape == (16,)
    # Integral of 50 for 100 samples (99 intervals * 0.05s * 50) = 247.5 kPa*s
    np.testing.assert_allclose(pti[0], 247.5, rtol=1e-3)


def test_insole_dataset_loading():
    dataset = InsoleTelemetryDataset(data_dir="data/raw/insole_telemetry", window_size=100, stride=20)
    assert len(dataset) > 0

    feat, label, mfi, sii = dataset[0]
    assert feat.shape == (18, 100)  # 18 channels, 100 temporal steps
    assert label.dim() == 0          # scalar hazard label
    assert mfi.shape == (16,)        # 16-channel MFI vector
    assert sii.dim() == 0            # scalar SII value


def test_rgb_wound_dataset_loading():
    dataset_path = "data/raw/uwm_wound_segmentation/data/Foot Ulcer Segmentation Challenge/train/images"
    mask_path = "data/raw/uwm_wound_segmentation/data/Foot Ulcer Segmentation Challenge/train/labels"

    if os.path.exists(dataset_path):
        dataset = RGBWoundDataset(image_dir=dataset_path, label_dir=mask_path, target_size=(256, 256), include_lab_ei=True)
        assert len(dataset) > 0

        img, mask = dataset[0]
        assert img.shape == (5, 256, 256)  # 5 channels: RGB + a* + EI
        if mask is not None:
            assert mask.shape == (256, 256)
            assert mask.dtype == torch.long
