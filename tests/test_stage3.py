"""
Unit tests for Stage 3 Post-Formation Remission & Repair Monitor.
Tests Siamese tracker, Kalman state filter, and Offloading Adherence Metric (OAM).
"""

import numpy as np
import torch
import pytest

from models.stage3.siamese_resnet50 import SiameseWoundTracker
from models.stage3.kalman_filter import WoundKalmanFilter
from models.stage3.offloading_adherence import OffloadingAdherenceCalculator
from models.stage3.stage3_pipeline import Stage3HealingMonitorPipeline


def test_siamese_tracker_forward():
    img0 = torch.randn(2, 3, 128, 128)
    img1 = torch.randn(2, 3, 128, 128)

    tracker = SiameseWoundTracker(feature_dim=128)
    out = tracker(img0, img1)

    assert out['emb_baseline'].shape == (2, 128)
    assert out['emb_followup'].shape == (2, 128)
    assert out['euclidean_distance'].shape == (2, 1)
    assert out['predicted_area_ratio'].shape == (2, 1)


def test_kalman_filter_trajectory():
    # Simulate shrinking wound over 5 days: 100 -> 85 -> 70 -> 55 -> 40 mm²
    measurements = [100.0, 85.0, 70.0, 55.0, 40.0]
    ekf = WoundKalmanFilter(initial_area=100.0, dt_days=1.0)

    states = []
    for m in measurements:
        ekf.predict()
        ekf.update(m)
        states.append(ekf.get_state())

    # Final area should be close to 40 mm²
    assert abs(states[-1]['area_mm2'] - 40.0) < 5.0
    # Healing rate (dArea/dt) should be negative (~ -15 mm²/day)
    assert states[-1]['healing_rate_mm2_per_day'] < 0.0


def test_offloading_adherence_calculator():
    calc = OffloadingAdherenceCalculator(pressure_threshold_kpa=32.0)

    # 100% compliant: pressure always < 32 kPa
    p_compliant = np.ones(100) * 15.0
    oam1 = calc.calculate_oam(p_compliant, total_steps=100)
    assert oam1 == 1.0

    # 50% non-compliant: pressure exceeds 32 kPa half the time
    p_mixed = np.array([40.0] * 50 + [15.0] * 50)
    oam2 = calc.calculate_oam(p_mixed, total_steps=100)
    assert oam2 == 0.5
