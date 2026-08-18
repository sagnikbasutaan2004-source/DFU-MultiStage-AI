"""
Stage 3: Post-Formation Remission & Repair Monitor Pipeline.
Combines Siamese Granulation Tracker, Extended Kalman Filter, and Offloading Adherence Metric.
"""

import numpy as np
import torch
import torch.nn as nn

from models.stage3.siamese_resnet50 import SiameseWoundTracker
from models.stage3.kalman_filter import WoundKalmanFilter
from models.stage3.offloading_adherence import OffloadingAdherenceCalculator


class Stage3HealingMonitorPipeline(nn.Module):
    """
    Unified Stage 3 Monitor Pipeline for tracking longitudinal wound remission and offloading compliance.
    """

    def __init__(self, feature_dim: int = 128, pressure_threshold_kpa: float = 32.0):
        super().__init__()

        self.siamese_tracker = SiameseWoundTracker(feature_dim=feature_dim)
        self.oam_calculator = OffloadingAdherenceCalculator(pressure_threshold_kpa=pressure_threshold_kpa)

    def forward(
        self,
        img_baseline: torch.Tensor,
        img_followup: torch.Tensor
    ) -> dict:
        return self.siamese_tracker(img_baseline, img_followup)

    def track_trajectory(
        self,
        area_measurements: list,
        dt_days: float = 1.0
    ) -> list:
        """
        Runs Extended Kalman Filter over a sequence of longitudinal wound area measurements (mm²).
        """
        if not area_measurements:
            return []

        ekf = WoundKalmanFilter(initial_area=area_measurements[0], dt_days=dt_days)
        trajectory = []

        for area in area_measurements:
            ekf.predict()
            ekf.update(area)
            trajectory.append(ekf.get_state())

        return trajectory
