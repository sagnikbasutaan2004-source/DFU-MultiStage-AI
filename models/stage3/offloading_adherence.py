"""
Stage 3: Offloading Adherence Metric (OAM) Computation Module.

Mathematical formulation:
    OAM = 1.0 - [ ∑_{t ∈ Gait} I( P_ulcer_zone(t) > P_ischemia_threshold ) / Total_Ambulatory_Steps ]
"""

import numpy as np


class OffloadingAdherenceCalculator:
    """
    Calculates Offloading Adherence Metric (OAM) from insole pressure telemetry.
    OAM = 1.0 -> 100% adherence to therapeutic offloading footwear.
    OAM < 0.70 -> Warning: Excessive pressure loading on healing wound site.
    """

    def __init__(self, pressure_threshold_kpa: float = 32.0):
        self.p_threshold = pressure_threshold_kpa

    def calculate_oam(
        self,
        ulcer_zone_pressure: np.ndarray,
        total_steps: int
    ) -> float:
        """
        Args:
            ulcer_zone_pressure: 1D array of pressure readings at the specific ulcer site (kPa).
            total_steps: Total number of steps recorded in session.

        Returns:
            float: OAM score in range [0.0, 1.0].
        """
        if total_steps <= 0:
            return 1.0

        # Count steps where pressure exceeded ischemic capillary closure pressure (~32 kPa / 240 mmHg)
        non_compliant_steps = np.sum(ulcer_zone_pressure > self.p_threshold)
        violation_ratio = min(1.0, non_compliant_steps / total_steps)

        oam = 1.0 - violation_ratio
        return float(np.clip(oam, 0.0, 1.0))
