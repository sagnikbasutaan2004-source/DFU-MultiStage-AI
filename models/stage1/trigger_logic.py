"""
Stage 1 Trigger Logic Gate.
Evaluates the continuous telemetry trigger condition:
Trigger = (ΔT > 2.2°C) OR (Hazard Score > 0.65)
"""

from typing import Tuple, Dict
import torch


class Stage1TriggerGate:
    """
    Continuous evaluation gate that decides whether to trigger Stage 2 clinical assessment.
    """

    def __init__(self, delta_t_threshold: float = 2.2, hazard_threshold: float = 0.65):
        self.delta_t_threshold = delta_t_threshold
        self.hazard_threshold = hazard_threshold

    def evaluate(self, delta_t_celsius: float, hazard_score: float) -> Dict[str, bool]:
        """
        Args:
            delta_t_celsius: Measured contralateral thermal differential (°C).
            hazard_score: Predicted Stage 1 hazard score from temporal transformer.

        Returns:
            dict containing:
                - 'triggered': bool
                - 'delta_t_triggered': bool
                - 'hazard_triggered': bool
        """
        delta_t_flag = delta_t_celsius > self.delta_t_threshold
        hazard_flag = hazard_score > self.hazard_threshold

        triggered = delta_t_flag or hazard_flag

        return {
            'triggered': triggered,
            'delta_t_triggered': delta_t_flag,
            'hazard_triggered': hazard_flag,
            'delta_t_value': delta_t_celsius,
            'hazard_score_value': hazard_score
        }

    def evaluate_batch(self, delta_t_tensor: torch.Tensor, hazard_tensor: torch.Tensor) -> torch.Tensor:
        """
        Evaluates batch of inputs and returns boolean mask tensor.
        """
        delta_t_flag = delta_t_tensor > self.delta_t_threshold
        hazard_flag = hazard_tensor > self.hazard_threshold
        return delta_t_flag | hazard_flag
