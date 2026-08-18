"""
Agentic Optimization Layer: 2-Player Game-Theoretic Nash Arbiter.
Computes optimal mixed-strategy Nash equilibrium probability p* to balance sensitivity against alert fatigue.
"""

from typing import Dict, Tuple
import numpy as np

from models.agentic.payoff_matrix import ClinicalPayoffMatrix


class GameTheoreticNashArbiter:
    """
    Minimax Nash Arbiter determining optimal clinical alert threshold tau_Nash*.
    """

    def __init__(self, payoff_config: ClinicalPayoffMatrix = None):
        if payoff_config is None:
            payoff_config = ClinicalPayoffMatrix()

        self.payoff = payoff_config
        self.payoff_matrix = self.payoff.get_payoff_matrix()
        self.tau_nash_star = self.solve_nash_equilibrium()

    def solve_nash_equilibrium(self) -> float:
        """
        Solves mixed-strategy Nash equilibrium probability p* for Diagnostic Agent A1.
        At equilibrium: p* * U11 + (1 - p*) * U21 = p* * U12 + (1 - p*) * U22
        """
        U = self.payoff_matrix
        u11, u12 = U[0, 0], U[0, 1]
        u21, u22 = U[1, 0], U[1, 1]

        # Denominator = (u11 - u12 - u21 + u22)
        denom = (u11 - u12 - u21 + u22)
        if abs(denom) < 1e-6:
            p_star = 0.5
        else:
            p_star = (u22 - u21) / denom

        return float(np.clip(p_star, 0.01, 0.99))

    def evaluate_intervention(
        self,
        predicted_hazard_prob: float,
        fuzzy_activation_logits: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates whether expected utility of intervention exceeds Nash threshold.

        Alert triggers when: E[Action] > tau_Nash*

        Args:
            predicted_hazard_prob: Probability of active tissue damage from Stage 1/2.
            fuzzy_activation_logits: Logits from TSK Neuro-Fuzzy layer.

        Returns:
            dict containing:
                - 'should_intervene': bool
                - 'expected_value': float
                - 'tau_nash_star': float
        """
        p = predicted_hazard_prob
        tau = self.tau_nash_star

        # Expected value of intervention action
        expected_value = p * self.payoff_matrix[0, 0] + (1.0 - p) * self.payoff_matrix[0, 1]
        should_intervene = bool(p > tau)

        return {
            'should_intervene': should_intervene,
            'expected_value': expected_value,
            'tau_nash_star': tau,
            'hazard_probability': p
        }
