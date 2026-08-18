"""
Agentic Optimization Layer: Payoff Matrix & Clinical Cost Formulation.
Zero-Sum Minimax Payoff Matrix between Diagnostic Agent (A1) and Auditor Agent (A2).
"""

import numpy as np


class ClinicalPayoffMatrix:
    """
    Formulates clinical cost payoff matrix:
    A1 (Diagnostic Agent): Action 0 = Intervene / Alert, Action 1 = Hold / Monitor
    A2 (Auditor Agent):    State 0 = Ulcer/Infection Present, State 1 = Benign/No Ulcer
    """

    def __init__(
        self,
        cost_saved: float = 15000.0,      # Saved amputation / emergency care cost ($)
        cost_visit: float = 150.0,         # Outpatient clinical visit cost ($)
        cost_fatigue: float = 50.0,        # False alarm clinical alert fatigue cost ($)
        cost_amputation: float = 50000.0   # Catastrophic cost of missed early ulcer / limb loss ($)
    ):
        self.cost_saved = cost_saved
        self.cost_visit = cost_visit
        self.cost_fatigue = cost_fatigue
        self.cost_amputation = cost_amputation

    def get_payoff_matrix(self) -> np.ndarray:
        """
        Returns 2x2 payoff matrix U(A1, A2) for Diagnostic Agent A1:

                    Ulcer Present (S0)      No Ulcer (S1)
        Intervene   (C_saved - C_visit)     (-C_fatigue)
        Hold        (-C_amputation)           (0.0)
        """
        u11 = self.cost_saved - self.cost_visit
        u12 = -self.cost_fatigue
        u21 = -self.cost_amputation
        u22 = 0.0

        return np.array([
            [u11, u12],
            [u21, u22]
        ], dtype=np.float64)
