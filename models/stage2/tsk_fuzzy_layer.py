"""
Stage 2: Differentiable First-Order Takagi-Sugeno-Kang (TSK) Neuro-Fuzzy Layer.
Encodes 24 Ontological Differential Podiatry Rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableTSKFuzzyLayer(nn.Module):
    """
    Differentiable First-Order TSK Neuro-Fuzzy Layer for Explainable Podiatric Differential Reasoning.
    """

    def __init__(self, in_features: int = 128, num_rules: int = 24, num_classes: int = 6):
        super().__init__()

        self.in_features = in_features
        self.num_rules = num_rules
        self.num_classes = num_classes

        # Gaussian Membership Function Centers (c_ri) and Widths (sigma_ri)
        self.centers = nn.Parameter(torch.randn(num_rules, in_features) * 0.1)
        self.sigmas = nn.Parameter(torch.ones(num_rules, in_features) * 1.0)

        # First-Order Consequent Parameters (p_r for slope, q_r for offset per rule per class)
        self.consequent_p = nn.Parameter(torch.randn(num_rules, num_classes, in_features) * 0.01)
        self.consequent_q = nn.Parameter(torch.zeros(num_rules, num_classes))

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input fused feature tensor Z_fused of shape (B, in_features) e.g., (B, 128)

        Returns:
            dict containing:
                - 'diagnosis_logits': (B, num_classes) differential diagnosis probability logits
                - 'rule_activations': (B, num_rules) normalized firing strength w_r_bar for explainability
        """
        batch_size = x.size(0)

        # Expand x for rules: (B, 1, M)
        x_exp = x.unsqueeze(1)  # (B, 1, M)

        # 1. Antecedent Gaussian Membership Grade: μ_ri(x_i)
        diff = (x_exp - self.centers.unsqueeze(0)) / (self.sigmas.unsqueeze(0).abs() + 1e-5)
        membership = torch.exp(-0.5 * (diff ** 2))  # (B, R, M)

        # 2. Rule Firing Strength: w_r(x) = mean over antecedents
        firing_strength = torch.mean(membership, dim=-1)  # (B, R)

        # Normalized Firing Strength: w_r_bar
        firing_sum = torch.sum(firing_strength, dim=-1, keepdim=True) + 1e-8
        w_bar = firing_strength / firing_sum  # (B, R)

        # 3. First-Order Consequent Functions: f_r(x) = p_r^T * x + q_r
        # consequent_p: (R, C, M), x: (B, M) -> linear_term: (B, R, C)
        linear_term = torch.einsum('rcm,bm->brc', self.consequent_p, x)
        consequent_out = linear_term + self.consequent_q.unsqueeze(0)  # (B, R, C)

        # 4. Final Fuzzy Output Aggregation: Y_hat = ∑ w_bar_r * f_r(x)
        fuzzy_output = torch.sum(w_bar.unsqueeze(-1) * consequent_out, dim=1)  # (B, C)

        return {
            'diagnosis_logits': fuzzy_output,
            'rule_activations': w_bar
        }
