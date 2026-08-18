"""
Unit tests for Agentic Optimization & Explainability Layer.
Tests Payoff Matrix, Nash Arbiter, Lemke-Howson equilibrium solver, and LLM Explainer.
"""

import numpy as np
import pytest

from models.agentic.payoff_matrix import ClinicalPayoffMatrix
from models.agentic.nash_arbiter import GameTheoreticNashArbiter
from models.agentic.llm_explainer import ClinicalLLMExplainer


def test_payoff_matrix():
    matrix = ClinicalPayoffMatrix(cost_saved=15000.0, cost_visit=150.0, cost_fatigue=50.0, cost_amputation=50000.0)
    U = matrix.get_payoff_matrix()
    assert U.shape == (2, 2)
    assert U[0, 0] == 14850.0  # 15000 - 150
    assert U[0, 1] == -50.0
    assert U[1, 0] == -50000.0
    assert U[1, 1] == 0.0


def test_nash_arbiter_equilibrium_solver():
    arbiter = GameTheoreticNashArbiter()
    p_star = arbiter.tau_nash_star
    # Nash equilibrium threshold p* must be bounded in (0, 1)
    assert 0.0 < p_star < 1.0


def test_nash_arbiter_intervention():
    arbiter = GameTheoreticNashArbiter()

    # High risk probability -> Should intervene
    res1 = arbiter.evaluate_intervention(predicted_hazard_prob=0.85, fuzzy_activation_logits=np.zeros(6))
    assert res1['should_intervene']

    # Low risk probability -> Should hold
    res2 = arbiter.evaluate_intervention(predicted_hazard_prob=0.01, fuzzy_activation_logits=np.zeros(6))
    assert not res2['should_intervene']


def test_llm_explainer_report_generation():
    explainer = ClinicalLLMExplainer()
    report = explainer.generate_clinical_report(
        predicted_class_idx=0,  # DFU
        confidence=0.92,
        rule_activations=[0.8, 0.2, 0.1],
        mfi_score=1.45,
        sii_score=0.82,
        oam_score=0.91,
        tissue_percentages={'granulation': 70.0, 'slough': 20.0, 'necrotic': 10.0}
    )

    assert "primary_diagnosis" in report
    assert "icd10_code" in report
    assert "texas_wagner_stage" in report
    assert "differential_rationale" in report
    assert "offloading_prescription" in report

    assert report["primary_diagnosis"] == "Diabetic Foot Ulcer (DFU)"
    assert "E11.621" in report["icd10_code"]
