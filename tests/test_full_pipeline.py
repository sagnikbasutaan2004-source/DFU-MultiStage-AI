"""
Integration Smoke Test for End-to-End Cascading Pipeline.
Validates Stage 1 -> Trigger Gate -> Stage 2 -> Stage 3 -> Agentic Nash Arbiter & Report.
"""

import torch
import pytest

from inference.full_pipeline import FullDFUSystemPipeline


def test_full_pipeline_end_to_end():
    pipeline = FullDFUSystemPipeline(device="cpu")

    # Generate synthetic input tensors
    telemetry = torch.randn(1, 18, 100)        # 18 channels, 100 samples
    rgb_img = torch.randn(1, 5, 128, 128)        # 5-channel RGB+a*+EI
    thermal_map = torch.randn(1, 1, 128, 128)    # 1-channel LWIR thermal map
    clinical_labs = torch.randn(1, 10)          # 10 clinical lab values

    img0 = torch.randn(1, 3, 128, 128)
    img1 = torch.randn(1, 3, 128, 128)

    # Case A: Thermal Trigger (ΔT = 2.8°C > 2.2°C)
    results = pipeline.process_patient_session(
        insole_telemetry=telemetry,
        delta_t_celsius=2.8,
        rgb_wound_img=rgb_img,
        thermal_map=thermal_map,
        clinical_labs=clinical_labs,
        baseline_wound_img=img0,
        followup_wound_img=img1
    )

    assert results['stage1_triggered']
    assert "stage2_diagnosis_class" in results
    assert "agentic_arbiter" in results
    assert "clinical_report" in results

    report = results['clinical_report']
    assert "primary_diagnosis" in report
    assert "icd10_code" in report
    assert "differential_rationale" in report
    assert "offloading_prescription" in report
