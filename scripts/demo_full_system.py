"""
Interactive End-to-End System Demonstration Script.
Demonstrates Stage 1 -> Trigger Gate -> Stage 2 -> Stage 3 -> Agentic Nash Arbiter -> LLM Explainer Report.
"""

import os
import json
import torch
import numpy as np

from inference.full_pipeline import FullDFUSystemPipeline


def main():
    print("=" * 80)
    print(" [DFU MULTI-STAGE AI SYSTEM] -- END-TO-END DEMO RUN")
    print("=" * 80)

    # Initialize full cascading system pipeline
    pipeline = FullDFUSystemPipeline(device="cpu")

    # 1. Prepare inputs
    # Telemetry: 18 channels (16 FSR, 1 Delta T, 1 RH) over 100 temporal steps (5 seconds at 20Hz)
    insole_telemetry = torch.randn(1, 18, 100)

    # Contralateral thermal differential Delta T = 2.6 deg C (Triggers Stage 1 alert!)
    delta_t_celsius = 2.6

    # Vision: 5-channel RGB + a* redness + Erythema Index map (128x128)
    rgb_wound_img = torch.randn(1, 5, 128, 128)

    # Thermal map: 1-channel LWIR thermal map
    thermal_map = torch.randn(1, 1, 128, 128)

    # Clinical labs: [HbA1c=9.2%, eGFR=55, Creatinine=1.4, Glucose=180, WBC=11.5, CRP=24, ESR=35, Age=62, BMI=29.5, Duration=14]
    clinical_labs = torch.tensor([[9.2, 55.0, 1.4, 180.0, 11.5, 24.0, 35.0, 62.0, 29.5, 14.0]], dtype=torch.float32)

    # Longitudinal wound image pair for Stage 3
    baseline_img = torch.randn(1, 3, 128, 128)
    followup_img = torch.randn(1, 3, 128, 128)

    print("\n[STAGE 1] Ingesting continuous 18-channel insole telemetry stream...")
    print(f" -> Contralateral Plantar Thermistor Differential Delta T: {delta_t_celsius} deg C")

    # Run cascading pipeline
    results = pipeline.process_patient_session(
        insole_telemetry=insole_telemetry,
        delta_t_celsius=delta_t_celsius,
        rgb_wound_img=rgb_wound_img,
        thermal_map=thermal_map,
        clinical_labs=clinical_labs,
        baseline_wound_img=baseline_img,
        followup_wound_img=followup_img
    )

    print(f" -> Stage 1 Hazard Score: {results['stage1_hazard_score']:.4f}")
    print(f" -> Microvascular Fatigue Index (MFI Peak): {results['stage1_mfi_max']:.2f}")
    print(f" -> Sudomotor Impairment Index (SII): {results['stage1_sii']:.2f}")

    if results['stage1_triggered']:
        print("\n[TRIGGER ACTIVATED] Delta T > 2.2 deg C condition met! Activating Stage 2 Clinical Engine...")
        print(f"\n[STAGE 2] Multi-Modal Cross-Attention Fusion & TSK Neuro-Fuzzy Reasoning:")
        print(f" -> Differential Diagnosis: {results['stage2_diagnosis_class']}")
        print(f" -> Clinical Confidence: {results['stage2_confidence']*100:.1f}%")
        print(f" -> Top Active TSK Fuzzy Rule Activation: {max(results['stage2_rule_activations']):.4f}")

        print(f"\n[STAGE 3] Remission & Healing Monitor:")
        print(f" -> Granulation Tissue Euclidean Distance: {results.get('stage3_wound_distance', 0.0):.4f}")
        print(f" -> Predicted Area Contraction Ratio: {results.get('stage3_area_ratio', 0.0):.4f}")

        print(f"\n[AGENTIC OPTIMIZATION] Game-Theoretic Nash Arbiter:")
        arb = results['agentic_arbiter']
        print(f" -> Minimax Nash Threshold (Tau_Nash*): {arb['tau_nash_star']:.4f}")
        print(f" -> Expected Clinical Utility: ${arb['expected_value']:,.2f}")
        print(f" -> Decision: {'INTERVENE / CLINICAL ALERT' if arb['should_intervene'] else 'MONITOR'}")

        print("\n" + "=" * 80)
        print(" GENERATED CLINICAL EXPLAINER REPORT (Med-Gemma / LLaVA-Med Format)")
        print("=" * 80)
        report = results['clinical_report']
        for key, val in report.items():
            title = key.replace('_', ' ').title()
            print(f"\n- {title}:\n  {val}")
        print("=" * 80)


if __name__ == "__main__":
    main()
