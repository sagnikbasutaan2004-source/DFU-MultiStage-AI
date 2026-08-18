"""
End-to-End Cascading System Topology Pipeline.
Connects Stage 1 -> Trigger Gate -> Stage 2 -> Stage 3 -> Agentic Nash Arbiter & LLM Explainer.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch

from models.stage1.insole_model import InsoleEarlyWarningModel
from models.stage1.trigger_logic import Stage1TriggerGate
from models.stage2.stage2_pipeline import Stage2DifferentialDiagnosticEngine
from models.stage3.stage3_pipeline import Stage3HealingMonitorPipeline
from models.agentic.nash_arbiter import GameTheoreticNashArbiter
from models.agentic.llm_explainer import ClinicalLLMExplainer


class FullDFUSystemPipeline:
    """
    Cascading End-to-End Multi-Stage AI Pipeline.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Stage 1: Pre-Ulcerative Early Warning
        self.stage1_model = InsoleEarlyWarningModel(in_channels=18, d_model=128).to(self.device)
        self.stage1_trigger = Stage1TriggerGate(delta_t_threshold=2.2, hazard_threshold=0.65)

        # Stage 2: Clinical Formation & Differential Diagnostic Engine
        self.stage2_engine = Stage2DifferentialDiagnosticEngine(feature_dim=128).to(self.device)

        # Stage 3: Post-Formation Healing Monitor
        self.stage3_monitor = Stage3HealingMonitorPipeline(feature_dim=128)

        # Agentic Layer
        self.arbiter = GameTheoreticNashArbiter()
        self.explainer = ClinicalLLMExplainer()

        self.stage1_model.eval()
        self.stage2_engine.eval()
        self.stage3_monitor.eval()

    def process_patient_session(
        self,
        insole_telemetry: torch.Tensor,      # (1, 18, 100)
        delta_t_celsius: float,
        rgb_wound_img: torch.Tensor,          # (1, 5, H, W)
        thermal_map: torch.Tensor,            # (1, 1, H, W)
        clinical_labs: torch.Tensor,          # (1, 10)
        baseline_wound_img: Optional[torch.Tensor] = None, # (1, 3, H, W)
        followup_wound_img: Optional[torch.Tensor] = None   # (1, 3, H, W)
    ) -> Dict:
        """
        Executes end-to-end cascading inference across all stages.
        """
        insole_telemetry = insole_telemetry.to(self.device)
        rgb_wound_img = rgb_wound_img.to(self.device)
        thermal_map = thermal_map.to(self.device)
        clinical_labs = clinical_labs.to(self.device)

        # ------------------- STAGE 1 -------------------
        with torch.no_grad():
            stage1_out = self.stage1_model(insole_telemetry)

        hazard_score = float(stage1_out['hazard_score'].cpu().item())
        mfi_vec = stage1_out['mfi_pred'].cpu().numpy().flatten()
        sii_val = float(stage1_out['sii_pred'].cpu().item())

        trigger_res = self.stage1_trigger.evaluate(delta_t_celsius=delta_t_celsius, hazard_score=hazard_score)

        results = {
            'stage1_hazard_score': hazard_score,
            'stage1_mfi_max': float(np.max(mfi_vec)),
            'stage1_sii': sii_val,
            'stage1_triggered': trigger_res['triggered'],
            'trigger_details': trigger_res
        }

        if not trigger_res['triggered']:
            results['status'] = "Patient clear. Continuous edge insole telemetry monitoring active."
            return results

        # ------------------- STAGE 2 -------------------
        with torch.no_grad():
            stage2_out = self.stage2_engine(rgb_wound_img, thermal_map, insole_telemetry, clinical_labs)

        diag_logits = stage2_out['diagnosis_logits'].cpu().numpy().flatten()
        diag_probs = torch.softmax(torch.from_numpy(diag_logits), dim=-1).numpy()
        pred_class_idx = int(np.argmax(diag_probs))
        confidence = float(diag_probs[pred_class_idx])

        rule_activations = stage2_out['rule_activations'].cpu().numpy().flatten().tolist()

        results['stage2_diagnosis_class'] = self.explainer.DIFFERENTIAL_CLASSES[pred_class_idx]
        results['stage2_confidence'] = confidence
        results['stage2_rule_activations'] = rule_activations

        # ------------------- STAGE 3 -------------------
        oam_score = 0.92  # Default therapeutic compliance
        if baseline_wound_img is not None and followup_wound_img is not None:
            baseline_wound_img = baseline_wound_img.to(self.device)
            followup_wound_img = followup_wound_img.to(self.device)

            with torch.no_grad():
                stage3_out = self.stage3_monitor(baseline_wound_img, followup_wound_img)

            results['stage3_wound_distance'] = float(stage3_out['euclidean_distance'].cpu().item())
            results['stage3_area_ratio'] = float(stage3_out['predicted_area_ratio'].cpu().item())

        # ------------------- AGENTIC LAYER -------------------
        arbiter_res = self.arbiter.evaluate_intervention(predicted_hazard_prob=confidence, fuzzy_activation_logits=diag_logits)
        results['agentic_arbiter'] = arbiter_res

        # Generate structured clinical report
        report = self.explainer.generate_clinical_report(
            predicted_class_idx=pred_class_idx,
            confidence=confidence,
            rule_activations=rule_activations,
            mfi_score=float(np.max(mfi_vec)),
            sii_score=sii_val,
            oam_score=oam_score,
            tissue_percentages={'granulation': 65.0, 'slough': 25.0, 'necrotic': 10.0}
        )
        results['clinical_report'] = report

        return results
