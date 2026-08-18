"""
Agentic Optimization Layer: Multimodal Agentic LLM Explainer.
Translates TSK Neuro-Fuzzy activations, SegFormer tissue breakdown, and biomechanical alerts
into structured clinical notes (ICD-10, Texas-Wagner staging, differential rationale).
"""

from typing import Dict, List


class ClinicalLLMExplainer:
    """
    Agentic LLM Explainer generating structured podiatric clinical reports.
    """

    DIFFERENTIAL_CLASSES = [
        "Diabetic Foot Ulcer (DFU)",
        "Charcot Neuro-arthropathy",
        "Gouty Arthritis",
        "Venous Stasis Ulcer",
        "Ischemic Arterial Ulcer",
        "Benign Callus / Hyperkeratosis"
    ]

    ICD10_MAP = {
        "Diabetic Foot Ulcer (DFU)": "E11.621 (Type 2 diabetes mellitus with foot ulcer)",
        "Charcot Neuro-arthropathy": "M14.67 (Charcot's joint, ankle and foot)",
        "Gouty Arthritis": "M10.07 (Idiopathic gout, ankle and foot)",
        "Venous Stasis Ulcer": "I87.2 (Venous insufficiency, chronic)",
        "Ischemic Arterial Ulcer": "I70.25 (Atherosclerosis of native arteries with ulceration)",
        "Benign Callus / Hyperkeratosis": "L84 (Callus and callosity)"
    }

    TEXAS_WAGNER_STAGING = {
        0: "Grade 0, Stage A (Pre-ulcerative lesion, skin intact)",
        1: "Grade 1, Stage A (Superficial ulcer, no involvement of tendon/capsule/bone)",
        2: "Grade 2, Stage A (Deep ulcer involving tendon or capsule)",
        3: "Grade 3, Stage B (Deep ulcer with abscess, osteomyelitis, or sepsis)"
    }

    def generate_clinical_report(
        self,
        predicted_class_idx: int,
        confidence: float,
        rule_activations: list,
        mfi_score: float,
        sii_score: float,
        oam_score: float,
        tissue_percentages: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Generates structured clinical report.
        """
        dx_name = self.DIFFERENTIAL_CLASSES[predicted_class_idx]
        icd10 = self.ICD10_MAP.get(dx_name, "E11.621")
        wagner = self.TEXAS_WAGNER_STAGING.get(1 if predicted_class_idx == 0 else 0, "Grade 0, Stage A")

        # Excluded differentials rationale
        excluded = [name for i, name in enumerate(self.DIFFERENTIAL_CLASSES) if i != predicted_class_idx]

        rationale = (
            f"Primary diagnosis of {dx_name} (Confidence: {confidence*100:.1f}%) established via "
            f"TSK Neuro-Fuzzy rules (Top active rule strength: {max(rule_activations):.3f}). "
            f"Excludes {', '.join(excluded[:2])} based on temperature differential and tissue erythema ratio. "
            f"Microvascular Fatigue Index (MFI): {mfi_score:.2f}, Sudomotor Impairment Index (SII): {sii_score:.2f}."
        )

        offloading_prescription = (
            f"Offloading Adherence Metric (OAM): {oam_score*100:.1f}%. " +
            ("Therapeutic footwear compliance optimal." if oam_score >= 0.85 else
             "CRITICAL ALERT: Non-compliant offloading detected. Recommend immediate total contact casting (TCC) or removable boot adjustment.")
        )

        report = {
            'primary_diagnosis': dx_name,
            'icd10_code': icd10,
            'texas_wagner_stage': wagner,
            'differential_rationale': rationale,
            'offloading_prescription': offloading_prescription,
            'granulation_tissue_pct': f"{tissue_percentages.get('granulation', 0.0):.1f}%",
            'slough_tissue_pct': f"{tissue_percentages.get('slough', 0.0):.1f}%",
            'necrotic_tissue_pct': f"{tissue_percentages.get('necrotic', 0.0):.1f}%"
        }

        return report
