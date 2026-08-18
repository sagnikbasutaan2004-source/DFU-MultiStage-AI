<p align="center">
  <h1 align="center">🦶 DFU-MultiStage-AI</h1>
  <p align="center">
    <strong>A Longitudinal, Multi-Modal AI System for Diabetic Foot Ulcer Detection, Differential Diagnosis & Healing Surveillance</strong>
  </p>
  <p align="center">
    <a href="#architecture">Architecture</a> •
    <a href="#stages">Pipeline Stages</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#mathematical-foundations">Math</a> •
    <a href="#citation">Citation</a>
  </p>
</p>

---

## Overview

Diabetic Foot Ulcers (DFUs) affect **15–25%** of diabetic patients during their lifetime, with **>50%** of ulcers becoming infected and **20%** of moderate-to-severe infections leading to amputation. This project implements a **multi-stage AI pipeline** that operates across the full clinical lifecycle — from **pre-ulcerative early warning** through **active wound differential diagnosis** to **post-formation healing surveillance** — combining edge sensing, deep learning, neuro-symbolic reasoning, and agentic AI.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     EDGE SENSING LAYER                                  │
│   16-Ch FSR Array (20Hz) │ Plantar Thermistors │ SHT40 Sudomotor RH%  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ BLE 5.2 / Edge TFLite Micro
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Pre-Ulcerative Early Warning Engine                          │
│  1D-CNN + Temporal Transformer → MFI, SII → Trigger Gate               │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ ΔT > 2.2°C OR Hazard > 0.65
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Clinical Formation & Differential Diagnostic Engine          │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌─────────────┐              │
│  │SegFormer │ │Thermal FPN│ │ TabNet   │ │ BiGRU Gait  │              │
│  │  B4+LAB  │ │ResNet-34  │ │ Clinical │ │  Dynamics   │              │
│  └────┬─────┘ └─────┬─────┘ └────┬─────┘ └──────┬──────┘              │
│       └──────────────┴───────────┴───────────────┘                     │
│                      ▼                                                  │
│       Cross-Modal Multi-Head Attention Fusion (d=128)                  │
│                      ▼                                                  │
│       Differentiable TSK Neuro-Fuzzy Layer (24 Rules)                  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Post-Formation Remission & Repair Monitor                    │
│  Siamese ResNet-50 + Extended Kalman Filter → dArea/dt, OAM            │
└──────────────────────────────┬──────────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  AGENTIC & GENERATIVE OPTIMIZATION LAYER                               │
│  2-Player Nash Arbiter │ Multimodal LLM Explainer (Med-Gemma)          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Stages

### Stage 1: Pre-Ulcerative Early Warning
- **Input:** 16-channel piezoresistive FSR (20Hz), plantar thermistors (ΔT), SHT40 sudomotor humidity (RH%)
- **Model:** 1D-CNN temporal feature extractor → Transformer encoder with multi-head attention
- **Outputs:** Microvascular Fatigue Index (MFI), Sudomotor Impairment Index (SII)
- **Trigger:** ΔT > 2.2°C OR composite hazard score > 0.65

### Stage 2: Clinical Formation & Differential Diagnostics
- **Vision Head:** SegFormer-B4 semantic segmentation (granulation, slough, necrosis) + CIE L\*a\*b\* Erythema Index
- **Thermal Head:** ResNet-34 + Feature Pyramid Network for spatial thermal asymmetry
- **Tabular Head:** TabNet with sparse attention for metabolic markers (HbA1c, eGFR, Creatinine)
- **Gait Head:** Bidirectional GRU encoding insole gait dynamics
- **Fusion:** Cross-modal multi-head attention (Q=RGB, K=[Thermal;Insole], V=Clinical)
- **Reasoning:** Differentiable first-order Takagi-Sugeno-Kang neuro-fuzzy layer with 24 ontological podiatric rules
- **Differential Dx:** DFU vs. Charcot arthropathy vs. Gout vs. Venous stasis vs. Arterial ulcer vs. Callus

### Stage 3: Post-Formation Healing Monitor
- **Tracker:** Siamese ResNet-50 with contrastive loss for granulation tissue progression
- **State Estimation:** Extended Kalman Filter tracking wound area and healing velocity (mm²/day)
- **Compliance:** Offloading Adherence Metric (OAM) from insole pressure monitoring

### Agentic Optimization Layer
- **Nash Arbiter:** 2-player zero-sum minimax game between Diagnostic Agent and Auditor Agent, solved via Lemke-Howson for optimal alert thresholds
- **LLM Explainer:** Med-Gemma / LLaVA-Med generates ICD-10 staging, offloading prescriptions, and differential rationales

## Mathematical Foundations

<details>
<summary><b>Stage 1: Sensor Mechanics</b></summary>

**Microvascular Fatigue Index (MFI):**

$$\text{MFI}(x, y) = \int_{0}^{t} \left[ P(x, y, \tau) \cdot \frac{\partial T(x, y, \tau)}{\partial \tau} \cdot \left(1 - \text{RH}_{\text{norm}}(\tau)\right) \right] d\tau$$

**Sudomotor Impairment Index (SII):**

$$\text{SII}(t) = \frac{\overline{\text{RH}}_{\text{insole}}(t) - \text{RH}_{\text{ambient}}}{\Delta \text{GaitLoad}(t)} \cdot \exp\left(-\frac{T_{\text{insole}}}{T_{\text{baseline}}}\right)$$

</details>

<details>
<summary><b>Stage 2: Fusion & Differential Reasoning</b></summary>

**Cross-Modal Attention Fusion:**

$$Z_{\text{fused}} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

**TSK Neuro-Fuzzy Layer:**

$$\hat{Y}_{\text{diagnosis}} = \sum_{r=1}^{R} \left[ \frac{w_r(\mathbf{x})}{\sum_{k=1}^R w_k(\mathbf{x})} \cdot \left(\mathbf{p}_r^T \mathbf{x} + q_r\right) \right]$$

</details>

<details>
<summary><b>Stage 3: Wound Trajectory</b></summary>

**Kalman State Space:**

$$\mathbf{x}_k = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix} \mathbf{x}_{k-1} + \mathbf{w}_k$$

**Offloading Adherence Metric:**

$$\text{OAM} = 1.0 - \frac{\sum_{t} \mathbb{I}(P_{\text{ulcer}}(t) > P_{\text{threshold}})}{\text{Total Steps}}$$

</details>

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/DFU-MultiStage-AI.git
cd DFU-MultiStage-AI

# Install dependencies with uv
pip install uv
uv sync

# Or with pip
pip install -r requirements.txt
```

## Usage

```python
# Stage 1: Pre-ulcerative early warning
from models.stage1.insole_model import InsoleEarlyWarning
from models.stage1.trigger_logic import TriggerGate

model = InsoleEarlyWarning(config="configs/stage1_insole.yaml")
mfi, sii = model(insole_data)
trigger = TriggerGate(delta_t_threshold=2.2, hazard_threshold=0.65)

if trigger(delta_t, mfi, sii):
    # Activate Stage 2 clinical assessment
    ...
```

## Project Status

| Phase | Status |
|---|---|
| Phase 1: Foundation & Data Infrastructure | 🔜 Next |
| Phase 2: Stage 1 — Pre-Ulcerative Warning | ⬜ Planned |
| Phase 3: Stage 2 — Differential Diagnostics | ⬜ Planned |
| Phase 4: Stage 3 — Healing Monitor | ⬜ Planned |
| Phase 5: Agentic Optimization | ⬜ Planned |
| Phase 6: Integration & Edge | ⬜ Planned |

## License

This project is part of ongoing research. License TBD.

## Citation

```bibtex
@software{dfu_multistage_ai_2026,
  title={DFU-MultiStage-AI: Multi-Modal Diabetic Foot Ulcer Detection and Monitoring},
  year={2026},
  url={https://github.com/<your-username>/DFU-MultiStage-AI}
}
```
