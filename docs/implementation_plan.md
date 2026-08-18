# DFU-MultiStage-AI — Implementation Plan

> This document tracks the full implementation roadmap for the Diabetic Foot Ulcer multi-stage AI system.
> See the main [README](../README.md) for project overview and architecture.

## Phased Build Order

### Phase 1: Foundation & Data Infrastructure

| # | Task | Status |
|---|---|---|
| 1.1 | Project scaffolding (`pyproject.toml`, directory tree, dependencies) | ⬜ |
| 1.2 | Configuration system (YAML configs with `pyyaml`) | ⬜ |
| 1.3 | Signal processing utilities (windowing, filtering, PTI) | ⬜ |
| 1.4 | Color science utilities (RGB → CIE L*a*b*, Erythema Index) | ⬜ |
| 1.5 | Synthetic insole data generator (16-ch FSR + thermistor + RH%) | ⬜ |
| 1.6 | Common training utilities (EarlyStopping, LR scheduling, metrics) | ⬜ |

### Phase 2: Stage 1 — Pre-Ulcerative Early Warning Engine

| # | Task | Status |
|---|---|---|
| 2.1 | Insole dataset & data loader (windowed sequences) | ⬜ |
| 2.2 | 1D-CNN encoder (multi-channel Conv1D feature extractor) | ⬜ |
| 2.3 | Temporal Transformer encoder (positional encoding + MHA) | ⬜ |
| 2.4 | Combined Stage 1 model (MFI + SII computation) | ⬜ |
| 2.5 | Trigger logic gate (ΔT > 2.2°C OR hazard > 0.65) | ⬜ |
| 2.6 | Training loop for Stage 1 | ⬜ |
| 2.7 | Unit tests for Stage 1 | ⬜ |

### Phase 3: Stage 2 — Clinical Formation & Differential Diagnostics

#### Phase 3A: RGB Vision Head
| # | Task | Status |
|---|---|---|
| 3A.1 | SegFormer-B4 tissue segmentation head | ⬜ |
| 3A.2 | Erythema Index (EI) computation module | ⬜ |
| 3A.3 | RGB dataset & augmentation pipeline | ⬜ |
| 3A.4 | Training loop for SegFormer | ⬜ |

#### Phase 3B: Thermal Head
| # | Task | Status |
|---|---|---|
| 3B.1 | ResNet-34 + FPN thermal feature extractor | ⬜ |
| 3B.2 | Thermal dataset & radiometric transforms | ⬜ |
| 3B.3 | Training loop for thermal head | ⬜ |

#### Phase 3C: Tabular Head
| # | Task | Status |
|---|---|---|
| 3C.1 | TabNet encoder for clinical labs | ⬜ |
| 3C.2 | Clinical tabular dataset loader | ⬜ |
| 3C.3 | Training loop for TabNet | ⬜ |

#### Phase 3D: Gait Head
| # | Task | Status |
|---|---|---|
| 3D.1 | Bidirectional GRU for gait dynamics | ⬜ |

#### Phase 3E: Multi-Modal Fusion & Reasoning
| # | Task | Status |
|---|---|---|
| 3E.1 | Cross-Modal Multi-Head Attention (d=128) | ⬜ |
| 3E.2 | Differentiable TSK Neuro-Fuzzy layer (24 rules) | ⬜ |
| 3E.3 | End-to-end Stage 2 pipeline assembly | ⬜ |
| 3E.4 | Training loop for full fusion | ⬜ |
| 3E.5 | Unit tests for Stage 2 | ⬜ |

### Phase 4: Stage 3 — Post-Formation Healing Monitor

| # | Task | Status |
|---|---|---|
| 4.1 | Siamese ResNet-50 granulation tracker | ⬜ |
| 4.2 | Extended Kalman Filter (area + velocity state-space) | ⬜ |
| 4.3 | Offloading Adherence Metric (OAM) | ⬜ |
| 4.4 | Healing dataset (longitudinal image pairs) | ⬜ |
| 4.5 | Training loop for Siamese network | ⬜ |
| 4.6 | Stage 3 pipeline assembly | ⬜ |
| 4.7 | Unit tests for Stage 3 | ⬜ |

### Phase 5: Agentic Optimization & Explainability

| # | Task | Status |
|---|---|---|
| 5.1 | Payoff matrix & cost configuration | ⬜ |
| 5.2 | Lemke-Howson Nash equilibrium solver | ⬜ |
| 5.3 | 2-Player Minimax Arbiter | ⬜ |
| 5.4 | LLM Explainer integration (Med-Gemma / LLaVA-Med) | ⬜ |
| 5.5 | Clinical note generation (ICD-10, Texas-Wagner) | ⬜ |
| 5.6 | Unit tests for Arbiter | ⬜ |

### Phase 6: Integration, Edge & Deployment

| # | Task | Status |
|---|---|---|
| 6.1 | Full cascading inference pipeline | ⬜ |
| 6.2 | BLE data stream simulator | ⬜ |
| 6.3 | TFLite Micro conversion (Stage 1) | ⬜ |
| 6.4 | Integration tests | ⬜ |
| 6.5 | Demo notebooks | ⬜ |
