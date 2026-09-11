# SemiGAN-MelanoPath

## Semi-supervised Learning for Label-Scarce Histopathology

SemiGAN-MelanoPath is a research framework investigating
semi-supervised deep learning for histopathology classification
under extreme label scarcity.

The project combines:

- Generative Adversarial Networks (GANs)
- Virtual Adversarial Training (VAT)
- Self-supervised rotation prediction
- Feature matching
- Patient-level data splitting
- Monte Carlo Dropout uncertainty estimation
- Cross-domain evaluation

## Research Objective

The primary objective is to investigate whether unlabeled
histopathology images can improve classification performance when
expert-labeled data are severely limited.

## Experimental Framework

The project will evaluate:

- E1 — Label scarcity
- E2 — Cross-domain transfer
- E3 — Annotation budget
- E4 — Ablation studies
- E5 — Uncertainty and calibration

## Implementation

The framework is implemented in Python using PyTorch and is designed
for reproducible research experiments.

## Status

Research implementation in progress.