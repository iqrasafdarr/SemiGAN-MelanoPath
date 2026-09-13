# SemiGAN-MelanoPath Experiment Framework

## Main Experiments

- E1: Supervised baseline
- E2: SemiGAN baseline
- E3: SemiGAN + VAT
- E4: SemiGAN + VAT + Rotation SSL
- E5: Full SemiGAN-MelanoPath

## Label Budgets

Experiments are evaluated under:

- 1%
- 5%
- 10%
- 20%
- 100%

The project uses patient-level splitting to prevent patient leakage.

Because patient-level class balancing may require more patients than the requested
nominal budget, every experiment must report both:

1. requested label budget
2. actual labeled patient percentage
3. actual labeled image percentage

These values must never be silently conflated.
