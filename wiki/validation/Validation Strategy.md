---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/AGENTS.md
  - FS/MODEL4_TUNING_RUNBOOK.md
  - FS/src/flashsim/model4_evaluation.py
---

# Validation Strategy

## Training validation

Weighted validation NLL is computed each epoch. Within one preprocessing pipeline it selects the best checkpoint and hyperparameters. Generalization gap, learning rate, gradient norm, non-finite batches, and batch-weight jitter diagnose optimization stability.

## Generated validation

Each pipeline finalist is sampled and inverse transformed into common physical coordinates. This stage selects A/B/C using:

- hard physics and guard gates;
- [[C2ST]] separation AUC;
- FGD and covariance distance;
- weighted KS and normalized weighted Wasserstein;
- [[Correlation Validation]];
- [[Tail Validation]] and [[CCDF]];
- rejection fraction and weight normalization.

## Final test

The held-out test is used once after the choice is frozen. It reports performance and must not change preprocessing, guard policy, architecture, checkpoint, or hyperparameters.

Cross-pipeline NLL comparison is invalid because the coordinate transforms and Jacobians differ. Physical-space generated-validation is the common comparison domain.

See [[Post-training Evaluation]] for the implemented commands, artifacts, global
multivariate metrics, and train-defined tail diagnostics.
