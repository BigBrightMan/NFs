---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/nf.py
  - FS/configs/model4/tuning_spaces/oaat_v1.yaml
---

# Rational-Quadratic Spline

Rational-quadratic spline transforms are flexible monotonic invertible maps used inside the autoregressive NF. They can model nonlinear marginal shapes more efficiently than an affine transform while preserving exact density evaluation.

The Model 4 baseline family uses RQ splines. The tuning study varies transform count, hidden width, block count, and learning rate while keeping the statistical target fixed. See [[Normalizing Flows]] and [[Model 4 Selection 2022-2025]].
