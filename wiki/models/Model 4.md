---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/model4.py
  - FS/src/flashsim/model4_generation.py
  - FS/MODEL4_RUNBOOK.md
  - FS/MODEL4_TUNING_RUNBOOK.md
---

# Model 4

## Purpose

Generate the unbiased, FLUKA-weighted physical kinematic density without learning `w` as a feature.

## Statistical target

Model 4 minimizes weighted NLL; see [[Maximum Likelihood]] and [[Weighted Density]]. `w` is a positive loss coefficient only.

## Current architecture

- Shared kinematic RQ-spline [[Normalizing Flows|NF]].
- `drop_z_E`: the NF learns `x, y, pz, px, py, t`.
- `E` is reconstructed from the muon mass shell.
- `z` is reconstructed from train-fitted scoring-plane coefficients.
- Training seed 42 and batch size 2048 are the established baseline conditions.

## Preprocessing

A/B/C are supported. Completed physical-space generated-validation selected [[Preprocessing A]] for 2022–2025; see [[Model 4 Selection 2022-2025]].

## Output weights

Generation creates identical kinematics in `w1` and `global_c` ROOT exports. See [[w1 and global_c]].

## Validation

Checkpoint selection uses weighted validation NLL within one preprocessing coordinate system. Cross-preprocessing selection uses generated samples in physical space, including C2ST, weighted KS/Wasserstein, covariance/FGD, rejection fraction, correlations, and tails.

## Known risks

- Tail extrapolation and rejection rate.
- Geometry bounds not fully interchangeable with data-derived robust bounds.
- Cross-pipeline NLL values are not directly comparable.
