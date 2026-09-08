---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/model4.py
  - FS/MODEL4_RUNBOOK.md
---

# Why Weighted Density

## Observation

FLUKA `w` is a sampling weight rather than a kinematic feature, and similar kinematics may carry different individual weights.

## Evidence

[[Model 1]] must model a difficult spiky weight distribution. [[Model 2]] predicts a conditional mean and cannot reproduce irreducible event-level variation.

## Decision

Use [[Model 4]] weighted NLL so `w` controls each training row's contribution without entering or leaving the NF.

## Reason

The generated kinematics directly follow the weighted physical density; downstream samples need only a global normalization constant.

## Trade-off

The effective training sample size is lower than the row count, and batches can have weight jitter. Weighted EDA and stability diagnostics are mandatory.

## Open question

MuonDIS rate closure must still be verified using the final `global_c` delivery.
