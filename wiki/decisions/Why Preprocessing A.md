---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS_output/campaigns/*/runs/model4/weighted_density/model_selection/validation/selected_model.json
---

# Why Preprocessing A

## Observation

Training NLL across A/B/C is not directly comparable because each pipeline changes coordinates differently.

## Evidence

Completed generated-validation in physical space selected A for 2022, 2023, 2024, and 2025. See [[Model 4 Selection 2022-2025]].

## Decision

Use [[Preprocessing A]] as the default Model 4 production pipeline. Retain B as a simple baseline and C as a robust ablation.

## Trade-off

A relies on train min/max for spatial logit transforms and can clip out-of-range validation/test rows. Clip fractions and tail behavior must remain visible in EDA.

## Open question

Future detector geometries or datasets with different tails require a new A/B/C validation rather than automatic reuse of A.
