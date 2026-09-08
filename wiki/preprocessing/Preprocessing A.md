---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/PREPROCESSING_PIPELINES.md
  - FS_output/campaigns/*/runs/model4/weighted_density/model_selection/validation/selected_model.json
---

# Preprocessing A

Pipeline A is the collaborator-style feature-specific transformation:

- `x,y,z`: train MinMax, clipping, logit, then standardization.
- `px,py`: signed `log1p`, then standardization.
- `E,pz`: fitted Box–Cox, then standardization.
- `t`: identity followed by standardization.

It maps bounded spatial features to the real line and compresses momentum/energy tails. Its main trade-off is dependence on train extrema: outliers can distort min/max, and validation/test rows outside the train range are clipped.

Under current [[Model 4]] generated-validation, A is selected for every campaign in [[Model 4 Selection 2022-2025]]. This makes it the current production default, not a universal guarantee for future datasets.
