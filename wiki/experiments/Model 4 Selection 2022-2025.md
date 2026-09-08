---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS_output/campaigns/fluka2022_muons_down_tclean_v1/runs/model4/weighted_density/model_selection/validation/selected_model.json
  - FS_output/campaigns/fluka2023_muons_down/runs/model4/weighted_density/model_selection/validation/selected_model.json
  - FS_output/campaigns/fluka2024_muons_up/runs/model4/weighted_density/model_selection/validation/selected_model.json
  - FS_output/campaigns/fluka2025_muons_horizontal/runs/model4/weighted_density/model_selection/validation/selected_model.json
---

# Model 4 Selection 2022-2025

The selection rule applies hard physics/guard gates, then lexicographically compares separation AUC, FGD, covariance distance, maximum weighted KS, normalized weighted Wasserstein, and rejection fraction.

| Year | Selected pipeline | Trial | C2ST separation AUC | Maximum weighted KS | Rejection |
|---:|---|---|---:|---:|---:|
| 2022 | A | baseline | 0.508834 | 0.006571 | 0.2265% |
| 2023 | A | baseline | 0.510964 | 0.008257 | 0.3055% |
| 2024 | A | baseline | 0.507903 | 0.011426 | 0.4645% |
| 2025 | A | lr-high (`1e-3`) | 0.511678 | 0.010909 | 0.5548% |

## Interpretation

[[Preprocessing A]] is the current production default for [[Model 4]]. This is an empirical conclusion for the four frozen campaigns, not a theorem that A will win on every future dataset.

For 2025, B and C finalists exceeded the 1% rejection hard gate. For 2022, B had slightly better values for some scalar distances, but A had a C2ST result closer to 0.5 and substantially lower rejection, so it won under the declared rule.

The held-out test must not be used to revise these choices. See [[Validation Strategy]].
