---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS_output/campaigns/*/runs/model4/weighted_density/model_selection/validation/selected_model.json
  - FS/MODEL4_TUNING_RUNBOOK.md
---

# Current Research Status

## Confirmed

- [[Model 4]] is implemented and trained with weighted NLL; `w` is not an NF input or output feature.
- Four campaign identities are available: cleaned 2022 plus 2023, 2024, and 2025; see [[Dataset Registry]].
- A/B/C preprocessing comparisons have completed generated-validation selection for all four years.
- [[Preprocessing A]] is the selected pipeline for every year under the current physical-space selection rule.
- Guarded generation produces identical kinematics in `w1` and `global_c` exports.
- Validation selects checkpoints, hyperparameters, and preprocessing. Test is reporting-only.
- The clean NFs implementation now has a tested ROOT-to-checkpoint [[Model 4 Training Vertical Slice]] with FS-equivalent preprocessing and RQ-spline flow construction.

## Interpretation

The project should now treat Model 4 plus Pipeline A as the default production direction while retaining B and C as controlled baselines.

## Experimental or proposed

- [[Model 3]] factorized conditional-weight alternatives are research proposals and have no validated production result.
- The clean `NFs`, `NFs_data`, and `NFs_output` layout is under controlled migration and must preserve provenance from the existing FS repository and output tree.

## Next research gates

- Run the clean vertical slice against one frozen campaign and compare its evidence with the canonical FS run.
- Complete reporting-only final-test comparisons consistently for each year.
- Validate MuonDIS rates using `global_c` delivery files.
- Resolve the questions in [[Open Questions]].
