---
status: confirmed
last_verified: 2026-09-09
sources:
  - FS/ARCHITECTURE.md
  - FS/MODEL4_TUNING_RUNBOOK.md
---

# End-to-End Workflow

```text
FLUKA ROOT
  -> selection and dataset manifest
  -> immutable 70/15/15 split
  -> fit A/B/C preprocessing on train only
  -> weighted-density EDA
  -> train Model 4 with weighted NLL
  -> select checkpoint with validation NLL
  -> generate validation samples
  -> physical-space generated-validation metrics
  -> freeze preprocessing and hyperparameters
  -> one reporting-only test generation/evaluation
  -> guarded 100k-event global_c delivery for MuonDIS
```

The train split fits preprocessing and the generation guard. The validation split selects model choices. The test split must not change preprocessing, guards, architecture, or hyperparameters.

Generated candidates are inverse transformed, deterministic `z` and `E` are reconstructed for the `drop_z_E` design, and guard failures are rejected and resampled until the requested accepted count is reached.

Related pages: [[Dataset Registry]], [[Preprocessing Pipelines]], [[Model 4]], [[Validation Strategy]], [[Why Rejection Sampling]], and [[w1 and global_c]].

The maintained implementation lives entirely in `flashsim_nf`: training and
checkpointing, legacy A/B/C metadata adaptation, physical reconstruction,
guarded generation, equal-weight exports, generated-validation metrics/plots,
validation-only winner freezing, final-test gating, and CERN submission. It
does not import the old `flashsim` package at runtime.
