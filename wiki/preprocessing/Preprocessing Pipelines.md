---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/PREPROCESSING_PIPELINES.md
  - FS/src/flashsim/preprocessing.py
---

# Preprocessing Pipelines

Pipelines A, B, and C are controlled alternatives for transforming the same physical variables. They are not different model families. All are fitted on train only, reused unchanged for validation/test, and end with standardization.

| Feature | [[Preprocessing A|A]] | [[Preprocessing B|B]] | [[Preprocessing C|C]] |
|---|---|---|---|
| `x,y,z` | MinMax, clip, logit | identity | median/IQR, asinh |
| `px,py` | signed log1p | scaled asinh | zero-centered IQR, asinh |
| `E` | Box–Cox | log | shifted log, robust asinh |
| `pz` | Box–Cox | log | log, robust asinh |
| `t` | identity | identity | identity |
| final step | standardize | standardize | standardize |

The transformation is part of model identity and provenance. A preprocessor fitted on one campaign must not be reused with a different dataset fingerprint.

Training NLL must only rank hyperparameters within the same pipeline. The A/B/C winner is selected using generated-validation metrics in common physical coordinates. See [[Why Preprocessing A]].
