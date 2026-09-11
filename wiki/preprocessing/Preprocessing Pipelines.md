---
status: confirmed
last_verified: 2026-09-11
sources:
  - FS/PREPROCESSING_PIPELINES.md
  - FS/src/flashsim/preprocessing.py
  - src/flashsim_nf/preprocessing/pipelines.py
---

# Preprocessing Pipelines

Pipelines A, B, and C are controlled alternatives for transforming the same physical variables. They are not different model families. All are fitted on train only, reused unchanged for validation/test, and end with standardization.

Pipeline D is a proposed extension of A; see [[Extended Model 4 Experiments]].

| Feature | [[Preprocessing A|A]] | [[Preprocessing B|B]] | [[Preprocessing C|C]] | [[Preprocessing D|D]] |
|---|---|---|---|---|
| `x,y,z` | MinMax, clip, logit | identity | median/IQR, asinh | MinMax, clip, logit |
| `px,py` | signed log1p | scaled asinh | zero-centered IQR, asinh | signed log1p |
| `E` | Box–Cox | log | shifted log, robust asinh | **log** |
| `pz` | Box–Cox | log | log, robust asinh | **log** |
| `t` | identity | identity | identity | identity |
| final step | standardize | standardize | standardize | standardize |

D is A with the two energy variables on natural log. Because Model 4 trains on `pz`
under `drop_ze` and on `E` under `drop_z_pz` but never on both, D differs from A in
exactly one *trained* feature in either ablation. A fifth pipeline E (log-`E` only)
is registered but redundant with D.

The transformation is part of model identity and provenance. A preprocessor fitted on one campaign must not be reused with a different dataset fingerprint.

Training NLL must only rank hyperparameters within the same pipeline. The A/B/C winner is selected using generated-validation metrics in common physical coordinates. See [[Why Preprocessing A]].
