---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/PREPROCESSING_PIPELINES.md
---

# Preprocessing C

Pipeline C is the robust alternative:

- `x,y,z`: median/IQR followed by `asinh`.
- `px,py`: zero-centered IQR-scaled `asinh`.
- `E`: `log(E-10 GeV)` followed by robust `asinh`.
- `pz`: log followed by robust `asinh`.
- `t`: identity followed by ordinary standardization.

C is resistant to many outliers and encodes `E>10 GeV`, but can compress tails too strongly and does not robustify `t`. In current Model 4 generated-validation it did not beat [[Preprocessing A]] and exceeded the 1% rejection hard gate for 2023–2025 finalists.
