---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/PREPROCESSING_PIPELINES.md
---

# Preprocessing B

Pipeline B is the simple baseline:

- `x,y,z,t`: identity followed by standardization.
- `px,py`: IQR-scaled `asinh`.
- `E,pz`: natural log.

It is easy to debug, avoids train-min/max clipping, and provides a useful reference for judging whether more complex transforms help. It can extrapolate outside spatial support and therefore relies more heavily on [[Why Rejection Sampling|generation guards]].

The first 2025 Model 4 baseline was frozen with B, but subsequent completed generated-validation selected [[Preprocessing A]]. B remains a benchmark rather than the current winner.
