---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/model4_evaluation.py
---

# Correlation Validation

Correlation validation checks dependence between variables rather than only their one-dimensional marginals.

- Pearson measures linear association.
- Spearman measures monotonic rank association.
- Correlation-difference matrices compare FLUKA and FlashSim pairwise structure.
- Two-dimensional contour or density comparisons reveal nonlinear geometry not summarized by one coefficient.

The primary figures should compare FLUKA and FlashSim visually. Numeric coefficients belong in separate matrices/tables rather than replacing upper-triangle plots with text. See [[Validation Strategy]].
