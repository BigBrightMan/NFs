---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/T_DIAGNOSTICS_RUNBOOK.md
  - FS/src/flashsim/model4_evaluation.py
---

# Tail Validation

Tail validation asks whether rare physical regions and generated extremes agree with FLUKA. It includes:

- full min-max log-y distributions;
- q0.001–q0.999 core distributions;
- [[CCDF]] curves;
- weighted tail fractions at fixed thresholds;
- quantile ratios and maximum generated values;
- rejection reasons before and after guarding;
- physics checks for energy, direction, mass shell, and scoring plane.

Robust bounds are catastrophic-safety controls, not definitions of the physical tail. Rejecting everything outside q0.001–q0.999 would remove real reference data by construction.
