---
status: confirmed
last_verified: 2026-09-11
sources:
  - FS/T_DIAGNOSTICS_RUNBOOK.md
  - FS/src/flashsim/model4_evaluation.py
---

# Tail Validation

Tail validation asks whether rare physical regions and generated extremes agree with FLUKA. It includes:

- full-range log-y tail distributions;
- train-weighted q0.001–q0.999 bulk distributions;
- a dedicated `log10(E/GeV)` bulk-and-tail view, applied only during plotting;
- [[CCDF]] curves;
- weighted tail fractions at fixed thresholds;
- quantile ratios and maximum generated values;
- rejection reasons before and after guarding;
- physics checks for energy, direction, mass shell, and scoring plane.

Robust bounds are diagnostics, not definitions of physical support or the
physical tail. Raw FLUKA min/max is recorded as a finite-sample observed
envelope. It is diagnostic-only for validation/test and becomes an operational
rejection policy only for frozen MuonDIS production. Exact physical constraints
come from the documented generation and reconstruction contract. Rejecting
everything outside q0.001–q0.999 would remove real reference data by
construction.
