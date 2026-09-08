---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/c2st.py
  - FS/src/flashsim/model4_evaluation.py
---

# C2ST

The classifier two-sample test trains a classifier to separate FLUKA from FlashSim using several features simultaneously. The project reports separation AUC:

$$
0.5 + |\mathrm{AUC}-0.5|.
$$

Therefore 0.5 is ideal and larger values mean easier separation. C2ST detects multivariate discrepancies that can be invisible in marginal histograms.

It must use held-out classifier splits and consistent weighting. It is one selection metric, not a replacement for physics constraints, correlations, or [[Tail Validation]].
