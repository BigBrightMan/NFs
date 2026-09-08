---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/README.md
  - FS/src/flashsim/model4.py
  - FS/src/flashsim/model4_generation.py
---

# FLUKA Weight

The FLUKA branch `w` is an event importance weight associated with biased simulation sampling. It changes statistical contribution, not the kinematic coordinates themselves.

The project has explored three treatments:

- [[Model 1]] generates `w` jointly with kinematics.
- [[Model 2]] predicts a conditional-mean event weight after generating kinematics.
- [[Model 4]] uses `w` only in the training loss to learn a [[Weighted Density]].

Weight spikes reduce effective sample size and can be difficult for a continuous density to represent exactly. Weight statistics must therefore include `sum_w`, effective sample size, extrema, and spike fractions, not only the row count.
