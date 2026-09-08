---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/model4_generation.py
  - FS/MODEL4_RUNBOOK.md
---

# w1 and global_c

[[Model 4]] generates kinematics once and exports two ROOT files with identical event rows:

- `w1`: every row has `w=1`; use for shape-only equal-weight checks.
- `global_c`: every row has

$$
c=\frac{\sum w_{\mathrm{reference}}}{N_{\mathrm{generated}}}.
$$

The constant makes generated `sum_w` equal the chosen FLUKA reference exposure. It does not make different bins carry different weights and does not reshape the sample; the shape has already been learned through [[Weighted Density]] training.

For MuonDIS delivery, `global_c` is the intended normalized sample unless the downstream study explicitly requests shape-only `w1`.
