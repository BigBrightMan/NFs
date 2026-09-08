---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/model4.py
  - FS/MODEL4_RUNBOOK.md
---

# Weighted Density

Weighted-density training treats each FLUKA row as carrying statistical importance `w_i`. The target empirical measure is proportional to:

$$
\sum_i w_i\,\delta(x-x_i).
$$

[[Model 4]] learns this density through weighted NLL. `w` is neither an NF input nor an NF output. Generated kinematics therefore represent the FLUKA-weighted physical density rather than the original biased row-count distribution.

For downstream normalization, the generated sample can receive one global constant weight. This does not reshape the generated distribution; see [[w1 and global_c]].

Do not apply the event-dependent [[Model 2]] reweighter to Model 4 output unless deliberately studying a different model, because that would weight the learned density a second time.
