---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/model4_evaluation.py
  - FS/src/flashsim/t_diagnostics.py
---

# CCDF

The complementary cumulative distribution function is:

$$
\mathrm{CCDF}(t)=P(X>t)=1-F(t).
$$

It answers how much probability or weighted exposure remains beyond a threshold. In FlashSim it is used to compare rare tails that are visually suppressed in ordinary histograms.

CCDF should be inspected in physical units with weighted FLUKA and generated curves, alongside event counts and guard rejection reasons. A matching core histogram does not imply a matching tail. See [[Tail Validation]].
