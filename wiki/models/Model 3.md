---
status: experimental
last_verified: 2026-09-08
sources:
  - FS/README.md
  - FS/src/flashsim/factorized_model.py
---

# Model 3

Model 3 is the factorized-weight research direction:

$$
p(x_{\mathrm{kin}},w)=p(x_{\mathrm{kin}})p(w\mid x_{\mathrm{kin}}).
$$

## Proposed variants

- **Model 3A — Pure Conditional Weight Flow:** generate kinematics, then sample a continuous conditional distribution `p(w|x_kin)`.
- **Model 3B — Spike Mixture plus Conditional Flow:** model discrete spike membership explicitly and use a conditional flow for the continuous remainder.

## Source conflict

> [!warning] Implementation status
> The repository README states that Model 3 is deliberately not implemented in the validated revision. The current working tree contains an experimental `factorized_model.py`, but there is no completed training, generated-validation, or final-test result establishing Model 3 as operational.

## Status

Hypothesis/experimental scaffold only. It must not be presented as a trained result. Compare with the implemented [[Model 1]], [[Model 2]], and [[Model 4]].
