---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/src/flashsim/nf.py
  - FS/src/flashsim/model.py
---

# Normalizing Flows

A normalizing flow maps a simple latent distribution to a target distribution through invertible transforms. The change-of-variables formula provides an exact density and therefore supports [[Maximum Likelihood]] training.

FlashSim supports affine and [[Rational-Quadratic Spline]] autoregressive transforms through the `nflows`-based implementation. The current [[Model 4]] uses an RQ-spline flow.

Strengths for this project include direct sampling, tractable likelihood, and multivariate dependence learning. Important limitations include extrapolation in tails and the inability of a continuous density to exactly represent a discrete point mass such as a weight spike.
