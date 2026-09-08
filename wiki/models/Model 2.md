---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/README.md
  - FS/ARCHITECTURE.md
  - FS/src/flashsim/model2.py
---

# Model 2

## Purpose

Separate kinematic generation from event-weight prediction.

## Architecture

- A shared 8D NF or CFM generates `x, y, z, E, pz, px, py, t`.
- A positive HGB Poisson reweighter learns the conditional mean `E[w|kinematics]` from raw 8D variables.
- An interpretable three-variable map using `E,x,y` was also studied.

## Strength

The generator does not need to reproduce discrete weight spikes directly.

## Limitation

A deterministic conditional mean cannot reproduce irreducible event-by-event weight variation for nearly identical kinematics.

## Status

Implemented historical comparison model. Do not apply its per-event reweighter to [[Model 4]] output because Model 4 already learns the weighted kinematic density.
