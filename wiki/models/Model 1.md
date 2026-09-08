---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/README.md
  - FS/ARCHITECTURE.md
---

# Model 1

## Purpose

Generate a joint nine-dimensional distribution of kinematics and event weight.

## Contract

- Features: `x, y, z, E, pz, px, py, t, w`.
- `w` is both a learned and generated feature.
- NF and CFM backends were explored with A/B/C preprocessing.

## Known limitation

The FLUKA weight distribution contains discrete or near-discrete spikes, while a continuous NF represents a continuous density. This can smear spikes and complicate downstream event weights.

## Status

Historical comparison model. Retain it for reproducibility, but the current main direction is [[Model 4]]. See also [[Model 2]] and [[FLUKA Weight]].
