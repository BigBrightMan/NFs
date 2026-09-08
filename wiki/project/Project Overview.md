---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/README.md
  - FS/ARCHITECTURE.md
---

# Project Overview

FlashSim learns distributions of FLUKA atmospheric-muon events for fast downstream simulation, including MuonDIS. The physical event variables are:

`x, y, z, E, pz, px, py, t`

The FLUKA event weight `w` is statistically important but is not itself a kinematic variable. Different model generations explored different treatments of `w`; the current main direction is [[Model 4]], a [[Weighted Density|weighted-density]] [[Normalizing Flows|normalizing flow]].

The project compares generated samples against FLUKA in physical space. The principal controls are immutable [[Dataset Registry|dataset identities]], fixed train/validation/test assignments, versioned [[Preprocessing Pipelines]], validation-only model selection, and reporting-only final tests.

See [[End-to-End Workflow]], [[Validation Strategy]], and [[Current Research Status]].
