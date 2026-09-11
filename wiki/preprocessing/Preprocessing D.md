---
status: proposed
last_verified: 2026-09-11
sources:
  - src/flashsim_nf/preprocessing/pipelines.py
  - wiki/experiments/Extended Model 4 Experiments.md
---

# Preprocessing D

Pipeline D is [[Preprocessing A]] with both energy variables moved from Box–Cox to
natural log:

- `x,y,z`: MinMax, clip, logit — **unchanged from A**
- `px,py`: signed `log1p` — **unchanged from A**
- `E`: `log(E)` — changed
- `pz`: `log(pz)` — changed
- `t`: identity — unchanged from A
- final step: standardize

## Why both energy variables, and why this is still a single-variable experiment

Model 4 never trains on `E` and `pz` together. Under `drop_ze` the flow learns
`(x,y,pz,px,py,t)` and `E` is reconstructed from the mass shell; under `drop_z_pz`
it learns `(x,y,E,px,py,t)` and `pz` is reconstructed. So in either ablation exactly
one of the changed pair is a trained feature, and D differs from A in exactly one
trained transform:

| ablation | trained features | D-vs-A difference |
|---|---|---|
| `drop_ze` | `x,y,pz,px,py,t` | `pz` only |
| `drop_z_pz` | `x,y,E,px,py,t` | `E` only |

One fitted pipeline therefore serves both experiments. This is enforced by
`test_pipeline_d_is_a_single_variable_change_within_each_ablation`.

## Motivation

Pipeline A fits Box–Cox lambda by maximum-normality MLE on the whole marginal. That
objective optimizes bulk shape and is blind to the hard `E > 10 GeV` production cut
inherited from FLUKA. Three consequences were measured on the frozen A/B/C runs:

- The fitted lambda is positive in all four campaigns (0.505 to 0.709), which
  allocates most of the standardized range to high energies and squeezes the region
  next to the cut.
- The lowest one percent of weighted mass is compressed into 0.0035–0.0058 sigma
  under A, versus 0.037–0.068 sigma under log — 7.5x to 17x narrower. A smooth
  spline flow cannot represent a cliff that sharp without spilling across it.
- `inv_boxcox` is undefined below `-1/lambda`, which sits only 0.075–0.203 sigma
  beneath the physical edge. No production run has hit it, but nothing prevents it.
  `exp` has no domain limit and returns a positive number by construction.

`log` also matches CMS FlashSim practice, where momentum-like positive heavy-tailed
variables use `Log1p` plus standardization and Box–Cox does not appear at all.

## Relationship to Pipeline E

Pipeline E was an earlier log-`E`-only variant. It is redundant with D: under
`drop_z_pz`, the only ablation that trains on `E`, D and E produce identical
model-space data because `pz` is dropped. E remains registered so historical
references resolve; prefer D for new experiments. See
`test_pipeline_d_matches_pipeline_e_on_the_drop_z_pz_feature_set`.

## Status

Proposed, not yet run. D has no fitted artifacts and no generated-validation
results. It is not a candidate for selection until generated-validation completes.
See [[Extended Model 4 Experiments]] for the run procedure and
[[Preprocessing Pipelines]] for the full comparison table.
