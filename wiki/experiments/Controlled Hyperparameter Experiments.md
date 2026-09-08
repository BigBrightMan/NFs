---
date: 2026-09-08
status: implemented-foundation
---

# Controlled Hyperparameter Experiments

Model 4 tuning is config-driven and follows the separation rules in [[Validation Strategy]]. Every meaningful change creates a fresh run with an immutable resolved config and a stable config hash.

## Fresh-run rule

A hyperparameter candidate starts from a fresh model initialization. It must not start from another candidate's checkpoint. A checkpoint continuation is recorded separately as described in [[Epoch 450 Branched Resume]] and is not a fair replacement for a fresh tuning trial.

The initial comparison fixes:

- dataset fingerprint and split assignment;
- preprocessing pipeline;
- Model 4 weighted-density objective;
- guard and reconstruction rules;
- training seed 42;
- every parameter except the named override.

Only a trial tagged `combined` may intentionally override more than one leaf parameter.

## Selection levels

Within one preprocessing pipeline, weighted validation NLL selects training behavior and hyperparameters. NLL values from A/B/C are not directly comparable because the transformed coordinate systems and Jacobians differ.

Finalists from [[Preprocessing A]], [[Preprocessing B]], and [[Preprocessing C]] are compared after generated-validation in physical space using C2ST, FGD, covariance distance, weighted KS, normalized weighted Wasserstein, correlations, 2D checks, tail CCDF, rejection statistics, and physics constraints.

The test split is reporting-only and cannot influence tuning.

## Checkpoint behavior

Each active training run writes:

```text
checkpoints/
├── best_model.pt
├── last_model.pt
├── epoch_0050_model.pt
├── epoch_0100_model.pt
├── ...
└── epoch_0500_model.pt
```

- `best_model.pt` updates whenever validation loss improves, including epochs not divisible by 50.
- `last_model.pt` updates every epoch.
- numbered checkpoints are full-state snapshots every 50 epochs.
- completed run directories are never reused or overwritten.

## Plan runs without submitting

From the NFs repository:

```bash
export PYTHONPATH="$PWD/src"

python3 scripts/plan_model4_tuning.py \
  --campaign configs/campaigns/2025.yaml \
  --pipelines A B C \
  --trials base lr_lo lr_hi t06 t10 h32 h96 b01 b03 \
  --training-seeds 42 \
  --data-root /absolute/path/to/NFs_data \
  --output-root /absolute/path/to/NFs_output
```

This only prints a reviewable plan. Supplying `--write generated_configs/plan.json` writes it once and refuses to overwrite an existing plan.

To schedule one additional trial after its frozen baseline is already complete, select the trial and add `--baseline-already-complete`. This explicit flag prevents a baseline from being omitted accidentally.

Promising finalists can later be confirmed with training seeds 17, 42, and 123. Generation seeds 1556, 2556, and 3556 retain their separate validation, final-test, and MuonDIS roles.

Related: [[Model 4]], [[NFs Pipeline Architecture]], and [[Model 4 Selection 2022-2025]].
