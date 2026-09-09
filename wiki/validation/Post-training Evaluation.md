---
status: implemented
last_verified: 2026-09-09
---

# Post-training Evaluation

Post-training work is deliberately split into two stages so that optimization
diagnostics cannot be confused with generated-sample quality.

## 1. Training validation diagnostics

Run this once a training directory contains `training_summary.json`,
`training/training_history.csv`, `_SUCCESS.json`, and both best and last
checkpoints:

```bash
python scripts/evaluate_model4_training.py \
  --run-directory "$run_directory"
```

It creates a new immutable `training_validation/` directory containing:

- `training_diagnostics.json`;
- `training_diagnostics.png` with train/validation weighted NLL, generalization
  gap, learning rate, gradient norm, batch-weight jitter, and non-finite batches;
- `_SUCCESS.json`.

This stage verifies checkpoint presence and hashes but never loads the held-out
test set. It helps diagnose convergence and overfitting. NLL may select a trial
only against other trials using the same preprocessing coordinates.

## 2. Generated-validation evaluation

After the best checkpoint has produced an inverse-transformed physical 8D ROOT
sample, compare it with the FLUKA validation split:

```bash
python scripts/evaluate_model4_generated.py \
  --dataset-id "$dataset_id" \
  --train-reference-root "$prepared/split/train_rawfeature.root" \
  --reference-root "$prepared/split/validation_rawfeature.root" \
  --generated-root "$generated_root" \
  --output-directory "$evaluation_directory" \
  --reference-split validation \
  --generated-weight-mode auto \
  --seed 1556
```

The FLUKA train and reference ROOT files must contain `w`. The generated sample
may use uniform weights or a constant `global_c`; normalized shape metrics are
identical for those two exports.

The evaluator writes `generated_evaluation.json` and `_SUCCESS.json`. Metrics
include weighted FGD, covariance distance, approximate multivariate Energy
Distance, sliced Wasserstein, per-feature weighted KS and normalized
Wasserstein, and two-sided tail diagnostics.

Tail thresholds and normalization scales are fitted from the train split. For
each feature and threshold, the report records reference/generated tail mass,
mass ratio, effective sample size, conditional tail mean, conditional
Wasserstein, quantile error, and upper-tail CCDF mismatch. A tail result with ESS
below the configured minimum is marked `insufficient_statistics`; it is not
silently treated as a model failure.

## Final test

Only after preprocessing, architecture, checkpoint, and guard choices are
frozen may the same generated evaluator be run with the test split and
`--reference-split test`. Test results are reporting-only and cannot trigger
another tuning round.

The clean repository does not yet contain checkpoint-to-physical-ROOT
generation. Until that component is ported, stage 1 can run immediately after
training, while stage 2 requires a compatible physical generated ROOT from the
existing pipeline.
