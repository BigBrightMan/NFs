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

The evaluator writes `generated_evaluation.json`, `_SUCCESS.json`, and five
default plot pages. Metrics include weighted FGD, covariance distance,
approximate multivariate Energy Distance, sliced Wasserstein, weighted HGB
C2ST, Pearson/Spearman correlation differences, per-feature weighted KS and
normalized Wasserstein, and two-sided tail diagnostics.

The figures comprise combined core and full-minmax feature pages, Pearson and
Spearman correlation comparisons, and tail CCDFs. `E` is displayed as
`log10(E/GeV)` only at plotting time. FLUKA is labelled `FLUKA (simulation)`,
generated data is `FlashSim (generated)`, density axes use `Density [a.u.]`,
and ratio panels show `(FlashSim - FLUKA) / FLUKA`.

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

## Complete checkpoint-to-evaluation command

For the completed four-year baseline matrix (A/B/C for each year), create the
12 validation configs without editing generated YAML files by hand:

```bash
python scripts/create_model4_validation_configs.py \
  --output-root /eos/user/t/tanansub/SWAN_projects/NFs_output \
  --guard-root /eos/user/t/tanansub/SWAN_projects/NFs_data/guards \
  --expected-count 12
```

The command fails unless it finds exactly one completed production baseline for
each discovered dataset/preprocessing pair. Existing identical configs are
verified; different existing configs are never overwritten. Dry-run and then
submit the exact matrix:

```bash
module load lxbatch/eossubmit
config_root=/eos/user/t/tanansub/SWAN_projects/NFs_output/resolved_configs/model4_generated_validation

python scripts/submit_model4_post_training_matrix.py \
  --config-root "$config_root" \
  --expected-count 12 \
  --dry-run

python scripts/submit_model4_post_training_matrix.py \
  --config-root "$config_root" \
  --expected-count 12
```

Each job performs training diagnostics if needed, generated-validation sampling,
physical reconstruction and guards, metrics, and the five validation plot pages.
It does not read the test split for model selection.

Resolve an immutable generation config from a completed training run:

```bash
python scripts/resolve_model4_generation_config.py \
  --run-directory "$run_directory" \
  --purpose validation \
  --guard-artifact "$guard_train_ref" \
  --output "$generation_config"
```

Submit generation plus generated-validation evaluation on CERN:

```bash
module load lxbatch/eossubmit
python scripts/submit_model4_post_training.py \
  --config "$generation_config" \
  --dry-run
python scripts/submit_model4_post_training.py \
  --config "$generation_config"
```

The worker performs training diagnostics if absent, loads `best_model.pt`,
inverse-transforms with the frozen A/B/C v3 metadata, reconstructs `z` from a
train-fitted scoring plane and `E` from the muon mass shell, applies the
train-reference robust guard, writes identical-kinematics `w1` and `global_c`
ROOT files, then evaluates against validation. No runtime import from the old
FS repository is used.

After every candidate has a validation evaluation, freeze the winner:

```bash
python scripts/select_model4_validation_winner.py \
  --evaluations $candidate_evaluation_jsons \
  --output "$selected_model_json"
```

The selector uses an equal-weight rank sum across validation-only FGD, Energy
Distance, sliced Wasserstein, mean weighted KS, mean normalized Wasserstein,
C2ST, and Pearson/Spearman errors. The frozen artifact is mandatory for final
test and MuonDIS generation.

Resolve the one reporting-only final test directly from that winner and submit
it through the same worker:

```bash
python scripts/resolve_model4_selected_generation_config.py \
  --selection "$selected_model_json" \
  --purpose test \
  --guard-artifact "$guard_all_ref" \
  --output "$final_test_config"

module load lxbatch/eossubmit
python scripts/submit_model4_post_training.py --config "$final_test_config"
```

After the final report is closed, create the 100,000-event MuonDIS delivery
from the same frozen winner:

```bash
python scripts/resolve_model4_selected_generation_config.py \
  --selection "$selected_model_json" \
  --purpose muondis \
  --number-events 100000 \
  --guard-artifact "$guard_all_ref" \
  --output "$muondis_config"

python scripts/submit_model4_post_training.py --config "$muondis_config"
```

Default seeds are 1556 for generated validation, 2556 for final test, and 3556
for MuonDIS. The final test compares with the frozen test split; MuonDIS
`global_c` uses `sum(w)` from train+validation+test divided by 100,000.

Detector x/y bounds remain disabled in the resolved manifest until an
authoritative campaign artifact is supplied; they are never inferred from an
observed sample min/max. Robust, finite-domain, mass-shell, and scoring-plane
contracts are active.
