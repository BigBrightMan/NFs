---
date: 2026-09-08
status: implemented
---

# Epoch 450 Branched Resume

This mechanism continues a Model 4 experiment from a saved epoch-450 checkpoint while treating the original run as read-only. It is part of the training stage in [[NFs Pipeline Architecture]].

## Safety contract

- The source checkpoint must exist and must report `epoch: 450`.
- The destination run directory must not exist. An existing path is a hard failure, even if it appears empty.
- Epoch 451 is the first resumed epoch.
- All new checkpoints, history, diagnostics, logs, and lineage records are written below the new run directory.
- The source checkpoint is hashed before restore. A changed source is a hard failure.
- Model weights are mandatory. Other state is restored when present and listed with its consequence when absent.
- Periodic NFs checkpoints are full-state checkpoints every 50 epochs, not inference-only snapshots.

## Current defaults

```yaml
maximum_epochs: 500
checkpoint_interval: 50
early_stopping:
  enabled: false
  patience: 20
```

Early-stopping state is still loaded when it exists so that an explicitly enabled branch can continue correctly. With the default disabled policy, the branch proceeds through epoch 500 unless training fails.

## Inspect before branching

```bash
PYTHONPATH=src python3 scripts/inspect_training_checkpoint.py \
  /absolute/path/to/checkpoints/epoch_0450_model.pt
```

The report names model, optimizer, scheduler, scaler, early-stopping, best-loss, best-model, and RNG availability.

## Prepare an immutable branch

```bash
PYTHONPATH=src python3 scripts/prepare_resume_branch.py \
  --checkpoint /absolute/path/to/checkpoints/epoch_0450_model.pt \
  --new-run-directory /absolute/path/to/new_resume_run \
  --expected-epoch 450
```

This creates the new output layout and `lineage/resume.json`; it does not start training. The composed `BranchedTrainingSession` restores the state and drives epochs 451–500 through injected Model 4 train/validation callbacks.

## Audit of downloaded legacy FS checkpoints

The local Mac copy contains no epoch-450 checkpoint. The downloaded Model 4 runs inspected on 2026-09-08 end at or before epoch 200. Their `last_model.pt` files contain:

- model state;
- optimizer state;
- ReduceLROnPlateau scheduler state;
- early-stopping counter;
- best validation loss.

They do not contain:

- AMP scaler state — the inspected legacy trainer did not use AMP, so this has no consequence unless the resumed branch enables AMP;
- Python, NumPy, Torch, CUDA, or data-loader RNG state — exact stochastic reproduction of the old trajectory is therefore impossible;
- embedded historical best-model weights — the best scalar loss can be continued, but the corresponding older best weights require the parent's separate `best_model.pt`;
- full optimizer state in numbered periodic checkpoints — the legacy code wrote lightweight periodic files. A legacy epoch-450 periodic file cannot be an exact resume source unless its payload was created by newer full-state code.

Therefore the actual epoch-450 file must be inspected on EOS (or copied locally) before the branch is submitted. The NFs inspector makes these limitations explicit instead of silently resetting state.

Related: [[Model 4]], [[Preprocessing Pipelines]], and [[Validation Strategy]].
