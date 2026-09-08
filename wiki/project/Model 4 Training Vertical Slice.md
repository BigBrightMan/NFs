---
status: implemented-local
last_verified: 2026-09-08
---

# Model 4 Training Vertical Slice

The clean NFs implementation can now train one Model 4 weighted-density normalizing flow from validated ROOT inputs through to restart-safe checkpoints.

## Data contract

- Model-space kinematics and raw FLUKA weights are read from separate ROOT files.
- Source metadata `(run, event, id, generation)` must align exactly row by row.
- Dataset ID, split ID, dataset fingerprint, split counts, preprocessing artifact hash, and feature order are validated before training.
- Only train and validation are accepted by the training adapter. Test remains reporting-only.
- FLUKA `w` weights the NLL and is never an NF input or output feature.

For `drop_z_E`, the NF feature order is `(x, y, pz, px, py, t)`. Both `z` and `E` are excluded from the NF and reconstructed by the downstream contract.

## Model and training

The RQ-spline graph is verified against the frozen FS backend. The training command uses AdamW, ReduceLROnPlateau, weighted train and validation NLL, gradient clipping, and optional CUDA mixed precision.

The default policy is 500 epochs with early stopping disabled. `last_model.pt` is refreshed every epoch, `best_model.pt` is refreshed on every validation improvement, and a full resumable checkpoint is written every 50 epochs. The best validation loss and its exact epoch are stored in checkpoint state and the training summary.

Every resolved run records train and validation weight summaries, including total weight and effective sample size.

## Resume contract

A branch resume restores a numbered full-state checkpoint into a new run directory and begins at the next epoch. It never overwrites the parent run. See [[Epoch 450 Branched Resume]].

## Verification

- `tests/test_preprocessing_equivalence.py`
- `tests/test_flow_backend_equivalence.py`
- `tests/test_root_adapter.py`
- `tests/test_model4_training_vertical.py`
- `tests/test_training_config_preflight.py`

The vertical test trains a tiny fresh ROOT-backed run, writes a complete epoch-2 checkpoint, and resumes it at epoch 3 in a separate directory. This is local implementation evidence, not evidence that the actual EOS epoch-450 run has been resumed.

Real campaign inputs must first be resolved and pass the read-only gate described in [[Training Resolution and Preflight]].
