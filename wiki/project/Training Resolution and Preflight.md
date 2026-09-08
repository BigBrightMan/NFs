---
status: implemented-local
last_verified: 2026-09-09
---

# Training Resolution and Preflight

Model 4 training begins with two separate commands. Resolution composes maintained scientific choices with real filesystem paths. Preflight then reads and validates the complete train and validation inputs without creating a run directory or changing any file.

The command-line pipeline supports the Python 3.9 environment currently used by the lxplus virtual environment.

## Why there are two commands

`resolve_model4_training_config.py` produces one immutable YAML for exactly one run. It reads row counts from the selected dataset config, so 2022, 2023, 2024, and 2025 may all have different numbers of events. It never hard-codes the 2025 count.

`preflight_model4_training.py` is a fail-closed read-only gate. Training is allowed only when every check passes.

The resolved config contains train and validation paths only. The held-out test count remains part of dataset provenance, but no test ROOT path is placed in the training data section and preflight does not load the test set.

## Checks performed

- dataset ID, year, fingerprint, split ID, and per-split row counts;
- the cleaned `fluka2022_muons_down_tclean_v1` identity for 2022;
- SHA-256 hashes of maintained dataset, preprocessing, model, tuning, fitted-preprocessor, feature-order, and split-manifest files;
- preprocessing A, B, or C fitted on train only;
- canonical prepared 8D order and Model 4 `drop_ze` order `(x, y, pz, px, py, t)`;
- weighted-NLL contract with FLUKA `w` excluded from the NF inputs;
- RQ-spline and training-policy validity;
- a new run destination inside the declared output root;
- stable config hash and run ID;
- complete ROOT scans of train and validation, including row alignment, finite features, positive weights, and weight summaries.

The fitted `preprocessor.joblib` is hashed but not unpickled by preflight. Its JSON metadata supplies the safe, language-neutral contract check.

## Example: local 2025 Pipeline B smoke run

```bash
cd /Users/bigbirght/Documents/Hermis/NFs
export PYTHONPATH="$PWD/src"

python scripts/resolve_model4_training_config.py \
  --dataset-config configs/datasets/fluka2025_muons_horizontal.yaml \
  --prepared-directory /Users/bigbirght/Documents/Hermis/FS_output/prepared_data \
  --output-root /Users/bigbirght/Documents/Hermis/NFs_output \
  --config-output /Users/bigbirght/Documents/Hermis/NFs_output/resolved_configs/2025_B_base_smoke_seed42.yaml \
  --preprocessing B \
  --stage smoke \
  --trial-id base \
  --training-seed 42

python scripts/preflight_model4_training.py \
  --config /Users/bigbirght/Documents/Hermis/NFs_output/resolved_configs/2025_B_base_smoke_seed42.yaml
```

Do not start training unless the JSON output says `status: pass`, `training_allowed: true`, `test_loaded: false`, and the expected row counts are correct.

After a passing smoke preflight:

```bash
python scripts/train_model4.py \
  --config /Users/bigbirght/Documents/Hermis/NFs_output/resolved_configs/2025_B_base_smoke_seed42.yaml
```

Run that direct command only in a suitable local or allocated GPU environment. On CERN, use the Condor submitter below.

## CERN pattern

Assuming the clean repository is `/eos/user/t/tanansub/SWAN_projects/NFs` and outputs are `/eos/user/t/tanansub/SWAN_projects/NFs_output`:

```bash
cd /eos/user/t/tanansub/SWAN_projects/NFs
source /eos/user/t/tanansub/venvBBfs/bin/activate
export PYTHONPATH="$PWD/src"

python scripts/resolve_model4_training_config.py \
  --dataset-config configs/datasets/fluka2023_muons_down.yaml \
  --prepared-directory /eos/user/t/tanansub/SWAN_projects/FS_output/campaigns/fluka2023_muons_down/prepared_data \
  --output-root /eos/user/t/tanansub/SWAN_projects/NFs_output \
  --config-output /eos/user/t/tanansub/SWAN_projects/NFs_output/resolved_configs/2023_B_base_production_seed42.yaml \
  --preprocessing B \
  --stage production \
  --trial-id base \
  --training-seed 42

python scripts/preflight_model4_training.py \
  --config /eos/user/t/tanansub/SWAN_projects/NFs_output/resolved_configs/2023_B_base_production_seed42.yaml
```

Review a CERN submission without sending it:

```bash
module load lxbatch/eossubmit

python scripts/submit_model4_training.py \
  --config /eos/user/t/tanansub/SWAN_projects/NFs_output/resolved_configs/2025_B_base_smoke_seed42.yaml \
  --dry-run
```

The dry run repeats the complete read-only preflight and prints the exact `condor_submit` command. It creates neither a log directory nor a run directory.

Submit exactly one GPU job:

```bash
module load lxbatch/eossubmit

python scripts/submit_model4_training.py \
  --config /eos/user/t/tanansub/SWAN_projects/NFs_output/resolved_configs/2025_B_base_smoke_seed42.yaml
```

The default request is one GPU, four CPUs, 12 GB of memory, 10 GB of disk, and at most seven days. Resource flags can be overridden explicitly. The worker activates `venvBBfs`, checks required dependencies and the GPU, repeats preflight against the worker-visible EOS files, then starts `train_model4.py`.

Logs for this example are written below:

```text
/eos/user/t/tanansub/SWAN_projects/NFs_output/condor_logs/
  campaigns/fluka2025_muons_horizontal/model4/preprocessing_B/smoke/
```

Use `condor_q` to see queued/running jobs. A dry run does not add a job to `condor_q`.

The production default is 500 epochs, early stopping disabled, and a numbered full-state checkpoint every 50 epochs. `best_model.pt` is still updated whenever validation NLL improves.

## Safety and reruns

Resolution refuses to overwrite an existing resolved YAML. Both resolution and preflight refuse an already-existing run directory. To change preprocessing, trial, seed, epoch count, batch size, or checkpoint interval, create a new resolved config with a new filename; the scientific config hash and run path will change accordingly.

Preflight is intentionally run again on lxplus even if it passed on the Mac, because the real CERN paths and files are the inputs that will be trained.
