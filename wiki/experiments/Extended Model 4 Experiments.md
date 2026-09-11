---
status: implemented
last_verified: 2026-09-11
---

# Extended Model 4 Experiments

This workflow adds three controlled experiments and one reference baseline. It
does not alter existing A/B/C runs or frozen splits.

## Scientific scope

1. **FLUKA noise floor, all years.** Split weighted FLUKA validation data into
   deterministic independent halves and repeat weighted bootstrap comparisons.
   The resulting q95 metric values show how much discrepancy is expected from
   finite reference statistics alone. Test data are never loaded.
2. **Pipeline D, all years.** Pipeline D is Pipeline A with both energy
   variables changed from Box-Cox to natural log: `log(E)` and `log(pz)`.
   Model 4 trains on `pz` under `drop_ze` and on `E` under `drop_z_pz`, never on
   both, so each ablation still compares D with A across exactly one trained
   feature. One fitted pipeline therefore serves both experiments. It is fitted
   on train and transforms train and validation only. Native artifacts are
   written to `NFs_data`.
3. **`drop_z_pz`, 2022 only.** Run Pipeline A and Pipeline D. The NF learns
   `(x,y,E,px,py,t)`; `z` is reconstructed from the train-fitted scoring plane and
   positive `pz` from the mass shell. With `pz` dropped, the only trained feature
   where D differs from A is `E` (Box-Cox versus `log`), so this is an isolated
   transform comparison. 2022 is chosen because it has the largest train split
   (1.08M rows) and the largest reference ESS (33,915), which gives the best chance
   of separating the effect from finite-sample noise. These are fresh runs, never
   checkpoint continuations. Pipeline E, the earlier log-E-only pipeline, is
   redundant with D here and produces identical model-space data.
4. **Full 8D, 2025 only.** Pipeline A smoke diagnostic. This tests whether a
   full-dimensional flow is numerically stable near the constrained mass-shell
   and scoring-plane manifold. It is not promoted automatically.

## CERN setup

```bash
cd /eos/user/t/tanansub/SWAN_projects/NFs
source /eos/user/t/tanansub/venvBBfs/bin/activate
export PYTHONPATH=/eos/user/t/tanansub/SWAN_projects/NFs/src

data_root=/eos/user/t/tanansub/SWAN_projects/NFs_data
output_root=/eos/user/t/tanansub/SWAN_projects/NFs_output
config_root="$output_root/resolved_configs/model4_extended_smoke_v3_logE_logpz"
```

## 1. Establish the four-year reference noise floor

Start with two repeats as a functional check:

```bash
python3 -u scripts/run_model4_noise_floor_all_years.py \
  --environment cern \
  --output-root "$output_root/noise_floor_smoke" \
  --repeats 2
```

Then run the fixed 20-repeat result in a new directory:

```bash
python3 -u scripts/run_model4_noise_floor_all_years.py \
  --environment cern \
  --output-root "$output_root" \
  --repeats 20 \
  --skip-existing
```

Each campaign writes
`campaigns/<dataset>/model4/reference_noise_floor/validation/metrics.json`.
The bootstrap sample size defaults to that year's complete validation row
count, so the floor is row-matched to standard generated-validation.

## 2. Fit and materialize Pipeline D

```bash
python3 -u scripts/prepare_model4_extended_preprocessing.py \
  --environment cern \
  --data-root "$data_root" \
  --pipelines D
```

The result is `preprocessing_D/` below each campaign in `NFs_data`, containing a
native composed `preprocessor.json`, train/validation model-space ROOT files, a
manifest, and a success marker. No test ROOT file is read or written. Pass
`--pipelines D E` only if the redundant log-E-only Pipeline E is needed for a
historical comparison.

## 3. Resolve all smoke configs

```bash
python3 -u scripts/create_model4_extended_experiment_configs.py \
  --environment cern \
  --output-root "$output_root" \
  --data-root "$data_root" \
  --config-output-directory "$config_root" \
  --stage smoke
```

This creates 7 configs: four Pipeline-D `drop_ze`, two 2022 `drop_z_pz` (A and D),
and one 2025 Pipeline-A full-8D diagnostic.

After the smoke gate passes, create fresh production configs for Pipeline D and
`drop_z_pz` (the 8D diagnostic is intentionally excluded):

```bash
production_root="$output_root/resolved_configs/model4_extended_production_v3_logE_logpz"
python3 -u scripts/create_model4_extended_experiment_configs.py \
  --environment cern \
  --output-root "$output_root" \
  --data-root "$data_root" \
  --config-output-directory "$production_root" \
  --stage production
```

## 4. Preflight and submit selected configs

Preflight all configs without writing training outputs:

```bash
find "$config_root" -name '*.yaml' -print0 | while IFS= read -r -d '' config
do
  python3 scripts/preflight_model4_training.py --config "$config" || exit 1
done
```

Preview submissions:

```bash
find "$config_root" -name '*.yaml' -print0 | while IFS= read -r -d '' config
do
  python3 scripts/submit_model4_training.py --config "$config" --dry-run
done
```

Submit only after every preflight passes:

```bash
module load lxbatch/eossubmit
find "$config_root" -name '*.yaml' -print0 | while IFS= read -r -d '' config
do
  python3 scripts/submit_model4_training.py --config "$config" || exit 1
done
```

To run only one experiment family, replace `"$config_root"` with one of
`$config_root/pipeline_D`, `$config_root/drop_z_pz`, or
`$config_root/full_8d_2025`.

## 5. Generated-validation after smoke or production

Create configs only for completed runs in the requested family. Examples:

```bash
# Four Pipeline-D runs
python3 scripts/create_model4_validation_configs.py \
  --stage smoke --ablations drop_ze --pipelines D --expected-count 4

# Two isolated Box-Cox-E versus log-E drop-z-pz runs (2022)
python3 scripts/create_model4_validation_configs.py \
  --stage smoke --ablations drop_z_pz --pipelines A D --expected-count 2

# One 2025 full-8D diagnostic
python3 scripts/create_model4_validation_configs.py \
  --stage smoke --ablations none --pipelines A --expected-count 1
```

Execute each resolved generation config with
`scripts/execute_model4_post_training.py --config <config>`. The same command
runs training diagnostics, guarded generation, and generated-vs-validation
evaluation. Full 8D is allowed to fail physical rejection; that is diagnostic
evidence for a singular-manifold problem, not a reason to loosen the contract.

## Decision rules

- First require finite training, no failed batches, and a finite best
  validation weighted NLL.
- Compare Pipeline D with A/B/C in common physical-space generated-validation
  metrics, never by cross-preprocessing NLL alone.
- Compare Pipeline A versus D within `drop_z_pz` to isolate Box-Cox(E) versus
  log(E), then compare the winning ablation with the same-year `drop_ze`
  baseline in physical-space metrics.
- Report each metric relative to the same-year FLUKA noise-floor q95 using:

```bash
python3 scripts/compare_model4_evaluation_to_noise_floor.py \
  --evaluation <generated_evaluation.json> \
  --noise-floor <reference_noise_floor/validation/metrics.json> \
  --output <new_noise_comparison.json>
```

- Freeze no new winner until generated-validation is complete. Never inspect
  held-out test data during these decisions.

Related: [[Validation Strategy]], [[Post-training Evaluation]],
[[Preprocessing Pipelines]], and [[Model 4]].
