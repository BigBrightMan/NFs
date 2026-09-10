---
status: implemented-local
last_verified: 2026-09-08
---

# Train versus All-FLUKA Guard

Two weighted robust-reference guards are intentionally kept separate:

- `guard_train_ref.json` is fitted on train only and may be used for generated-validation and selection.
- `guard_all_ref.json` is fitted on clean train, validation, and test. It is
  marked `production_only` and `selection_allowed: false`, and is activated only
  after the method and model are frozen.

The all-reference artifact cannot be used to improve a validation or test
claim. Generated-validation and final-test reporting both use the train guard.
The all-reference artifact stabilizes the final production support used for
MuonDIS.

## Bound definition

The guard has two distinct layers. Every physical feature is mapped to its
configured stable space for catastrophic-tail diagnostics; those robust bounds
are the wider side of the weighted 3×IQR fence and weighted
q0.0001/q0.9999 limits. They are reported but do not reject events. Separately,
all eight physical features (`x`, `y`, reconstructed `z`, `px`, `py`, `pz`,
reconstructed `E`, and `t`) must remain inside the raw observed FLUKA minimum
and maximum for that dataset and reference scope. These hard-support extrema
are unweighted because they describe support, not density. FLUKA `w` is used
only to estimate the target-density robust diagnostics; it is not guarded as a
generated feature.

The train artifact gets its physical hard support from train only. The all-clean
artifact gets a separate per-year support from train+validation+test and remains
production-only. Campaigns are never pooled, so 2022, 2023, 2024, and 2025 each
have their own bounds.

The cleaned `fluka2022_muons_down_tclean_v1` identity is mandatory for 2022. Campaigns are fitted separately and never pooled across years.

## Command

```bash
PYTHONPATH=src python3 scripts/fit_compare_reference_guards.py \
  --dataset-config configs/datasets/fluka2022_muons_down_tclean_v1.yaml \
  --prepared-directory /absolute/path/to/prepared_data \
  --output-directory /absolute/new/path/in/NFs_data/guards/fluka2022_muons_down_tclean_v1/train_vs_all_clean_v3
```

Add `--generated-root /path/to/pre_rejection_proposals.root` to compare both guards on exactly the same generated proposals.

The output directory must not already exist. Outputs contain both immutable artifacts, a JSON comparison, a compact bounds CSV, a manifest, and `_SUCCESS.json`.

## Local reference-only result

The downloaded Mac data were available for 2022–2024. Reference rejection remained below 0.1% for both guards. The largest relative physical-bound changes were approximately 1.4% in 2022, 1.3% in 2023, and 2.9% in 2024. In 2024 the all-reference guard rejected slightly more weight, showing that more fitting rows do not guarantee wider weighted bounds.

No 2025 artifact was produced locally because the downloaded path did not contain all three raw split ROOT files. No generated-proposal comparison was claimed because a verified pre-rejection proposal ROOT was not supplied.

See [[Why Rejection Sampling]], [[Validation Strategy]], and [[Dataset Registry]].
