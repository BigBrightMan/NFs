---
status: implemented-local
last_verified: 2026-09-11
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

The guard has three distinct layers:

1. The explicit physical contract rejects non-finite output, `E <= 10 GeV`,
   `pz <= 0`, mass-shell violations, and scoring-plane violations. These rules
   come from the model and campaign contract rather than sample extrema.
2. Weighted robust q0.0001/q0.9999 and 3×IQR bounds diagnose catastrophic
   tails. They never reject.
3. Raw FLUKA min/max forms a finite-sample **observed envelope**, not a claimed
   physical limit. The train envelope is diagnostic-only during validation and
   final-test reporting. The all-clean envelope is allowed to reject only as a
   frozen operational production policy for MuonDIS.

Observed-envelope extrema are unweighted because they describe which rows were
observed; FLUKA `w` estimates the target-density robust diagnostics. Campaigns
are never pooled, so every year has separate envelopes.

The cleaned `fluka2022_muons_down_tclean_v1` identity is mandatory for 2022. Campaigns are fitted separately and never pooled across years.

## Command

```bash
PYTHONPATH=src python3 scripts/fit_compare_reference_guards.py \
  --dataset-config configs/datasets/fluka2022_muons_down_tclean_v1.yaml \
  --prepared-directory /absolute/path/to/prepared_data \
  --output-directory /absolute/new/path/in/NFs_data/guards/fluka2022_muons_down_tclean_v1/train_vs_all_clean_v4
```

Add `--generated-root /path/to/pre_rejection_proposals.root` to compare both guards on exactly the same generated proposals.

The output directory must not already exist. Outputs contain both immutable artifacts, a JSON comparison, a compact bounds CSV, a manifest, and `_SUCCESS.json`.

## Local reference-only result

The downloaded Mac data were available for 2022–2024. Reference rejection remained below 0.1% for both guards. The largest relative physical-bound changes were approximately 1.4% in 2022, 1.3% in 2023, and 2.9% in 2024. In 2024 the all-reference guard rejected slightly more weight, showing that more fitting rows do not guarantee wider weighted bounds.

No 2025 artifact was produced locally because the downloaded path did not contain all three raw split ROOT files. No generated-proposal comparison was claimed because a verified pre-rejection proposal ROOT was not supplied.

See [[Why Rejection Sampling]], [[Validation Strategy]], and [[Dataset Registry]].
