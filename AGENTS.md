# NFs repository instructions

## Scope

- `NFs` is the clean Model 4-centered implementation.
- Do not copy the complete FS repository or import Model 1/2 workflows.
- Preserve validated scientific behavior with explicit FS-equivalence tests.
- Write new orchestration using composition over inheritance.

## Data discipline

- Data and fitted data artifacts belong in `NFs_data`, not this repository.
- Runtime outputs belong in `NFs_output`, not this repository.
- Fit preprocessing and selection/validation guards on train only.
- An all-clean-splits reference guard is permitted only after the model and guard
  method are frozen, must declare `production_only` and
  `selection_allowed: false`, and must never influence validation or test claims.
- Validation selects; test reports only.
- Never change dataset identity, split, preprocessing, or guard provenance silently.

## Configuration and runs

- Human-maintained templates and plans belong in `configs/` and `pipelines/`.
- Resolved job configurations belong with outputs and are immutable after submission.
- Every run has a stable `run_id`, changeable `display_name`, and config hash.
- Every successful stage writes a manifest and `_SUCCESS.json`.
- Treat every meaningful hyperparameter change as a fresh run; never initialize a tuning candidate from another candidate's checkpoint.
- Keep the dataset split, preprocessing artifact, guard, weighting contract, and training seed fixed during an initial one-at-a-time comparison.
- Require an explicit baseline. Multiple simultaneous overrides require an explicit `combined` trial tag.
- Use validation for selection and reserve test for one final report.
- Compare validation NLL only within the same preprocessing coordinate system. Compare A/B/C with generated-validation metrics in physical space.
- Confirm only promising finalists with multiple training seeds; do not run every exploratory candidate with many seeds by default.
- Continuation or checkpoint-resume runs are a distinct experiment type and are not fair substitutes for fresh hyperparameter trials.

## Checkpoint policy

- `last_model.pt` is atomically updated every epoch during an active run.
- `best_model.pt` is atomically updated whenever validation loss improves, regardless of checkpoint interval.
- Numbered `epoch_NNNN_model.pt` checkpoints are full-state and saved every 50 epochs by default.
- Full-state means model, optimizer, scheduler, optional AMP scaler, early-stopping state, best validation loss, RNG, data-loader RNG, and resolved config.
- Completed run directories are immutable. Resumed training must branch to a new absent run directory.

## Model 4 evaluation-plot contract

- Use one shared figure template for generated-validation and final-test plots.
- Put an italic, bold `FlashSim in progress` watermark in the upper-left figure
  margin. It must not overlap the title, legend, axes, or plotted data.
- Every figure header states year/dataset, evaluation split, Model 4,
  preprocessing pipeline, feature ablation, generated event count, and FLUKA
  reference event count.
- Use blue for `FLUKA (simulation)`, orange for `FlashSim (generated)`, and red
  points for `(FlashSim - FLUKA) / FLUKA`.
- Ratio uncertainties are statistical sum-of-squared-weight approximations and
  must be drawn as error bars where the FLUKA denominator is nonzero.
- Use bold plot titles and axis labels. Density axes are `Density [a.u.]`.
- Call q0.001-q0.999 the **bulk**, never the core. The bulk interval is fitted
  with FLUKA train weights. The **tail** page uses the full observed range with
  log-y, and tail evidence also includes CCDF and bulk/tail metric tables.
- Plot physical `E` on the common feature pages and also create a dedicated
  `log10(E/GeV)` bulk-and-tail page. Plotting transforms never modify ROOT data.
- Pearson and Spearman pages compare weighted FLUKA with FlashSim and annotate
  every matrix cell numerically, including the FlashSim-minus-FLUKA matrix.
- Do not create duplicate shape plots for `w1` and `global_c`, because their
  generated kinematics are identical.

## Migration

- Record every port in `MIGRATION_MANIFEST.yaml`.
- Migration states are `planned`, `ported`, `adapted`, `verified`, and `canonical`.
- Do not mark a component canonical until reference behavior is reproduced.

## Work log

- After every meaningful implementation or research update, append a timestamped entry to `wiki/logs/YYYY/MM/YYYY-MM-DD.md` using Asia/Shanghai local time.
- Update `wiki/logs/Daily Work Log.md` when a new daily page is created.
- Record changed files, verification, decisions, and next work.
- Never invent precise timestamps for retrospective entries.
