# Research Log

Detailed timestamped implementation notes are indexed in [[Daily Work Log]].

## 2026-09-08

- Implemented the first NFs foundation slice: package metadata, repository rules, dataset/campaign references, Model 4 baseline and named tuning trials, guard policy, stage contracts, and execution plans.
- Added deterministic config composition and hashing, safe short run IDs, canonical artifact paths, atomic run manifests, and a generated central run index.
- Ported the Model 4 weighted-NLL objective and sample-weight summary from FS commit `9df6e2d`; direct FS-equivalence tests pass.
- Added `MIGRATION_MANIFEST.yaml` with source commits, source checksums, destination modules, migration status, and hashes of the four frozen selected-model files.
- Added the timestamped [[2026-09-08]] daily implementation record and established the daily logging convention.
- Added [[NFs Pipeline Architecture]], the canonical design for the new Model 4-centered workspace.
- Documented the FS-to-NFs migration map, stage contracts, data/output layout, configuration layering, extension points, and definition of done.
- Chose composition over inheritance for preprocessing, generation, evaluation, execution, guards, and dataset-specific selection.
- Documented the hyperparameter workflow: immutable baseline, trial overrides, within-pipeline validation-NLL ranking, physical-space generated-validation, targeted combination search, and finalist multi-seed confirmation.
- Created the initial Obsidian-compatible NFs research wiki.
- Added the central project, model, preprocessing, dataset, validation, experiment, decision, and infrastructure pages.
- Recorded the validation-selected preprocessing result for 2022–2025.
- Recorded the distinction between confirmed implementation, interpretation, proposed Model 3 work, and open questions.
- Recorded source conflicts: the historical Model 4 runbook freezes Pipeline B, while completed generated-validation selects Pipeline A; Model 3 is described in research discussions but is not a validated production model.

See [[index|NFs Research Wiki]] and [[Current Research Status]].

## 2026-09-09

- Completed the standalone Model 4 post-training pipeline in NFs: inverse A/B/C,
  physics reconstruction, guarded ROOT generation, `w1`/`global_c`, weighted
  generated-validation metrics and plots, winner freezing, and final-stage
  selection gates.
- Kept authoritative detector x/y bounds as an explicit unresolved input rather
  than inferring a detector definition from FLUKA sample extrema.
- Added exact-count four-year validation config creation and matrix submission,
  then verified the complete repository with 62 passing tests.
- Locked guard scope by stage: train-only for validation/test reporting and
  all-clean-FLUKA for frozen MuonDIS production, with a four-year guard builder.
- Corrected weighted Spearman to use weighted empirical-CDF mid-ranks and
  versioned the generated-evaluation metric contract as version 2.
