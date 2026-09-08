---
status: confirmed
last_verified: 2026-09-08
---

# Source and Provenance

This wiki is not the source of truth. Project claims should be traceable to one or more of:

- `FS/`: source code, configurations, tests, README files, and runbooks.
- `FS_output/campaigns/<dataset_id>/prepared_data/`: manifests, fixed splits, fitted preprocessing, and guards.
- `FS_output/campaigns/<dataset_id>/runs/`: training, generation, validation, and model-selection artifacts.
- CERN code root: `/eos/user/t/tanansub/SWAN_projects/FS`.
- CERN output root: `/eos/user/t/tanansub/SWAN_projects/FS_output`.

Dataset fingerprints and split IDs are stronger provenance than filenames. A copied ROOT file without its manifest should not be treated as a canonical dataset.

Pages use these labels:

- **Confirmed**: directly supported by code, config, manifest, or completed output.
- **Interpretation**: a reasoned reading of confirmed evidence.
- **Hypothesis**: an untested research idea.
- **Open question**: evidence or a decision is still missing.

If sources conflict, the conflict must be recorded instead of silently choosing one.

See [[Dataset Registry]], [[Paths and Seeds]], and [[Current Research Status]].
