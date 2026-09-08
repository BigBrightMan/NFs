---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS_output/campaigns/*/prepared_data/dataset_manifest.json
  - FS_output/campaigns/*/prepared_data/split/split_manifest.json
  - FS_output/prepared_data/dataset_manifest.json
  - FS_output/prepared_data/split/split_manifest.json
---

# Dataset Registry

All campaigns use an immutable row-wise, weight-stratified 70/15/15 split with seed 42.

| Year | Dataset ID | Selected rows | Train | Validation | Test |
|---:|---|---:|---:|---:|---:|
| 2022 | `fluka2022_muons_down_tclean_v1` | 1,544,491 | 1,081,144 | 231,673 | 231,674 |
| 2023 | `fluka2023_muons_down` | 1,213,877 | 849,713 | 182,082 | 182,082 |
| 2024 | `fluka2024_muons_up` | 770,065 | 539,045 | 115,510 | 115,510 |
| 2025 | `fluka2025_muons_horizontal` | 507,882 | 355,516 | 76,183 | 76,183 |

The 2022 dataset is the isolated cleaned campaign described in [[Why 2022 t-clean]]. The old uncleaned 2022 outputs are not interchangeable with it.

The manifest, dataset fingerprint, split ID, assignment hash, row counts, and preprocessing references must travel together. See [[Source and Provenance]].
