---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/AGENTS.md
  - FS/MODEL4_RUNBOOK.md
  - FS/MODEL4_TUNING_RUNBOOK.md
---

# Paths and Seeds

## Existing canonical paths

- Local repository: `/Users/bigbirght/Documents/Hermis/FS`.
- Local outputs: `/Users/bigbirght/Documents/Hermis/FS_output`.
- CERN repository: `/eos/user/t/tanansub/SWAN_projects/FS`.
- CERN outputs: `/eos/user/t/tanansub/SWAN_projects/FS_output`.

## Planned clean layout

- `Hermis/NFs`: Model 4-centered code and this wiki.
- `Hermis/NFs_data`: immutable datasets, splits, preprocessing, statistics, and guards.
- `Hermis/NFs_output`: checkpoints, samples, metrics, and plots.

## Seed roles

- Training/split seed 42: established baseline and immutable split assignment.
- Generation seed 1556: generated-validation.
- Generation seed 2556: final/test generation in the established workflow.
- Generation seed 3556: 100,000-event MuonDIS delivery workflow.

Seed purpose is part of provenance. Different roles should not silently reuse or rename seeds.

See [[Source and Provenance]], [[Dataset Registry]], and [[End-to-End Workflow]].
