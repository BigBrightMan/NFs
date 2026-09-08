---
status: confirmed
last_verified: 2026-09-08
sources:
  - FS/notes/FLUKA2022_TCLEAN_V1_RUNBOOK.md
---

# Why 2022 t-clean

## Observation

The original 2022 source contains one catastrophic time value:

- `run=960`
- `event=50823`
- `id=13`
- `generation=4`
- `t=1396479058135.057`

Seven other rows outside the diagnostic 3-IQR fence were considered physical tail and retained.

## Decision

Create the isolated dataset `fluka2022_muons_down_tclean_v1`, exclude exactly that identified row before splitting, and leave the original campaign unchanged.

## Reason

A single nonphysical value can dominate scaling and destabilize tail learning. The fail-closed explicit identity prevents broad, silent tail trimming.

## Result

The cleaned dataset contains 1,544,491 selected rows. See [[Dataset Registry]].
