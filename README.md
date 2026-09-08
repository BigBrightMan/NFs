# FlashSim NFs

Clean, Model 4-centered normalizing-flow pipeline for FLUKA atmospheric-muon generation and validation.

This project is being extracted incrementally from the existing `FS` repository. Validated scientific behavior is ported with equivalence tests; orchestration, configuration, naming, manifests, and output layout are redesigned.

## Current phase

Foundation and controlled migration. The existing `FS` repository and `FS_output` remain canonical until a complete NFs vertical slice reproduces the frozen reference evidence.

Implemented foundations now include numerically verified composition-based preprocessing A/B/C, an FS-equivalent RQ-spline backend, a fail-closed ROOT adapter, end-to-end Model 4 weighted-NLL training, non-destructive full-state checkpoint branching, a real training-config resolver, a read-only full-data preflight, and one-job EOS-backed HTCondor submission. See [Model 4 Training Vertical Slice](wiki/project/Model%204%20Training%20Vertical%20Slice.md), [Training Resolution and Preflight](wiki/project/Training%20Resolution%20and%20Preflight.md), and [Epoch 450 Branched Resume](wiki/experiments/Epoch%20450%20Branched%20Resume.md).

## Layout

```text
configs/      Maintained dataset, preprocessing, model, guard, and campaign choices
pipelines/    Stage definitions and execution plans
src/          Reusable implementation
scripts/      Thin command-line entry points
tests/        Unit and FS-equivalence tests
wiki/         Obsidian-compatible research knowledge base
```

See [NFs Pipeline Architecture](wiki/project/NFs%20Pipeline%20Architecture.md) and [MIGRATION_MANIFEST.yaml](MIGRATION_MANIFEST.yaml).
