---
status: design-approved
last_verified: 2026-09-08
scope: NFs Model 4 pipeline redesign
sources:
  - FS/ARCHITECTURE.md
  - FS/MODEL4_RUNBOOK.md
  - FS/MODEL4_TUNING_RUNBOOK.md
  - FS/PREPROCESSING_PIPELINES.md
  - FS/src/flashsim/model4.py
  - FS/src/flashsim/model4_generation.py
  - FS/src/flashsim/model4_evaluation.py
---

# NFs Pipeline Architecture

## Purpose

`NFs` is the clean Model 4-centered successor workspace. It must preserve the scientific provenance and reproducibility of the existing `FS` repository without copying its historical workflow complexity into the new design.

The intended top-level separation is:

```text
Hermis/
├── FS/             # Existing Model 1/2/4 code and reproducibility archive
├── FS_output/      # Existing experiment artifacts; do not reorganize in place
├── NFs/            # New code, configuration, tests, documentation, and wiki
├── NFs_data/       # Immutable datasets and data-derived artifacts
└── NFs_output/     # New training, generation, evaluation, and delivery outputs
```

`FS` and `FS_output` remain sources of truth during migration. A component becomes canonical in `NFs` only after it reproduces the relevant existing result.

Related pages: [[Project Overview]], [[Current Research Status]], [[End-to-End Workflow]], and [[Source and Provenance]].

## Design principles

1. Use configuration to distinguish datasets, years, preprocessing, trials, and execution environments.
2. Do not create year-specific pipeline code.
3. Fit data-derived artifacts on train only.
4. Use validation for selection and test for reporting only.
5. Keep source code, data, and outputs in separate roots.
6. Every stage must record inputs, resolved configuration, provenance, outputs, and completion status.
7. Prefer composition over inheritance.
8. Do not overwrite a completed run silently.
9. Generated job configuration is an output artifact, not a hand-edited source configuration.
10. Migration must be incremental and reversible.

## Old system compared with NFs

| Existing FS behavior | NFs change | Benefit |
|---|---|---|
| Many create, submit, generate, and evaluate scripts | One pipeline runner and one submitter composed from stages | Smaller public interface and easier automation |
| Some EOS paths embedded in generated configurations | Dataset registry plus artifact-store configuration | Same pipeline can run locally or at CERN |
| Historical 2025 output layout differs from campaign-local years | One campaign layout for 2022–2025 | Easier cross-year inspection |
| `applications/` does not expose preprocessing in the path | Outputs live below `preprocessing_A`, `B`, or `C` | Model identity is visible from the path |
| Generated YAML files can mix with maintained templates | Templates stay in Git; resolved job snapshots go to outputs | Clear ownership and reproducibility |
| Guard integration evolved across v1 and v2 | Guard is an explicit configured component | Policies can change without rewriting the generator |
| Evaluation logic and plots span multiple entry points | One evaluation suite composed from metric components | Metrics can be added independently |
| Completion is sometimes inferred by counting files | Each stage writes a manifest and `_SUCCESS.json` | Reliable status and restart behavior |
| Dataset exceptions can become one-off scripts | Explicit fail-closed selection rules in dataset config | Special cases remain auditable |
| Local and CERN behavior can diverge | Executor is injected into the same pipeline | No duplicate local/CERN pipeline implementations |

## Repository structure

```text
NFs/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── wiki/
│
├── configs/
│   ├── datasets/
│   │   ├── fluka2022_muons_down_tclean_v1.yaml
│   │   ├── fluka2023_muons_down.yaml
│   │   ├── fluka2024_muons_up.yaml
│   │   └── fluka2025_muons_horizontal.yaml
│   ├── preprocessing/
│   │   ├── A.yaml
│   │   ├── B.yaml
│   │   └── C.yaml
│   ├── models/model4/
│   │   ├── baseline.yaml
│   │   └── tuning_space.yaml
│   ├── guards/
│   │   ├── guard_audit.yaml
│   │   └── guard_reject_v2.yaml
│   └── campaigns/
│       ├── 2022.yaml
│       ├── 2023.yaml
│       ├── 2024.yaml
│       └── 2025.yaml
│
├── pipelines/model4/
│   ├── README.md
│   ├── pipeline.yaml
│   ├── stages/
│   │   ├── 00_preflight.yaml
│   │   ├── 10_prepare_data.yaml
│   │   ├── 20_eda.yaml
│   │   ├── 30_train.yaml
│   │   ├── 40_training_validation.yaml
│   │   ├── 50_generated_validation.yaml
│   │   ├── 60_model_selection.yaml
│   │   ├── 70_final_test.yaml
│   │   └── 80_muondis_delivery.yaml
│   └── plans/
│       ├── baseline_only.yaml
│       ├── tuning_abc.yaml
│       ├── generated_validation.yaml
│       ├── final_test.yaml
│       └── muondis_100k.yaml
│
├── src/nfs/
│   ├── contracts/
│   ├── data/
│   ├── preprocessing/
│   ├── models/model4/
│   ├── reconstruction/
│   ├── guards/
│   ├── generation/
│   ├── exporters/
│   ├── evaluation/
│   ├── pipeline/
│   ├── execution/
│   └── io/
│
├── scripts/
│   ├── run_pipeline.py
│   ├── submit_pipeline.py
│   ├── inspect_campaign.py
│   └── lint_wiki.py
└── tests/
```

The scripts are thin command-line adapters. Scientific behavior belongs in `src/nfs`; orchestration definitions belong in `pipelines`; user-maintained choices belong in `configs`.

## Data structure

```text
NFs_data/
└── campaigns/
    └── <dataset_id>/
        ├── source/
        │   └── source_manifest.json
        ├── manifests/
        ├── splits/
        │   ├── train.root
        │   ├── validation.root
        │   └── test.root
        ├── preprocessing/
        │   ├── A/
        │   ├── B/
        │   └── C/
        ├── guards/
        │   └── guard_reject_v2.json
        └── diagnostics/
```

Rules:

- Dataset versions are immutable.
- A new selection or cleaning rule creates a new dataset ID.
- Manifests, fingerprints, split IDs, row counts, and fitted artifacts are part of the dataset contract.
- ROOT files and data artifacts do not enter Git.
- Avoid symlinks; configs should point to explicit data roots.
- [[Preprocessing Pipelines|Preprocessors]] and guards are fitted from train only.

See [[Dataset Registry]] and [[Why 2022 t-clean]].

## Output structure

```text
NFs_output/
└── campaigns/
    └── <dataset_id>/
        └── model4/
            ├── preprocessing_A/
            │   └── drop_z_E/
            │       ├── tuning/
            │       │   └── <trial>/train_seed_<seed>/
            │       ├── production/
            │       │   └── train_seed_<seed>/
            │       ├── generated_validation/
            │       ├── final_test/
            │       └── muondis_delivery/
            ├── preprocessing_B/
            ├── preprocessing_C/
            └── model_selection/
                └── validation/
```

There is no generic `applications/` layer. Dataset, model, preprocessing, ablation, trial, and seed must be visible in the path or run manifest.

Every completed stage writes:

```text
resolved_config.yaml
stage_manifest.json
resource_usage.json        # when applicable
_SUCCESS.json
```

Partial files use a temporary suffix and are promoted atomically only after validation.

## Pipeline stages

### 00 — Preflight

Inputs:

- dataset and campaign configuration;
- expected branches and feature order;
- manifests and fingerprints;
- required upstream artifacts.

Checks:

- dataset identity and year;
- ROOT readability and row counts;
- dataset fingerprint and split ID;
- preprocessing/EDA/guard compatibility;
- checkpoint objective and feature contract;
- destination collision and existing completion state.

The stage fails before creating scientific outputs when a contract is not satisfied.

### 10 — Prepare data

```text
raw FLUKA
  -> source validation
  -> muon selection
  -> explicit fail-closed exclusions
  -> immutable 70/15/15 split
  -> fit A/B/C on train
  -> transform train/validation/test
```

Outputs belong in `NFs_data`, not `NFs_output`. Test membership is fixed here but test values do not guide model selection.

### 20 — EDA

Produces:

- weighted and unweighted raw distributions;
- weight distribution, `sum_w`, extrema, spikes, and effective sample size;
- train/validation/test comparison;
- A/B/C model-space diagnostics;
- full-range tail and train-weighted q0.001–q0.999 bulk plots;
- Pearson/Spearman and 2D diagnostics;
- [[CCDF]] and tail summaries;
- inverse-transform round-trip checks.

EDA validates a frozen data/preprocessing contract. Changing data or preprocessing invalidates the corresponding EDA hash.

### 30 — Train

The current main model is [[Model 4]]:

- [[Weighted Density|weighted-density]] RQ-spline NF;
- `w` weights the NLL but is not an input or output;
- `drop_z_E` learns `x,y,pz,px,py,t`;
- `E` and `z` are reconstructed deterministically;
- weighted validation NLL is evaluated every epoch;
- best and last checkpoints are maintained;
- numbered full-state checkpoints are saved every 50 epochs by default;
- training runs for at most 500 epochs;
- early stopping is disabled by default and, when enabled explicitly, uses validation only.

Training records loss, validation loss, learning rate, gradient norm, epoch time, non-finite batches, batch-weight statistics, and resume state.

The checkpoint contract is detailed in [[Controlled Hyperparameter Experiments]]: best is updated on every validation improvement, last is updated every epoch, and numbered full-state snapshots are written every 50 epochs.

### 40 — Training validation

Purpose:

- diagnose optimization and overfitting;
- choose the best epoch/checkpoint;
- rank hyperparameters within one preprocessing pipeline.

Outputs include learning curves, learning-rate curves, generalization gap, checkpoint evolution, stability flags, and a within-pipeline ranking table.

NLL values from A/B/C must not be directly ranked against each other because their coordinates and Jacobians differ.

### 50 — Generated validation

Each preprocessing finalist is sampled with validation seed 1556 and compared against the FLUKA validation split in physical coordinates.

Required evidence:

- hard physics/guard pass status;
- rejection fraction and reasons;
- [[C2ST]] separation AUC;
- FGD and covariance distance;
- weighted KS and normalized weighted Wasserstein;
- [[Correlation Validation]];
- [[Tail Validation]] and [[CCDF]];
- weight normalization and effective sample statistics;
- combined bulk/tail figures and optional separate feature plots.

This stage selects preprocessing and architecture behavior, not the final test.

### 60 — Model selection

Selection order:

```text
hard physics and provenance gates
  -> C2ST separation AUC
  -> FGD
  -> covariance distance
  -> maximum weighted KS
  -> normalized weighted Wasserstein
  -> rejection fraction
```

The result is frozen in `selected_model.json`. Current evidence is summarized in [[Model 4 Selection 2022-2025]] and supports [[Preprocessing A]] for all four campaigns.

### 70 — Final test

The frozen winner is generated with test seed 2556 and compared once with the held-out test split. Results are reporting-only.

The final test must not change:

- preprocessing;
- guard policy;
- feature ablation;
- architecture or hyperparameters;
- selected checkpoint;
- threshold definitions.

### 80 — MuonDIS delivery

The delivery plan generates 100,000 accepted events with seed 3556 and the frozen guard policy. One kinematic sample is exported as:

- `w1`, for equal-weight shape checks;
- `global_c`, for downstream normalization.

The intended MuonDIS file is normally `global_c`, with:

$$
c=\frac{\sum w_{\mathrm{FLUKA\ reference}}}{N_{\mathrm{generated}}}.
$$

See [[w1 and global_c]] and [[Why Rejection Sampling]].

## Composition over inheritance

### Rule

Use inheritance to define a true substitutable interface. Use composition to assemble scientific behavior.

Avoid a hierarchy such as:

```text
BasePipeline
  -> Model4Pipeline
      -> Model4Pipeline2022
      -> Model4Pipeline2023
      -> Model4Pipeline2024
      -> Model4Pipeline2025
```

It creates subclasses for every year, guard, preprocessing, and environment combination.

Prefer:

```python
pipeline = Pipeline(
    dataset=dataset_registry.load(dataset_id),
    selector=MuonSelector(...),
    splitter=ImmutableSplitter(...),
    preprocessor=preprocessor_registry.create("A"),
    model=Model4(...),
    objective=WeightedNLL(),
    reconstruction=DropZEReconstruction(...),
    guard=GuardRejectV2(...),
    evaluator=EvaluationSuite([...]),
    artifact_store=ArtifactStore(output_root),
    executor=CondorExecutor(...),
)
```

The pipeline for another year changes configuration, not class hierarchy.

### Preprocessing composition

```python
PipelineA = TransformChain([
    SpatialMinMaxLogit(),
    SignedLog1p(["px", "py"]),
    BoxCox(["E", "pz"]),
    Standardize(),
])
```

```python
PipelineB = TransformChain([
    Identity(["x", "y", "z", "t"]),
    ScaledAsinh(["px", "py"]),
    Log(["E", "pz"]),
    Standardize(),
])
```

```python
PipelineC = TransformChain([
    RobustSpatialAsinh(),
    ZeroCenteredMomentumAsinh(),
    ShiftedEnergyLogAsinh(shift=10.0),
    RobustLogAsinh(["pz"]),
    Standardize(),
])
```

No `PipelineA2025` subclass is needed.

### Generation composition

```python
generation = GenerationPipeline(
    sampler=NFSampler(),
    inverse_transform=InversePreprocessor(),
    reconstruction=DropZEReconstruction(),
    guard=GuardRejectV2(),
    exporters=[
        UnitWeightExporter(),
        GlobalConstantWeightExporter(),
    ],
)
```

The same accepted kinematics are passed to both exporters, ensuring `w1` and `global_c` remain event-identical.

### Evaluation composition

```python
evaluation = EvaluationSuite([
    WeightedKSMetrics(),
    WassersteinMetrics(),
    C2STMetrics(),
    GaussianDistanceMetrics(),
    CorrelationMetrics(),
    TailMetrics(),
    PhysicsConstraintMetrics(),
])
```

A new metric is registered as a component rather than added to one monolithic evaluator.

### Execution composition

```python
executor = LocalExecutor()
```

or:

```python
executor = CondorExecutor(eos_submit=True)
```

Scientific stages remain the same. Do not create separate local and CERN pipeline classes.

### Dataset-specific selection composition

The 2022 exclusion belongs in dataset configuration:

```yaml
selection:
  explicit_exclusions:
    - label: catastrophic_t_single_row_2022
      run: 960
      event: 50823
      id: 13
      generation: 4
      expected_matches: 1
```

The selector enforces `expected_matches: 1`; no special `Prepare2022Pipeline` is required.

### Acceptable inheritance

Inheritance remains appropriate where a framework or substitutable contract requires it, for example:

- PyTorch modules inheriting `torch.nn.Module`;
- small abstract protocols for a metric, transform, executor, or exporter;
- implementations that obey the same input/output contract.

It should not be used merely to reuse workflow code.

## Configuration composition

A resolved run configuration is assembled from layers:

```text
dataset config
  + preprocessing config
  + model baseline
  + trial overrides
  + guard policy
  + execution plan
  + explicit runtime overrides
  = resolved_config.yaml
```

Precedence must be deterministic and recorded. The resolved snapshot is immutable after submission.

Example campaign configuration:

```yaml
dataset: fluka2025_muons_horizontal
year: 2025

preprocessing:
  candidates: [A, B, C]

model:
  family: model4
  ablation: drop_z_E
  architecture: rq_spline
  training_seed: 42
  batch_size: 2048

generation:
  validation_seed: 1556
  final_test_seed: 2556
  muondis_seed: 3556

guard:
  policy: guard_reject_v2

paths:
  data_root: /Users/bigbirght/Documents/Hermis/NFs_data
  output_root: /Users/bigbirght/Documents/Hermis/NFs_output
```

The CERN deployment changes roots and executor settings, not scientific choices.

## Hyperparameter tuning

### Baseline is immutable

```yaml
# configs/models/model4/baseline.yaml
architecture: rq_spline
num_transforms: 8
hidden_features: 64
num_blocks: 2
learning_rate: 0.0005
batch_size: 2048
maximum_epochs: 500
early_stopping:
  enabled: false
  patience: 20
checkpoint_interval: 50
training_seed: 42
```

Do not edit this file repeatedly to create experiments.

### Trial overrides

```yaml
# configs/models/model4/tuning_space.yaml
base: configs/models/model4/baseline.yaml

trials:
  - name: baseline
    overrides: {}

  - name: lr_low
    overrides:
      learning_rate: 0.0002

  - name: lr_high
    overrides:
      learning_rate: 0.001

  - name: transforms_10
    overrides:
      num_transforms: 10

  - name: hidden_96
    overrides:
      hidden_features: 96

  - name: blocks_3
    overrides:
      num_blocks: 3
```

Each trial receives a unique name, resolved config, config hash, run directory, and manifest. Trial outputs never overwrite one another.

### Fair-comparison controls

The first tuning round holds constant:

- dataset fingerprint;
- train/validation split;
- preprocessing pipeline;
- feature ablation;
- training seed;
- evaluation procedure.

Only the declared trial override changes.

### Selection stages

Stage 1 ranks trials within each preprocessing pipeline using stability gates and best weighted validation NLL. Stage 2 generates one finalist per preprocessing pipeline and compares them in physical space using [[Validation Strategy]].

The test split never participates in tuning.

### Recommended search sequence

1. **OAAT:** vary learning rate, transforms, hidden width, and blocks separately.
2. **Small combination search:** combine only promising OAAT values in 4–8 targeted trials.
3. **Multi-seed confirmation:** retrain only the top one or two configurations with seeds such as 17, 42, and 123.
4. **Freeze winner:** write `selected_model.json` before final test.

This is more interpretable than an immediate full grid and much cheaper than multi-seeding every candidate.

### Changes that require retraining

| Change | Retrain? |
|---|---|
| Learning rate | Yes |
| Batch size | Yes |
| Number of transforms | Yes |
| Hidden width | Yes |
| Number of blocks | Yes |
| Preprocessing A/B/C | Yes |
| Feature ablation/reconstruction | Yes |
| Weighted versus unweighted objective | Yes |
| Training seed | Yes |
| Generation seed | No |
| Number of generated events | No |
| Guard v1 to v2 | No |
| `w1` versus `global_c` | No |
| New evaluation metric or plot | No |

## Execution plans

Plans select work without editing generated job YAML.

Example baseline plan:

```yaml
stages:
  - preflight
  - train
  - training_validation

selection:
  preprocessing: [A, B, C]
  trials: [baseline]
```

Example MuonDIS plan:

```yaml
stages:
  - preflight
  - muondis_delivery

number_events: 100000
generation_seed: 3556
guard: guard_reject_v2
weight_exports: [w1, global_c]
```

Target command-line shape:

```bash
python scripts/run_pipeline.py \
  --campaign configs/campaigns/2025.yaml \
  --plan pipelines/model4/plans/generated_validation.yaml
```

For CERN submission:

```bash
python scripts/submit_pipeline.py \
  --campaign configs/campaigns/2025.yaml \
  --plan pipelines/model4/plans/generated_validation.yaml \
  --dry-run
```

The dry run performs preflight and prints jobs; it does not submit them.

## Extension guide

| Goal | Required addition | Core pipeline change? |
|---|---|---|
| Add a new year | Dataset and campaign config | No |
| Add preprocessing D | Transform composition and registry entry | Usually no |
| Add a new Model 4 trial | Tuning-space override | No |
| Add Model 5 | Model and objective components | Pipeline contract only if outputs differ |
| Add a metric | Metric component and suite registration | No |
| Add guard v3 | Guard implementation and policy config | No |
| Add an output format | Exporter component | No |
| Add a local/cluster backend | Executor component | No |
| Add a workflow subset | Plan YAML | No |

New components must declare their input contract, output contract, provenance fields, and tests.

## Migration map

Initial mapping from FS to NFs:

```text
FS/src/flashsim/model4.py
  -> NFs/src/nfs/models/model4/objective.py

FS/src/flashsim/model4_generation.py
  -> NFs/src/nfs/generation/model4.py
  -> NFs/src/nfs/exporters/root.py

FS/src/flashsim/model4_evaluation.py
  -> NFs/src/nfs/evaluation/suite.py
  -> NFs/src/nfs/evaluation/metrics/

FS/src/flashsim/generation_guard.py
  -> NFs/src/nfs/guards/

FS/src/flashsim/preprocessing.py
  -> NFs/src/nfs/preprocessing/

FS/scripts/create_model4_*.py
FS/scripts/submit_model4_*.py
  -> NFs/scripts/run_pipeline.py
  -> NFs/scripts/submit_pipeline.py
```

Migration states:

```text
planned -> copied -> adapted -> verified -> canonical
```

No existing FS component is deleted merely because a new component exists. Verification must compare manifests, row counts, resolved preprocessing, accepted generated events, metrics, and representative plots.

## Migration phases

### Phase 1 — Foundation

- Create package, config loader, contracts, artifact store, and pipeline interfaces.
- Add read-only adapters for existing FS data and checkpoints.
- Do not move EOS outputs.

### Phase 2 — Model 4 vertical slice

- Reproduce one frozen 2025 Pipeline A run from preflight through generated validation.
- Compare the NFs result with the existing selected-model evidence.
- Validate identical `w1`/`global_c` kinematics and constant normalization.

### Phase 3 — Four campaigns

- Register cleaned 2022, 2023, 2024, and 2025.
- Reproduce the selected Pipeline A workflow for each year.
- Run consistent final-test and MuonDIS delivery stages.

### Phase 4 — Legacy boundary

- Keep Model 1/2 and historical campaign scripts in FS for reproducibility.
- Import only shared components still required by Model 4.
- Mark superseded interfaces explicitly; remove nothing until reproducibility is demonstrated.

## Definition of done

The NFs pipeline becomes canonical when:

- all four dataset identities and splits validate against their frozen manifests;
- preprocessing A/B/C round trips and hashes are reproduced;
- Model 4 weighted NLL and checkpoints satisfy the same contracts;
- generated-validation reproduces the declared winners within expected stochastic variation;
- final-test evaluation is reporting-only by construction;
- MuonDIS `global_c` delivery records the correct normalization source;
- every stage is restart-safe and provenance-complete;
- the wiki and operational runbook match the implemented CLI.

## Planned changes

- Implement the directory and package skeleton.
- Define component protocols and resolved-config schema.
- Implement the dataset registry and artifact-store path builder.
- Port the Model 4 objective, preprocessing, reconstruction, guard, generation, and evaluation components.
- Add local and CERN executors.
- Add legacy read-only import adapters.

## Open architectural questions

- Whether `NFs_data` should hold local ROOT copies or only manifests pointing to EOS for large campaigns.
- The authoritative detector `x,y` bounds and units for each campaign.
- Whether final-test and 100k MuonDIS validation should be separate report types sharing one metric suite.
- Whether experimental [[Model 3]] should eventually share the same generation/export/evaluation components.
- Which FS scripts remain necessary solely for Model 1/2 reproducibility.

See [[Open Questions]] for the broader research list.
