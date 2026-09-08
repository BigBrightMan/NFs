"""Controlled, config-driven Model 4 hyperparameter experiment plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..artifacts import ArtifactRoots, ArtifactStore, RunKey
from ..config import config_hash, deep_merge, load_yaml
from ..naming import build_run_id, validate_slug


@dataclass(frozen=True)
class TuningRun:
    run_id: str
    display_name: str
    dataset_id: str
    year: int
    preprocessing: str
    trial_id: str
    training_seed: int
    config_hash: str
    run_directory: str
    changed_parameters: tuple[str, ...]
    resolved_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _leaf_paths(value: Mapping[str, Any], prefix: str = "") -> tuple[str, ...]:
    result: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.extend(_leaf_paths(item, path))
        else:
            result.append(path)
    return tuple(sorted(result))


def _resolve_reference(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def create_tuning_plan(
    *,
    project_root: str | Path,
    campaign_config: str | Path,
    tuning_space: str | Path,
    output_root: str | Path,
    data_root: str | Path,
    pipelines: tuple[str, ...] = ("A", "B", "C"),
    trial_ids: tuple[str, ...] | None = None,
    training_seeds: tuple[int, ...] = (42,),
    require_baseline: bool = True,
) -> list[TuningRun]:
    """Resolve immutable tuning runs without creating their output directories."""

    root = Path(project_root).resolve()
    campaign = load_yaml(campaign_config)
    space = load_yaml(tuning_space)
    dataset_path = _resolve_reference(root, campaign["dataset"])
    dataset = load_yaml(dataset_path)
    baseline = load_yaml(_resolve_reference(root, space["base"]))
    guard = load_yaml(_resolve_reference(root, campaign["guard"]))
    candidates = [trial for trial in space["trials"] if trial.get("enabled", True)]
    selected = set(trial_ids) if trial_ids is not None else None
    if selected is not None:
        candidates = [trial for trial in candidates if trial["trial_id"] in selected]
        missing = selected - {trial["trial_id"] for trial in candidates}
        if missing:
            raise ValueError(f"Unknown or disabled trial IDs: {sorted(missing)}")
    if require_baseline and not any(
        "baseline" in trial.get("tags", []) for trial in candidates
    ):
        raise ValueError("Every tuning plan must contain an explicit baseline")

    allowed_pipelines = tuple(str(item).upper() for item in pipelines)
    if not set(allowed_pipelines) <= set(campaign["preprocessing_candidates"]):
        raise ValueError("Requested preprocessing is not enabled by the campaign")
    if not training_seeds or any(seed < 0 for seed in training_seeds):
        raise ValueError("training_seeds must contain non-negative integers")

    store = ArtifactStore(ArtifactRoots(data=Path(data_root), output=Path(output_root)))
    plans: list[TuningRun] = []
    for pipeline in allowed_pipelines:
        preprocessing = load_yaml(root / f"configs/preprocessing/{pipeline}.yaml")
        if preprocessing.get("fit_split") != "train":
            raise ValueError("Preprocessing must be fitted on train only")
        for trial in candidates:
            trial_id = validate_slug(str(trial["trial_id"]), field="trial_id")
            overrides = trial.get("overrides", {})
            changed = _leaf_paths(overrides)
            if len(changed) > 1 and "combined" not in trial.get("tags", []):
                raise ValueError(
                    f"Trial {trial_id!r} changes multiple parameters without a "
                    "'combined' tag: {changed}"
                )
            for seed in training_seeds:
                resolved = deep_merge(baseline, overrides)
                resolved["experiment"] = {
                    "type": "fresh_hyperparameter_trial",
                    "trial_id": trial_id,
                    "display_name": str(trial["display_name"]),
                    "changed_parameters": list(changed),
                    "training_seed": seed,
                    "data_roles": {
                        "fit": "train",
                        "selection": "validation",
                        "test": "reporting_only",
                    },
                }
                resolved["dataset"] = dataset
                resolved["preprocessing"] = preprocessing
                resolved["guard"] = guard
                identity = config_hash(resolved)
                key = RunKey(
                    dataset_id=str(dataset["dataset_id"]),
                    preprocessing=pipeline,
                    ablation=str(resolved["model"]["ablation"]),
                    trial_id=trial_id,
                    training_seed=seed,
                    config_hash=identity,
                )
                plans.append(
                    TuningRun(
                        run_id=build_run_id(
                            year=int(dataset["year"]),
                            model="m4",
                            preprocessing=pipeline,
                            trial_id=trial_id,
                            training_seed=seed,
                            config_hash=identity,
                        ),
                        display_name=str(trial["display_name"]),
                        dataset_id=str(dataset["dataset_id"]),
                        year=int(dataset["year"]),
                        preprocessing=pipeline,
                        trial_id=trial_id,
                        training_seed=seed,
                        config_hash=identity,
                        run_directory=str(store.tuning_run(key)),
                        changed_parameters=changed,
                        resolved_config=resolved,
                    )
                )
    return plans
