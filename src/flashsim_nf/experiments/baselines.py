"""Create immutable baseline training configs across campaigns and pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import load_yaml
from .training_config import (
    resolve_model4_training_config,
    write_resolved_training_config,
)


@dataclass(frozen=True)
class BaselineConfigRecord:
    dataset_id: str
    year: int
    preprocessing: str
    stage: str
    status: str
    config_path: str
    run_id: str
    run_directory: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataset_configs(
    project: Path, dataset_ids: Sequence[str] | None
) -> list[Path]:
    candidates = sorted((project / "configs/datasets").glob("*.yaml"))
    if dataset_ids is None:
        return candidates
    selected = set(dataset_ids)
    result = [
        config_path
        for config_path in candidates
        if load_yaml(config_path).get("dataset_id") in selected
    ]
    found = {str(load_yaml(config_path)["dataset_id"]) for config_path in result}
    missing = selected - found
    if missing:
        raise ValueError(f"Unknown dataset IDs: {sorted(missing)}")
    return result


def _validate_existing(
    config: dict[str, Any],
    *,
    dataset_id: str,
    pipeline: str,
    stage: str,
    training_seed: int,
) -> None:
    expected = {
        "dataset_id": (config.get("dataset") or {}).get("dataset_id"),
        "preprocessing": (config.get("preprocessing") or {}).get("id"),
        "stage": (config.get("experiment") or {}).get("stage"),
        "trial_id": (config.get("experiment") or {}).get("trial_id"),
        "training_seed": (config.get("training") or {}).get("seed"),
    }
    observed = {
        "dataset_id": dataset_id,
        "preprocessing": pipeline,
        "stage": stage,
        "trial_id": "base",
        "training_seed": training_seed,
    }
    if expected != observed:
        raise ValueError(
            "Existing resolved config does not match the requested baseline: "
            f"{expected} != {observed}"
        )


def create_model4_baseline_configs(
    *,
    project_root: str | Path,
    output_root: str | Path,
    config_output_directory: str | Path,
    environment: str,
    stage: str = "smoke",
    pipelines: Sequence[str] = ("A", "B", "C"),
    dataset_ids: Sequence[str] | None = None,
    training_seed: int = 42,
    skip_existing: bool = False,
) -> list[BaselineConfigRecord]:
    """Resolve one baseline config per dataset/pipeline without loading ROOT data."""

    project = Path(project_root).resolve()
    destination = Path(config_output_directory)
    if not destination.is_absolute():
        raise ValueError("config_output_directory must be absolute")
    if environment not in {"local", "cern"}:
        raise ValueError("environment must be local or cern")
    normalized_pipelines = tuple(str(item).upper() for item in pipelines)
    if not normalized_pipelines or not set(normalized_pipelines) <= {"A", "B", "C"}:
        raise ValueError("pipelines must contain only A, B, and/or C")
    if len(normalized_pipelines) != len(set(normalized_pipelines)):
        raise ValueError("pipelines must not contain duplicates")

    records: list[BaselineConfigRecord] = []
    pending: list[tuple[Path, dict[str, Any], str, int, str]] = []
    for dataset_config in _dataset_configs(project, dataset_ids):
        dataset = load_yaml(dataset_config)
        dataset_id = str(dataset["dataset_id"])
        year = int(dataset["year"])
        prepared = Path(dataset["legacy_prepared_paths"][environment])
        for pipeline in normalized_pipelines:
            config_path = destination / (
                f"{year}_{pipeline}_base_{stage}_seed{training_seed}.yaml"
            )
            if config_path.exists():
                if not skip_existing:
                    raise FileExistsError(
                        f"Resolved config already exists: {config_path}"
                    )
                existing = load_yaml(config_path)
                _validate_existing(
                    existing,
                    dataset_id=dataset_id,
                    pipeline=pipeline,
                    stage=stage,
                    training_seed=training_seed,
                )
                records.append(
                    BaselineConfigRecord(
                        dataset_id=dataset_id,
                        year=year,
                        preprocessing=pipeline,
                        stage=stage,
                        status="existing_verified",
                        config_path=str(config_path.resolve()),
                        run_id=str(existing["resolution"]["run_id"]),
                        run_directory=str(existing["output"]["run_directory"]),
                    )
                )
                continue
            resolved = resolve_model4_training_config(
                project_root=project,
                dataset_config=dataset_config,
                prepared_directory=prepared,
                output_root=output_root,
                preprocessing=pipeline,
                stage=stage,
                trial_id="base",
                training_seed=training_seed,
            )
            pending.append((config_path, resolved, dataset_id, year, pipeline))

    for config_path, resolved, dataset_id, year, pipeline in pending:
        written = write_resolved_training_config(config_path, resolved)
        records.append(
            BaselineConfigRecord(
                dataset_id=dataset_id,
                year=year,
                preprocessing=pipeline,
                stage=stage,
                status="created",
                config_path=str(written),
                run_id=str(resolved["resolution"]["run_id"]),
                run_directory=str(resolved["output"]["run_directory"]),
            )
        )
    return records
