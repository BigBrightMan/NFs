"""Pure construction of one EOS-backed HTCondor Model 4 submission."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class CondorResources:
    gpus: int = 1
    cpus: int = 4
    memory: str = "12GB"
    disk: str = "10GB"
    maximum_runtime_seconds: int = 604800

    def __post_init__(self) -> None:
        if self.gpus <= 0:
            raise ValueError("Model 4 training requires at least one GPU")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")
        if not _SAFE_COMPONENT.fullmatch(self.memory):
            raise ValueError("memory must be a simple HTCondor quantity")
        if not _SAFE_COMPONENT.fullmatch(self.disk):
            raise ValueError("disk must be a simple HTCondor quantity")
        if self.maximum_runtime_seconds <= 0:
            raise ValueError("maximum_runtime_seconds must be positive")


@dataclass(frozen=True)
class CondorSubmission:
    command: tuple[str, ...]
    run_name: str
    log_directory: Path
    run_directory: Path


def _absolute_file(value: str | Path, *, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    if not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path.resolve()


def _absolute_directory(value: str | Path, *, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path.resolve()


def _safe(value: Any, *, name: str) -> str:
    text = str(value)
    if not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(f"Unsafe {name}: {text!r}")
    return text


def build_condor_submission(
    *,
    config_path: str | Path,
    config: Mapping[str, Any],
    project_root: str | Path,
    activate_path: str | Path,
    resources: CondorResources,
) -> CondorSubmission:
    """Return a validated command without creating files or submitting work."""

    resolved_config = _absolute_file(config_path, name="resolved config")
    project = _absolute_directory(project_root, name="project_root")
    activate = _absolute_file(activate_path, name="environment activation script")
    submit_file = _absolute_file(
        project / "condor/submit_model4_training.sub", name="Condor submit file"
    )
    launcher = _absolute_file(
        project / "condor/run_model4_training.sh", name="Condor worker launcher"
    )
    if not launcher.stat().st_mode & 0o111:
        raise PermissionError(f"Condor worker launcher is not executable: {launcher}")

    resolution = config["resolution"]
    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    experiment = config["experiment"]
    output = config["output"]
    output_root = _absolute_directory(resolution["output_root"], name="output_root")
    run_directory = _absolute_directory(
        output["run_directory"], name="run_directory"
    )
    try:
        run_directory.relative_to(output_root)
    except ValueError as error:
        raise ValueError("run_directory is outside output_root") from error
    if run_directory.exists():
        raise FileExistsError(f"Run directory already exists: {run_directory}")

    run_name = _safe(resolution["run_id"], name="run_id")
    dataset_id = _safe(dataset["dataset_id"], name="dataset_id")
    pipeline = _safe(preprocessing["id"], name="preprocessing")
    stage = _safe(experiment["stage"], name="stage")
    log_subdir = f"campaigns/{dataset_id}/model4/preprocessing_{pipeline}/{stage}"
    log_directory = output_root / "condor_logs" / log_subdir

    command = (
        "condor_submit",
        f"project={project}",
        f"output_root={output_root}",
        f"activate={activate}",
        f"config={resolved_config}",
        f"run_name={run_name}",
        f"gpus={resources.gpus}",
        f"cpus={resources.cpus}",
        f"memory={resources.memory}",
        f"disk={resources.disk}",
        f"maximum_runtime={resources.maximum_runtime_seconds}",
        f"log_subdir={log_subdir}",
        str(submit_file),
    )
    return CondorSubmission(
        command=command,
        run_name=run_name,
        log_directory=log_directory,
        run_directory=run_directory,
    )
