"""Full-state checkpoints and non-destructive branched resume."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import write_json_atomic
from .state import EarlyStoppingState, RestoredComponent, ResumeReport

FORMAT_NAME = "flashsim_nf.full_training_state"
FORMAT_VERSION = 1


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Training checkpoints require the optional torch dependency"
        ) from error
    return torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    torch = _torch()
    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["torch_cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng_state(state: dict[str, Any]) -> None:
    torch = _torch()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def full_checkpoint_payload(
    *,
    epoch: int,
    model: Any,
    optimizer: Any,
    scheduler: Any | None,
    scaler: Any | None,
    early_stopping: EarlyStoppingState,
    config: dict[str, Any],
    best_model_state_dict: dict[str, Any] | None = None,
    data_loader_generator: Any | None = None,
) -> dict[str, Any]:
    """Build a checkpoint capable of exact continuation where PyTorch permits."""

    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
        if scheduler is not None
        else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "early_stopping_state": early_stopping.to_dict(),
        "best_validation_loss": early_stopping.best_validation_loss,
        "best_epoch": early_stopping.best_epoch,
        "best_model_state_dict": best_model_state_dict,
        "rng_state": capture_rng_state(),
        "data_loader_generator_state": (
            data_loader_generator.get_state()
            if data_loader_generator is not None
            else None
        ),
        "config": config,
    }


def save_checkpoint_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    torch = _torch()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(destination)


def inspect_checkpoint(path: str | Path) -> dict[str, Any]:
    torch = _torch()
    source = Path(path)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    keys = {
        "model": "model_state_dict" in checkpoint,
        "optimizer": "optimizer_state_dict" in checkpoint,
        "scheduler": checkpoint.get("scheduler_state_dict") is not None,
        "scaler": checkpoint.get("scaler_state_dict") is not None,
        "early_stopping": (
            "early_stopping_state" in checkpoint
            or "epochs_without_improvement" in checkpoint
        ),
        "best_validation_loss": any(
            key in checkpoint
            for key in (
                "best_validation_loss",
                "best_validation_weighted_nll",
                "best_validation_nll",
            )
        ),
        "best_epoch": (
            checkpoint.get("best_epoch") is not None
            or checkpoint.get("early_stopping_state", {}).get("best_epoch")
            is not None
        ),
        "best_model_weights": checkpoint.get("best_model_state_dict") is not None,
        "rng": "rng_state" in checkpoint,
        "data_loader_rng": checkpoint.get("data_loader_generator_state") is not None,
    }
    return {
        "path": str(source.resolve()),
        "sha256": sha256(source),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "format": checkpoint.get("format", "legacy_fs"),
        "format_version": checkpoint.get("format_version"),
        "available_state": keys,
    }


def prepare_resume_directory(
    source_checkpoint: str | Path,
    destination_run: str | Path,
    *,
    expected_epoch: int = 450,
) -> dict[str, Any]:
    """Create a fresh branch directory and immutable lineage before training."""

    source = Path(source_checkpoint).resolve()
    destination = Path(destination_run).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {source}")
    inspection = inspect_checkpoint(source)
    if inspection["epoch"] != expected_epoch:
        raise ValueError(
            f"Expected epoch {expected_epoch}, checkpoint contains epoch "
            f"{inspection['epoch']}"
        )
    if destination.exists():
        raise FileExistsError(
            f"Destination run already exists; refusing to overwrite it: {destination}"
        )
    source_run = (
        source.parent.parent if source.parent.name == "checkpoints" else source.parent
    ).resolve()
    if destination == source_run:
        raise ValueError("Destination run must differ from the original run")

    destination.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "training", "diagnostics", "logs", "lineage"):
        (destination / name).mkdir()
    write_json_atomic(
        destination / "run_contract.json",
        {"mode": "branch_resume", "status": "prepared", "writable_run": True},
    )
    lineage = {
        "operation": "branch_resume",
        "source_run": str(source_run),
        "source_checkpoint": inspection,
        "destination_run": str(destination),
        "start_epoch": expected_epoch + 1,
        "original_run_writable": False,
    }
    write_json_atomic(destination / "lineage" / "resume.json", lineage)
    return lineage


def prepare_fresh_directory(destination_run: str | Path) -> dict[str, Any]:
    """Create a new training run and reject every existing destination."""

    destination = Path(destination_run).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Destination run already exists; refusing to overwrite it: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "training", "diagnostics", "logs", "lineage"):
        (destination / name).mkdir()
    contract = {"mode": "fresh", "status": "prepared", "writable_run": True}
    write_json_atomic(destination / "run_contract.json", contract)
    return contract


def _best_loss(checkpoint: dict[str, Any]) -> float | None:
    for key in (
        "best_validation_loss",
        "best_validation_weighted_nll",
        "best_validation_nll",
    ):
        if key in checkpoint and checkpoint[key] is not None:
            return float(checkpoint[key])
    return None


def _best_epoch(checkpoint: dict[str, Any]) -> int | None:
    saved_early = checkpoint.get("early_stopping_state") or {}
    value = checkpoint.get("best_epoch", saved_early.get("best_epoch"))
    return int(value) if value is not None else None


def restore_checkpoint(
    source_checkpoint: str | Path,
    destination_run: str | Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    data_loader_generator: Any | None = None,
    early_stopping_enabled: bool = False,
    early_stopping_patience: int = 20,
    expected_epoch: int = 450,
) -> tuple[EarlyStoppingState, ResumeReport]:
    """Restore available state while recording every non-restorable component."""

    torch = _torch()
    source = Path(source_checkpoint).resolve()
    destination = Path(destination_run).resolve()
    lineage_path = destination / "lineage" / "resume.json"
    if not lineage_path.is_file():
        raise FileNotFoundError(
            "Prepare a new resume directory before restoring training state"
        )
    lineage = json.loads(lineage_path.read_text())
    if lineage["source_checkpoint"]["sha256"] != sha256(source):
        raise ValueError(
            "Source checkpoint changed after the resume branch was prepared"
        )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    epoch = int(checkpoint["epoch"])
    if epoch != expected_epoch:
        raise ValueError(f"Expected epoch {expected_epoch}, got {epoch}")

    components: list[RestoredComponent] = []
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no model_state_dict; training cannot resume")
    model.load_state_dict(checkpoint["model_state_dict"])
    components.append(RestoredComponent("model", True))

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None:
        components.append(
            RestoredComponent(
                "optimizer",
                False,
                "Optimizer moments restart from zero, so epoch 451 is not an "
                "exact continuation.",
            )
        )
    else:
        optimizer.load_state_dict(optimizer_state)
        components.append(RestoredComponent("optimizer", True))

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is None:
        components.append(
            RestoredComponent(
                "scheduler",
                scheduler_state is None,
                "Checkpoint has scheduler state but this run has no scheduler."
                if scheduler_state is not None
                else None,
            )
        )
    elif scheduler_state is None:
        components.append(
            RestoredComponent(
                "scheduler",
                False,
                "Learning-rate schedule restarts and may diverge from the "
                "original trajectory.",
            )
        )
    else:
        scheduler.load_state_dict(scheduler_state)
        components.append(RestoredComponent("scheduler", True))

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is None:
        components.append(
            RestoredComponent(
                "scaler",
                scaler_state is None,
                "Checkpoint used AMP scaling but this run has no scaler."
                if scaler_state is not None
                else None,
            )
        )
    elif scaler_state is None:
        components.append(
            RestoredComponent(
                "scaler",
                False,
                "AMP scale restarts; early resumed steps may scale gradients "
                "differently.",
            )
        )
    else:
        scaler.load_state_dict(scaler_state)
        components.append(RestoredComponent("scaler", True))

    best_loss = _best_loss(checkpoint)
    best_epoch = _best_epoch(checkpoint)
    legacy_counter = checkpoint.get("epochs_without_improvement")
    saved_early = checkpoint.get("early_stopping_state")
    if saved_early is not None:
        early = EarlyStoppingState(
            enabled=early_stopping_enabled,
            patience=early_stopping_patience,
            best_validation_loss=float(saved_early["best_validation_loss"]),
            best_epoch=(
                int(saved_early["best_epoch"])
                if saved_early.get("best_epoch") is not None
                else best_epoch
            ),
            epochs_without_improvement=int(saved_early["epochs_without_improvement"]),
        )
        components.append(RestoredComponent("early_stopping", True))
    elif best_loss is not None and legacy_counter is not None:
        early = EarlyStoppingState(
            enabled=early_stopping_enabled,
            patience=early_stopping_patience,
            best_validation_loss=best_loss,
            best_epoch=best_epoch,
            epochs_without_improvement=int(legacy_counter),
        )
        components.append(RestoredComponent("early_stopping", True))
    else:
        early = EarlyStoppingState(
            enabled=early_stopping_enabled,
            patience=early_stopping_patience,
            best_validation_loss=best_loss if best_loss is not None else float("inf"),
            best_epoch=best_epoch,
        )
        components.append(
            RestoredComponent(
                "early_stopping",
                False,
                "Patience counter restarts; this matters only when early "
                "stopping is enabled.",
            )
        )
    components.append(
        RestoredComponent(
            "best_validation_loss",
            best_loss is not None,
            None
            if best_loss is not None
            else (
                "Best tracking restarts at infinity and may select a different "
                "best epoch."
            ),
        )
    )
    components.append(
        RestoredComponent(
            "best_epoch",
            best_epoch is not None,
            None
            if best_epoch is not None
            else "The historical best loss may be known, but its epoch is unknown.",
        )
    )

    rng_state = checkpoint.get("rng_state")
    if rng_state is None:
        components.append(
            RestoredComponent(
                "rng",
                False,
                "Data order and stochastic operations cannot exactly reproduce "
                "the old trajectory.",
            )
        )
    else:
        restore_rng_state(rng_state)
        components.append(RestoredComponent("rng", True))

    loader_state = checkpoint.get("data_loader_generator_state")
    if data_loader_generator is None:
        components.append(
            RestoredComponent(
                "data_loader_rng",
                loader_state is None,
                "Checkpoint has data-loader RNG state but this run did not provide "
                "its generator."
                if loader_state is not None
                else None,
            )
        )
    elif loader_state is None:
        components.append(
            RestoredComponent(
                "data_loader_rng",
                False,
                "Shuffled minibatch order cannot exactly continue from epoch 450.",
            )
        )
    else:
        data_loader_generator.set_state(loader_state)
        components.append(RestoredComponent("data_loader_rng", True))

    components.append(
        RestoredComponent(
            "best_model_weights",
            checkpoint.get("best_model_state_dict") is not None,
            None
            if checkpoint.get("best_model_state_dict") is not None
            else (
                "The historical best metric is known, but its model weights are "
                "not embedded; "
                "retain the parent best checkpoint as a lineage reference."
            ),
        )
    )
    report = ResumeReport(
        source_checkpoint=str(source),
        source_epoch=epoch,
        start_epoch=epoch + 1,
        destination_run=str(destination),
        components=tuple(components),
    )
    write_json_atomic(destination / "lineage" / "restore_report.json", report.to_dict())
    return early, report


class CheckpointWriter:
    """Write only inside one fresh run; periodic checkpoints contain full state."""

    def __init__(self, run_directory: str | Path, *, interval: int = 50) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.directory = self.run_directory / "checkpoints"
        if not (self.run_directory / "run_contract.json").is_file():
            raise ValueError("CheckpointWriter requires a prepared run directory")
        if interval <= 0:
            raise ValueError("Checkpoint interval must be positive")
        self.interval = interval

    def save(self, epoch: int, payload: dict[str, Any], *, improved: bool) -> None:
        save_checkpoint_atomic(self.directory / "last_model.pt", payload)
        if improved:
            save_checkpoint_atomic(self.directory / "best_model.pt", payload)
        if epoch % self.interval == 0:
            numbered = self.directory / f"epoch_{epoch:04d}_model.pt"
            if numbered.exists():
                raise FileExistsError(f"Refusing to overwrite checkpoint: {numbered}")
            save_checkpoint_atomic(numbered, payload)
