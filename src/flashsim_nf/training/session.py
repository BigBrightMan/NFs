"""A callback-driven training loop that composes with any NF implementation."""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional, Union

from ..manifest import write_json_atomic
from .checkpoints import (
    CheckpointWriter,
    full_checkpoint_payload,
    prepare_fresh_directory,
    prepare_resume_directory,
    restore_checkpoint,
)
from .state import EarlyStoppingState, ResumeReport, TrainingPolicy

EpochResult = Union[float, Mapping[str, float]]
TrainEpoch = Callable[[Any, Any, Optional[Any], int], EpochResult]
ValidateEpoch = Callable[[Any, int], EpochResult]


def _epoch_result(value: EpochResult, *, stage: str) -> tuple[float, dict[str, float]]:
    if isinstance(value, Mapping):
        if "loss" not in value:
            raise ValueError(f"{stage} epoch metrics must contain 'loss'")
        metrics = {str(key): float(item) for key, item in value.items()}
        return metrics.pop("loss"), metrics
    return float(value), {}


class TrainingSession:
    """Run fresh or resumed training through injected epoch operations."""

    def __init__(
        self,
        *,
        run_directory: str | Path,
        model: Any,
        optimizer: Any,
        scheduler: Any | None,
        scaler: Any | None,
        policy: TrainingPolicy,
        config: dict[str, Any],
        early_stopping: EarlyStoppingState,
        start_epoch: int,
        mode: str,
        resume_report: ResumeReport | None = None,
        best_model_state_dict: dict[str, Any] | None = None,
        data_loader_generator: Any | None = None,
    ) -> None:
        self.run_directory = Path(run_directory).resolve()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.policy = policy
        self.config = config
        self.early_stopping = early_stopping
        self.start_epoch = start_epoch
        self.mode = mode
        self.resume_report = resume_report
        self.best_model_state_dict = best_model_state_dict
        self.data_loader_generator = data_loader_generator
        self.writer = CheckpointWriter(
            self.run_directory, interval=policy.checkpoint_interval
        )

    @classmethod
    def fresh(
        cls,
        *,
        destination_run: str | Path,
        model: Any,
        optimizer: Any,
        scheduler: Any | None,
        scaler: Any | None,
        data_loader_generator: Any | None = None,
        policy: TrainingPolicy,
        config: dict[str, Any],
    ) -> TrainingSession:
        prepare_fresh_directory(destination_run)
        write_json_atomic(Path(destination_run) / "config_resolved.json", config)
        return cls(
            run_directory=destination_run,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            policy=policy,
            config=config,
            early_stopping=EarlyStoppingState(
                enabled=policy.early_stopping_enabled,
                patience=policy.early_stopping_patience,
            ),
            start_epoch=1,
            mode="fresh",
            data_loader_generator=data_loader_generator,
        )

    @classmethod
    def branch(
        cls,
        *,
        source_checkpoint: str | Path,
        destination_run: str | Path,
        model: Any,
        optimizer: Any,
        scheduler: Any | None,
        scaler: Any | None,
        data_loader_generator: Any | None = None,
        policy: TrainingPolicy,
        config: dict[str, Any],
        expected_epoch: int = 450,
    ) -> TrainingSession:
        if policy.maximum_epochs <= expected_epoch:
            raise ValueError(
                "maximum_epochs must be greater than the resume checkpoint epoch"
            )
        prepare_resume_directory(
            source_checkpoint, destination_run, expected_epoch=expected_epoch
        )
        write_json_atomic(Path(destination_run) / "config_resolved.json", config)
        early, report = restore_checkpoint(
            source_checkpoint,
            destination_run,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_loader_generator=data_loader_generator,
            early_stopping_enabled=policy.early_stopping_enabled,
            early_stopping_patience=policy.early_stopping_patience,
            expected_epoch=expected_epoch,
        )
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Branched training requires torch") from error
        source_payload = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        return cls(
            run_directory=destination_run,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            policy=policy,
            config=config,
            early_stopping=early,
            start_epoch=report.start_epoch,
            mode="branch_resume",
            resume_report=report,
            best_model_state_dict=source_payload.get("best_model_state_dict"),
            data_loader_generator=data_loader_generator,
        )

    def run(
        self,
        *,
        train_epoch: TrainEpoch,
        validate_epoch: ValidateEpoch,
    ) -> list[dict[str, Any]]:
        """Run the configured epoch range only inside the prepared run."""

        history: list[dict[str, Any]] = []
        best_model_state = self.best_model_state_dict
        for epoch in range(self.start_epoch, self.policy.maximum_epochs + 1):
            started = time.monotonic()
            train_loss, train_metrics = _epoch_result(
                train_epoch(self.model, self.optimizer, self.scaler, epoch),
                stage="train",
            )
            validation_loss, validation_metrics = _epoch_result(
                validate_epoch(self.model, epoch), stage="validation"
            )
            if self.scheduler is not None:
                self.scheduler.step(validation_loss)
            improved = self.early_stopping.observe(validation_loss, epoch=epoch)
            if improved:
                best_model_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "best_validation_loss": self.early_stopping.best_validation_loss,
                "epochs_without_improvement": (
                    self.early_stopping.epochs_without_improvement
                ),
                "epoch_seconds": time.monotonic() - started,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            history.append(row)
            self._write_history(history)
            payload = full_checkpoint_payload(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                early_stopping=self.early_stopping,
                config=self.config,
                best_model_state_dict=best_model_state,
                data_loader_generator=self.data_loader_generator,
            )
            self.writer.save(epoch, payload, improved=improved)
            progress = [
                f"Epoch {epoch}/{self.policy.maximum_epochs}",
                f"train NLL={train_loss:.6f}",
                f"validation NLL={validation_loss:.6f}",
                (
                    "best="
                    f"{self.early_stopping.best_validation_loss:.6f}"
                    f"@{self.early_stopping.best_epoch}"
                ),
                f"LR={row['learning_rate']:.3e}",
            ]
            if "train_gradient_norm" in row:
                progress.append(
                    f"gradient norm={row['train_gradient_norm']:.4f}"
                )
            if "train_batch_weight_mean_cv" in row:
                progress.append(
                    f"weight CV={row['train_batch_weight_mean_cv']:.4f}"
                )
            progress.append(f"time={row['epoch_seconds']:.1f}s")
            if improved:
                progress.append("new best")
            print(" | ".join(progress), flush=True)
            if self.early_stopping.should_stop:
                break
        summary = {
            "status": "complete",
            "mode": self.mode,
            "source_epoch": (
                self.resume_report.source_epoch if self.resume_report else None
            ),
            "start_epoch": self.start_epoch,
            "last_epoch": history[-1]["epoch"],
            "maximum_epochs": self.policy.maximum_epochs,
            "early_stopping_enabled": self.early_stopping.enabled,
            "stopped_early": self.early_stopping.should_stop,
            "best_validation_loss": self.early_stopping.best_validation_loss,
            "best_epoch": self.early_stopping.best_epoch,
        }
        write_json_atomic(self.run_directory / "training_summary.json", summary)
        write_json_atomic(
            self.run_directory / "_SUCCESS.json",
            {"stage": "training", "last_epoch": history[-1]["epoch"]},
        )
        return history

    def _write_history(self, rows: list[dict[str, Any]]) -> None:
        destination = self.run_directory / "training" / "training_history.csv"
        temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)


# Compatibility for code written before fresh-run support was added.
BranchedTrainingSession = TrainingSession
