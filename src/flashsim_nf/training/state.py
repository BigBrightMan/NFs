"""Configuration and state records for restart-safe training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EarlyStoppingState:
    """Serializable early-stopping state; disabled by default."""

    enabled: bool = False
    patience: int = 20
    best_validation_loss: float = float("inf")
    best_epoch: int | None = None
    epochs_without_improvement: int = 0

    def observe(self, validation_loss: float, *, epoch: int | None = None) -> bool:
        improved = validation_loss < self.best_validation_loss
        if improved:
            self.best_validation_loss = float(validation_loss)
            self.best_epoch = int(epoch) if epoch is not None else self.best_epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.enabled and self.epochs_without_improvement >= self.patience

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingPolicy:
    maximum_epochs: int = 500
    checkpoint_interval: int = 50
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 20

    def __post_init__(self) -> None:
        if self.maximum_epochs <= 0:
            raise ValueError("maximum_epochs must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TrainingPolicy:
        early = config.get("early_stopping", {"enabled": False, "patience": 20})
        if isinstance(early, bool):
            enabled = early
            patience = int(config.get("early_stopping_patience", 20))
        elif isinstance(early, Mapping):
            enabled = bool(early.get("enabled", False))
            patience = int(early.get("patience", 20))
        else:
            raise TypeError("training.early_stopping must be a bool or mapping")
        return cls(
            maximum_epochs=int(
                config.get("maximum_epochs", config.get("max_epochs", 500))
            ),
            checkpoint_interval=int(config.get("checkpoint_interval", 50)),
            early_stopping_enabled=enabled,
            early_stopping_patience=patience,
        )


@dataclass(frozen=True)
class RestoredComponent:
    name: str
    restored: bool
    consequence: str | None = None


@dataclass(frozen=True)
class ResumeReport:
    source_checkpoint: str
    source_epoch: int
    start_epoch: int
    destination_run: str
    components: tuple[RestoredComponent, ...] = field(default_factory=tuple)

    @property
    def missing(self) -> tuple[RestoredComponent, ...]:
        return tuple(
            component for component in self.components if not component.restored
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_checkpoint": self.source_checkpoint,
            "source_epoch": self.source_epoch,
            "start_epoch": self.start_epoch,
            "destination_run": self.destination_run,
            "components": [asdict(component) for component in self.components],
        }
