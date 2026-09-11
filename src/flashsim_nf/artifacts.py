"""Canonical NFs data and output path construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .naming import validate_slug


@dataclass(frozen=True)
class ArtifactRoots:
    data: Path
    output: Path


@dataclass(frozen=True)
class RunKey:
    dataset_id: str
    preprocessing: str
    ablation: str
    trial_id: str
    training_seed: int
    config_hash: str

    def __post_init__(self) -> None:
        validate_slug(self.dataset_id, field="dataset_id")
        validate_slug(self.ablation, field="ablation")
        validate_slug(self.trial_id, field="trial_id")
        validate_slug(self.config_hash, field="config_hash")
        if self.preprocessing not in {"A", "B", "C", "D", "E"}:
            raise ValueError("preprocessing must be A, B, C, D, or E")
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")


class ArtifactStore:
    """Create canonical paths without performing scientific I/O."""

    def __init__(self, roots: ArtifactRoots) -> None:
        self.roots = roots

    def campaign_data(self, dataset_id: str) -> Path:
        validate_slug(dataset_id, field="dataset_id")
        return self.roots.data / "campaigns" / dataset_id

    def tuning_run(self, key: RunKey) -> Path:
        return (
            self.roots.output
            / "campaigns"
            / key.dataset_id
            / "model4"
            / f"preprocessing_{key.preprocessing}"
            / key.ablation
            / "tuning"
            / key.trial_id
            / f"config_{key.config_hash}"
            / f"train_seed_{key.training_seed}"
        )

    def model_selection(self, dataset_id: str) -> Path:
        validate_slug(dataset_id, field="dataset_id")
        return (
            self.roots.output
            / "campaigns"
            / dataset_id
            / "model4"
            / "model_selection"
            / "validation"
        )
