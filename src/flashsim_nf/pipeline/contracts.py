"""Small composition-first interfaces for pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class PipelineContext:
    dataset_id: str
    data_root: Path
    output_root: Path
    resolved_config: dict[str, Any]


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    outputs: tuple[Path, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)


class Stage(Protocol):
    name: str

    def run(self, context: PipelineContext) -> StageResult:
        """Execute one stage using only its declared context."""
        ...
