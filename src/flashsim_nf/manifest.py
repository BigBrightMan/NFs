"""Run-manifest and central-index primitives."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    display_name: str
    status: str
    dataset_id: str
    year: int
    preprocessing: str
    trial_id: str
    training_seed: int
    config_hash: str
    resolved_config: str
    source_commit: str
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically beside the destination."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def write_run_manifest(path: str | Path, manifest: RunManifest) -> None:
    write_json_atomic(path, manifest.to_dict())


def build_run_index(
    output_root: str | Path, destination: str | Path
) -> list[dict[str, Any]]:
    """Discover run manifests and write a deterministic CSV index."""
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("campaigns/**/manifest.json")):
        payload = json.loads(path.read_text())
        rows.append(
            {
                "run_id": payload["run_id"],
                "display_name": payload["display_name"],
                "status": payload["status"],
                "dataset_id": payload["dataset_id"],
                "year": payload["year"],
                "preprocessing": payload["preprocessing"],
                "trial_id": payload["trial_id"],
                "training_seed": payload["training_seed"],
                "config_hash": payload["config_hash"],
                "manifest": str(path),
            }
        )

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "display_name",
        "status",
        "dataset_id",
        "year",
        "preprocessing",
        "trial_id",
        "training_seed",
        "config_hash",
        "manifest",
    ]
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return rows
