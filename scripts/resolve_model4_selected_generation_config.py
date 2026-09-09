#!/usr/bin/env python3
"""Resolve final-test or MuonDIS generation directly from a frozen winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from flashsim_nf.generation import resolve_generation_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--purpose", choices=("test", "muondis"), required=True)
    parser.add_argument("--guard-artifact", required=True)
    parser.add_argument("--number-events", type=int)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selection = Path(args.selection).resolve()
    artifact = json.loads(selection.read_text())
    if (
        artifact.get("status") != "frozen"
        or artifact.get("test_data_used") is not False
    ):
        raise ValueError("Selection must be frozen from validation without test data")
    run = artifact.get("winner", {}).get("training_run")
    if not run:
        raise ValueError(
            "Selection artifact does not identify the winning training run"
        )
    config = resolve_generation_config(
        run_directory=run,
        purpose=args.purpose,
        guard_artifact=args.guard_artifact,
        number_events=args.number_events,
        generation_seed=args.generation_seed,
        batch_size=args.batch_size,
        frozen_selection=selection,
    )
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    print(destination)
    print(config["output"]["directory"])


if __name__ == "__main__":
    main()
