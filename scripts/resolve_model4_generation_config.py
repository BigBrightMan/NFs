#!/usr/bin/env python3
"""Resolve one immutable Model 4 generation config from a completed run."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from flashsim_nf.generation import resolve_generation_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True)
    parser.add_argument(
        "--purpose", choices=("validation", "test", "muondis"), required=True
    )
    parser.add_argument("--guard-artifact", required=True)
    parser.add_argument("--number-events", type=int)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--maximum-attempt-multiplier", type=float, default=2.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frozen-selection")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = resolve_generation_config(
        run_directory=args.run_directory,
        purpose=args.purpose,
        guard_artifact=args.guard_artifact,
        number_events=args.number_events,
        generation_seed=args.generation_seed,
        batch_size=args.batch_size,
        maximum_attempt_multiplier=args.maximum_attempt_multiplier,
        device=args.device,
        frozen_selection=args.frozen_selection,
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
