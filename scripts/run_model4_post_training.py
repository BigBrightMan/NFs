#!/usr/bin/env python3
"""Run the complete post-training stage for one immutable Model 4 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from flashsim_nf.generation import resolve_generation_config
from flashsim_nf.pipeline import execute_post_training_config


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
    args = parser.parse_args()

    run = Path(args.run_directory).resolve()
    training_config = json.loads((run / "config_resolved.json").read_text())
    config = resolve_generation_config(
        run_directory=run,
        purpose=args.purpose,
        guard_artifact=args.guard_artifact,
        number_events=args.number_events,
        generation_seed=args.generation_seed,
        batch_size=args.batch_size,
        maximum_attempt_multiplier=args.maximum_attempt_multiplier,
        device=args.device,
        frozen_selection=args.frozen_selection,
    )
    seed = config["generation"]["seed"]
    output_root = Path(training_config["resolution"]["output_root"])
    dataset_id = config["dataset"]["dataset_id"]
    run_id = training_config["resolution"]["run_id"]
    config_path = (
        output_root
        / "resolved_configs"
        / "model4_generation"
        / dataset_id
        / args.purpose
        / f"{run_id}_genseed{seed}.yaml"
    )
    if config_path.exists():
        raise FileExistsError(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    summary = execute_post_training_config(config_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
