#!/usr/bin/env python3
"""Resolve one real Model 4 training config for any maintained campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashsim_nf.experiments import (
    resolve_model4_training_config,
    write_resolved_training_config,
)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--prepared-directory", required=True)
    parser.add_argument("--preprocessing-directory")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-output", required=True)
    parser.add_argument(
        "--preprocessing", choices=("A", "B", "C", "D", "E"), required=True
    )
    parser.add_argument(
        "--ablation", choices=("drop_ze", "drop_z_pz", "none"), default="drop_ze"
    )
    parser.add_argument("--stage", choices=("smoke", "production"), required=True)
    parser.add_argument("--trial-id", default="base")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--maximum-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    early = parser.add_mutually_exclusive_group()
    early.add_argument("--early-stopping", action="store_true", dest="early")
    early.add_argument("--no-early-stopping", action="store_false", dest="early")
    parser.set_defaults(early=None)
    args = parser.parse_args()

    config = resolve_model4_training_config(
        project_root=project,
        dataset_config=args.dataset_config,
        prepared_directory=args.prepared_directory,
        preprocessing_directory=args.preprocessing_directory,
        output_root=args.output_root,
        preprocessing=args.preprocessing,
        ablation=args.ablation,
        stage=args.stage,
        trial_id=args.trial_id,
        training_seed=args.training_seed,
        maximum_epochs=args.maximum_epochs,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        early_stopping=args.early,
    )
    path = write_resolved_training_config(args.config_output, config)
    print(
        json.dumps(
            {
                "resolved_config": str(path),
                "run_id": config["resolution"]["run_id"],
                "config_hash": config["resolution"]["config_hash"],
                "run_directory": config["output"]["run_directory"],
                "expected_train_rows": config["data"]["train"]["expected_rows"],
                "expected_validation_rows": config["data"]["validation"][
                    "expected_rows"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
