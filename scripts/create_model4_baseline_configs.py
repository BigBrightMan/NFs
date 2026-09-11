#!/usr/bin/env python3
"""Create real baseline configs for selected Model 4 years and pipelines."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from flashsim_nf.experiments import create_model4_baseline_configs


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("local", "cern"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-output-directory", required=True)
    parser.add_argument("--stage", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--pipelines", nargs="+", default=["A", "B", "C"])
    parser.add_argument(
        "--ablation", choices=("drop_ze", "drop_z_pz", "none"), default="drop_ze"
    )
    parser.add_argument("--preprocessing-data-root")
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    records = create_model4_baseline_configs(
        project_root=project,
        output_root=args.output_root,
        config_output_directory=args.config_output_directory,
        environment=args.environment,
        stage=args.stage,
        pipelines=args.pipelines,
        ablation=args.ablation,
        preprocessing_data_root=args.preprocessing_data_root,
        dataset_ids=args.dataset_ids,
        training_seed=args.training_seed,
        skip_existing=args.skip_existing,
    )
    statuses = Counter(record.status for record in records)
    print(
        json.dumps(
            {
                "number_configs": len(records),
                "status_counts": dict(sorted(statuses.items())),
                "configs": [record.to_dict() for record in records],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
