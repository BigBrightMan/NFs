#!/usr/bin/env python3
"""Create Pipeline D, drop-z-pz, and 2025 full-8D experiment configs.

Pipeline D is Pipeline A with `E` and `pz` both on natural log. Under `drop_ze`
the trained difference from A is `pz` alone; under `drop_z_pz` it is `E` alone.
Both comparisons therefore stay single-variable with one fitted pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashsim_nf.experiments import create_model4_baseline_configs


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("local", "cern"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config-output-directory", required=True)
    parser.add_argument("--stage", choices=("smoke", "production"), default="smoke")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    destination = Path(args.config_output_directory).resolve()
    common = {
        "project_root": project,
        "output_root": args.output_root,
        "environment": args.environment,
        "stage": args.stage,
        "training_seed": args.training_seed,
        "skip_existing": args.skip_existing,
    }
    records = []
    records.extend(
        create_model4_baseline_configs(
            **common,
            config_output_directory=destination / "pipeline_D",
            pipelines=("D",),
            ablation="drop_ze",
            preprocessing_data_root=args.data_root,
        )
    )
    records.extend(
        create_model4_baseline_configs(
            **common,
            config_output_directory=destination / "drop_z_pz",
            pipelines=("A", "D"),
            ablation="drop_z_pz",
            preprocessing_data_root=args.data_root,
        )
    )
    if args.stage == "smoke":
        records.extend(
            create_model4_baseline_configs(
                **common,
                config_output_directory=destination / "full_8d_2025",
                pipelines=("A",),
                ablation="none",
                dataset_ids=("fluka2025_muons_horizontal",),
            )
        )
    print(
        json.dumps(
            {
                "number_configs": len(records),
                "configs": [record.to_dict() for record in records],
                "test_data_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
