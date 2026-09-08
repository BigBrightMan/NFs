#!/usr/bin/env python3
"""Resolve selected Model 4 tuning runs into one reviewable JSON plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashsim_nf.experiments import create_tuning_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--campaign", required=True)
    parser.add_argument(
        "--tuning-space", default="configs/models/model4/tuning_space.yaml"
    )
    parser.add_argument("--pipelines", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--trials", nargs="+")
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--write")
    parser.add_argument(
        "--baseline-already-complete",
        action="store_true",
        help=(
            "Allow a subset plan without baseline when its frozen baseline "
            "already exists"
        ),
    )
    args = parser.parse_args()

    runs = create_tuning_plan(
        project_root=args.project_root,
        campaign_config=args.campaign,
        tuning_space=args.tuning_space,
        output_root=args.output_root,
        data_root=args.data_root,
        pipelines=tuple(args.pipelines),
        trial_ids=tuple(args.trials) if args.trials else None,
        training_seeds=tuple(args.training_seeds),
        require_baseline=not args.baseline_already_complete,
    )
    payload = {"number_runs": len(runs), "runs": [run.to_dict() for run in runs]}
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.write:
        destination = Path(args.write)
        if destination.exists():
            parser.error(f"Refusing to overwrite existing plan: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered)
        print(f"wrote {destination}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
