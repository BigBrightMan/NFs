#!/usr/bin/env python3
"""Evaluate generated Model 4 physical kinematics against weighted FLUKA."""

from __future__ import annotations

import argparse
import json

from flashsim_nf.evaluation import EvaluationSettings, evaluate_generated_root_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--train-reference-root", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--reference-split", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--tree-name", default="nt")
    parser.add_argument(
        "--generated-weight-mode",
        choices=("auto", "branch", "uniform"),
        default="auto",
    )
    parser.add_argument("--energy-sample-size", type=int, default=1000)
    parser.add_argument("--energy-repeats", type=int, default=5)
    parser.add_argument("--sliced-projections", type=int, default=64)
    parser.add_argument("--minimum-tail-ess", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=1556)
    args = parser.parse_args()
    settings = EvaluationSettings(
        energy_sample_size=args.energy_sample_size,
        energy_repeats=args.energy_repeats,
        sliced_wasserstein_projections=args.sliced_projections,
        minimum_tail_ess=args.minimum_tail_ess,
        random_seed=args.seed,
    )
    report = evaluate_generated_root_files(
        dataset_id=args.dataset_id,
        train_reference_root=args.train_reference_root,
        reference_root=args.reference_root,
        generated_root=args.generated_root,
        output_directory=args.output_directory,
        reference_split=args.reference_split,
        tree_name=args.tree_name,
        generated_weight_mode=args.generated_weight_mode,
        settings=settings,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
