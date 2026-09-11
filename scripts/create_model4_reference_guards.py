#!/usr/bin/env python3
"""Create the four-year train/all-FLUKA Model 4 guard matrix safely."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from flashsim_nf.config import load_yaml


def _complete_output(path: Path, dataset_id: str) -> bool:
    required = (
        path / "guard_train_ref.json",
        path / "guard_all_ref.json",
        path / "guard_comparison.json",
        path / "guard_bounds.csv",
        path / "manifest.json",
        path / "_SUCCESS.json",
    )
    if not all(item.is_file() for item in required):
        return False
    train = json.loads(required[0].read_text())
    all_clean = json.loads(required[1].read_text())
    return (
        train.get("dataset_id") == dataset_id
        and train.get("format_version") == 3
        and train.get("fit_scope") == "train"
        and train.get("selection_allowed") is True
        and train.get("empirical_support", {}).get("contract_id")
        == "per_dataset_observed_envelope_v2"
        and train.get("empirical_support", {}).get("action") == "operational_reject"
        and all_clean.get("dataset_id") == dataset_id
        and all_clean.get("format_version") == 3
        and all_clean.get("fit_scope") == "all_clean_splits"
        and all_clean.get("selection_allowed") is False
        and all_clean.get("empirical_support", {}).get("contract_id")
        == "per_dataset_observed_envelope_v2"
        and all_clean.get("empirical_support", {}).get("action")
        == "operational_reject"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config-root", default="configs/datasets")
    parser.add_argument("--environment", choices=("local", "cern"), default="cern")
    parser.add_argument(
        "--guard-root", default="/eos/user/t/tanansub/SWAN_projects/NFs_data/guards"
    )
    parser.add_argument(
        "--settings", default="configs/guards/reference_scope_comparison.yaml"
    )
    parser.add_argument("--expected-count", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_paths = sorted(Path(args.dataset_config_root).glob("fluka*.yaml"))
    comparison = load_yaml(args.settings)
    comparison_id = str(comparison["comparison_id"])
    if len(config_paths) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} dataset configs, found {len(config_paths)}"
        )
    plans = []
    failures = []
    for config_path in config_paths:
        dataset = load_yaml(config_path)
        dataset_id = str(dataset["dataset_id"])
        prepared = Path(dataset["legacy_prepared_paths"][args.environment]).resolve()
        required_inputs = [
            prepared / "split/split_manifest.json",
            *[
                prepared / "split" / f"{split}_rawfeature.root"
                for split in ("train", "validation", "test")
            ],
        ]
        missing = [str(path) for path in required_inputs if not path.is_file()]
        if missing:
            failures.append(f"{dataset_id}: missing inputs {missing}")
        output = (
            Path(args.guard_root).resolve()
            / dataset_id
            / comparison_id
        )
        if output.exists() and not _complete_output(output, dataset_id):
            failures.append(f"{dataset_id}: incomplete/conflicting output {output}")
        plans.append(
            {
                "dataset_id": dataset_id,
                "dataset_config": config_path.resolve(),
                "prepared_directory": prepared,
                "output_directory": output,
                "status": "verified_existing" if output.exists() else "planned",
            }
        )
    if failures:
        raise RuntimeError("Guard matrix preflight failed:\n- " + "\n- ".join(failures))

    print(
        json.dumps(
            {
                "status": "pass",
                "environment": args.environment,
                "count": len(plans),
                "dry_run": args.dry_run,
                "plans": [
                    {key: str(value) for key, value in plan.items()}
                    for plan in plans
                ],
            },
            indent=2,
        )
    )
    if args.dry_run:
        return
    worker = Path(__file__).with_name("fit_compare_reference_guards.py")
    for plan in plans:
        if plan["status"] == "verified_existing":
            continue
        subprocess.run(
            [
                sys.executable,
                str(worker),
                "--dataset-config",
                str(plan["dataset_config"]),
                "--prepared-directory",
                str(plan["prepared_directory"]),
                "--output-directory",
                str(plan["output_directory"]),
                "--settings",
                str(Path(args.settings).resolve()),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
