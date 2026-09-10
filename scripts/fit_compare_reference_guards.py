#!/usr/bin/env python3
"""Fit train/all-FLUKA robust guards and compare them on identical data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from flashsim_nf.config import load_yaml
from flashsim_nf.guards import (
    HARD_SUPPORT_FEATURES,
    PHYSICAL_FEATURES,
    compare_reference_guards,
    fit_reference_guard,
    load_generated_root,
    load_reference_splits,
)
from flashsim_nf.manifest import write_json_atomic


def _write_bound_table(path: Path, report: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "feature",
                "train_physical_lower",
                "train_physical_upper",
                "all_physical_lower",
                "all_physical_upper",
                "hard_support_train_lower",
                "hard_support_train_upper",
                "hard_support_all_lower",
                "hard_support_all_upper",
            ],
        )
        writer.writeheader()
        for feature in PHYSICAL_FEATURES:
            row = {"feature": feature, **report["bounds"][feature]}
            if feature in HARD_SUPPORT_FEATURES:
                support = report["hard_support"][feature]
                row.update(
                    {
                        "hard_support_train_lower": support["train_physical_lower"],
                        "hard_support_train_upper": support["train_physical_upper"],
                        "hard_support_all_lower": support["all_physical_lower"],
                        "hard_support_all_upper": support["all_physical_upper"],
                    }
                )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--prepared-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument(
        "--settings",
        default="configs/guards/reference_scope_comparison.yaml",
    )
    parser.add_argument("--generated-root")
    args = parser.parse_args()

    dataset = load_yaml(args.dataset_config)
    comparison_config = load_yaml(args.settings)
    settings = comparison_config["robust_reference"]
    output = Path(args.output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing overwrite: {output}")
    splits = load_reference_splits(dataset, args.prepared_directory)
    common = {
        "dataset_id": str(dataset["dataset_id"]),
        "dataset_fingerprint": str(dataset["dataset_fingerprint"]),
        "split_id": str(dataset["split"]["id"]),
        "iqr_multiplier": float(settings["iqr_multiplier"]),
        "lower_quantile": float(settings["lower_quantile"]),
        "upper_quantile": float(settings["upper_quantile"]),
    }
    train_guard = fit_reference_guard(splits, fit_scope="train", **common)
    all_guard = fit_reference_guard(splits, fit_scope="all_clean_splits", **common)
    generated = (
        load_generated_root(args.generated_root) if args.generated_root else None
    )
    comparison = compare_reference_guards(
        splits,
        train_guard=train_guard,
        all_guard=all_guard,
        generated=generated,
    )

    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output / "guard_train_ref.json", train_guard)
    write_json_atomic(output / "guard_all_ref.json", all_guard)
    write_json_atomic(output / "guard_comparison.json", comparison)
    _write_bound_table(output / "guard_bounds.csv", comparison)
    manifest = {
        "stage": "guard_reference_scope_comparison",
        "comparison_id": comparison_config["comparison_id"],
        "dataset_id": dataset["dataset_id"],
        "dataset_config": str(Path(args.dataset_config).resolve()),
        "comparison_config": str(Path(args.settings).resolve()),
        "prepared_directory": str(Path(args.prepared_directory).resolve()),
        "generated_root": (
            str(Path(args.generated_root).resolve()) if args.generated_root else None
        ),
        "outputs": [
            "guard_train_ref.json",
            "guard_all_ref.json",
            "guard_comparison.json",
            "guard_bounds.csv",
        ],
    }
    write_json_atomic(output / "manifest.json", manifest)
    write_json_atomic(output / "_SUCCESS.json", {"stage": manifest["stage"]})
    print(json.dumps({"output_directory": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
