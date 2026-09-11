#!/usr/bin/env python3
"""Compute validation-only FLUKA noise-floor reports for all maintained years."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashsim_nf.config import load_yaml
from flashsim_nf.evaluation import compute_reference_noise_floor


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("local", "cern"), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--row-matched-rows", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    reports = []
    for config_path in sorted((project / "configs/datasets").glob("*.yaml")):
        dataset = load_yaml(config_path)
        dataset_id = str(dataset["dataset_id"])
        prepared = Path(dataset["legacy_prepared_paths"][args.environment])
        output = (
            Path(args.output_root).resolve()
            / "campaigns"
            / dataset_id
            / "model4"
            / "reference_noise_floor"
            / "validation"
            / "metrics.json"
        )
        if output.exists() and args.skip_existing:
            reports.append(
                {"dataset_id": dataset_id, "status": "existing", "output": str(output)}
            )
            continue
        report = compute_reference_noise_floor(
            dataset_id=dataset_id,
            train_root=prepared / "split/train_rawfeature.root",
            validation_root=prepared / "split/validation_rawfeature.root",
            output=output,
            repeats=args.repeats,
            row_matched_rows=args.row_matched_rows,
        )
        reports.append(
            {
                "dataset_id": dataset_id,
                "status": report["status"],
                "output": str(output),
            }
        )
    print(json.dumps({"number_datasets": len(reports), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
