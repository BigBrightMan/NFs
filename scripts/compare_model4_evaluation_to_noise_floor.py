#!/usr/bin/env python3
"""Compare generated-validation scores with the FLUKA-vs-FLUKA q95 floor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashsim_nf.evaluation import selection_metrics
from flashsim_nf.manifest import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--noise-floor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluation_path = Path(args.evaluation).resolve()
    floor_path = Path(args.noise_floor).resolve()
    payload = json.loads(evaluation_path.read_text())
    floor = json.loads(floor_path.read_text())
    if payload.get("reference_split") != "validation":
        raise ValueError("Noise-floor comparison is for validation evaluation only")
    if payload.get("dataset_id") != floor.get("dataset_id"):
        raise ValueError("Evaluation and noise floor use different datasets")
    scores = selection_metrics(payload["evaluation"])
    comparison = {}
    for name, value in scores.items():
        q95 = float(floor["bootstrap"]["summary"][name]["q95"])
        comparison[name] = {
            "model": value,
            "reference_noise_q95": q95,
            "model_over_noise_q95": value / q95 if q95 > 0 else None,
            "within_reference_noise_q95": value <= q95,
        }
    report = {
        "status": "complete",
        "dataset_id": payload["dataset_id"],
        "test_data_used": False,
        "evaluation": str(evaluation_path),
        "noise_floor": str(floor_path),
        "comparison": comparison,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
