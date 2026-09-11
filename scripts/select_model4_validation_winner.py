#!/usr/bin/env python3
"""Freeze one winner using generated-validation metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashsim_nf.evaluation import selection_metrics
from flashsim_nf.manifest import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(value).resolve() for value in args.evaluations]
    candidates = []
    for path in paths:
        payload = json.loads(path.read_text())
        if (
            payload.get("format") != "flashsim_nf.generated_evaluation"
            or payload.get("format_version") != 3
        ):
            raise ValueError(
                f"Unsupported evaluation metric contract (need version 3): {path}"
            )
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete evaluation: {path}")
        if payload.get("reference_split") != "validation":
            raise ValueError("Winner selection may use validation evaluations only")
        if payload.get("selection_allowed") is False:
            raise ValueError(
                "Refusing a diagnostic evaluation in winner selection "
                f"(generation_purpose={payload.get('generation_purpose')!r}): {path}. "
                "Samples generated under the all-clean guard carry validation and "
                "test information in their rejection boundary."
            )
        generation_manifest = json.loads(
            (path.parent.parent / "generation_manifest.json").read_text()
        )
        candidates.append(
            {
                "path": str(path),
                "dataset_id": payload["dataset_id"],
                "generated_root": payload["inputs"]["generated_root"],
                "training_run": generation_manifest["training_run"],
                "preprocessing": generation_manifest["preprocessing"],
                "metrics": selection_metrics(payload["evaluation"]),
            }
        )
    if len({item["dataset_id"] for item in candidates}) != 1:
        raise ValueError("All candidates must belong to one dataset")
    metric_names = list(candidates[0]["metrics"])
    for name in metric_names:
        order = sorted(
            range(len(candidates)), key=lambda index: candidates[index]["metrics"][name]
        )
        for rank, index in enumerate(order, start=1):
            candidates[index].setdefault("ranks", {})[name] = rank
    for candidate in candidates:
        candidate["rank_sum"] = sum(candidate["ranks"].values())
    winner = min(candidates, key=lambda value: (value["rank_sum"], value["path"]))
    artifact = {
        "status": "frozen",
        "selection_data": "generated_validation_only",
        "test_data_used": False,
        "dataset_id": winner["dataset_id"],
        "method": "equal_weight_rank_sum_lower_is_better",
        "metrics": metric_names,
        "candidates": candidates,
        "winner": winner,
    }
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    write_json_atomic(destination, artifact)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
