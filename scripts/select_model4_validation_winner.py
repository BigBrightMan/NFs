#!/usr/bin/env python3
"""Freeze one winner using generated-validation metrics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from flashsim_nf.manifest import write_json_atomic


def _score(payload: dict[str, Any]) -> dict[str, float]:
    evaluation = payload["evaluation"]
    global_metrics = evaluation["global_multivariate"]
    marginal = evaluation["marginal"]
    return {
        "frechet_gaussian_distance": float(global_metrics["frechet_gaussian_distance"]),
        "energy_distance": float(
            global_metrics["energy_distance"]["mean_clipped_nonnegative"]
        ),
        "sliced_wasserstein": float(global_metrics["sliced_wasserstein"]["mean"]),
        "mean_weighted_ks": float(
            np.mean([values["weighted_ks"] for values in marginal.values()])
        ),
        "mean_normalized_weighted_wasserstein": float(
            np.mean(
                [
                    values["normalized_weighted_wasserstein"]
                    for values in marginal.values()
                ]
            )
        ),
        "c2st_distance_from_half": float(
            global_metrics["c2st"]["distance_from_ideal_half"]
        ),
        "pearson_mean_absolute_difference": float(
            evaluation["correlations"]["pearson"]["mean_absolute_difference"]
        ),
        "spearman_mean_absolute_difference": float(
            evaluation["correlations"]["spearman"]["mean_absolute_difference"]
        ),
    }


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
            or payload.get("format_version") != 2
        ):
            raise ValueError(
                f"Unsupported evaluation metric contract (need version 2): {path}"
            )
        if payload.get("status") != "complete":
            raise ValueError(f"Incomplete evaluation: {path}")
        if payload.get("reference_split") != "validation":
            raise ValueError("Winner selection may use validation evaluations only")
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
                "metrics": _score(payload),
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
