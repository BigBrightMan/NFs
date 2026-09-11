#!/usr/bin/env python3
"""Compare one Model 4 sample generated under the train guard with the all-clean guard.

Both runs use the same training checkpoint, the same generation seed, and the same
validation reference. The only difference is which observed FLUKA envelope rejected
proposals: the train-fitted envelope (selection-legal) or the all-clean
train+validation+test envelope (diagnostic only).

The question this answers is empirical rather than philosophical. The all-clean
envelope is the widest support FLUKA ever produced and will never change, so it is
tempting to treat it as ground truth. It also contains validation and test
information, so using it to shape generation would void the held-out-test claim.
This script measures how much that choice is actually worth, so the decision can be
made against numbers.

The train-guard evaluation stays the selection artifact. The all-guard evaluation is
reported here only, and `select_model4_validation_winner.py` refuses it by design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flashsim_nf.evaluation.quality import selection_metrics

# The selection scalars come from the shared definition so this comparison cannot
# drift from what winner selection actually ranks on. `energy_distance` is reported
# but flagged: its raw estimate is noise-dominated (|mean| < std in every baseline
# run) and the clipped form turns a negative estimate into an apparent perfect score.
NOISE_DOMINATED_METRICS = frozenset({"energy_distance"})

EXTRA_GLOBAL_METRICS = ("covariance_frobenius_distance", "c2st_roc_auc")


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _flat_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    """Selection scalars plus two global descriptors not used in the ranking."""

    values = dict(selection_metrics(evaluation))
    glob = evaluation["global_multivariate"]
    values["covariance_frobenius_distance"] = float(
        glob["covariance_frobenius_distance"]
    )
    values["c2st_roc_auc"] = float(glob["c2st"]["roc_auc"])
    return values


def _relative_change(train: float, allclean: float) -> float | None:
    if train == 0.0:
        return None
    return (allclean - train) / abs(train)


def _rejection(manifest: dict[str, Any]) -> dict[str, Any]:
    rejection = manifest.get("rejection", {})
    reasons = rejection.get("reason_counts_nonexclusive", {})
    return {
        "attempted_rows": rejection.get("attempted_rows"),
        "accepted_rows": rejection.get("accepted_rows"),
        "rejected_rows": rejection.get("rejected_rows"),
        "rejection_fraction": rejection.get("rejection_fraction"),
        "envelope_reason_counts": {
            name: count
            for name, count in reasons.items()
            if name.startswith(("production_empirical_", "hard_support_"))
        },
        "exact_reason_counts": {
            name: reasons.get(name, 0)
            for name in ("nonfinite", "E_le_10", "pz_le_0")
        },
        "maximum_generated_energy_gev": manifest.get("physical_contract", {}).get(
            "maximum_generated_energy_gev"
        ),
        "maximum_generated_pz_gev": manifest.get("physical_contract", {}).get(
            "maximum_generated_pz_gev"
        ),
    }


def _envelope_bounds(manifest: dict[str, Any]) -> dict[str, dict[str, float]]:
    excursion = manifest.get("empirical_envelope_excursion", {})
    return {
        name: {
            "lower": entry.get("envelope_lower"),
            "upper": entry.get("envelope_upper"),
        }
        for name, entry in (excursion.get("features") or {}).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-guard-evaluation",
        required=True,
        help="generated_evaluation.json from the purpose=validation run",
    )
    parser.add_argument(
        "--all-guard-evaluation",
        required=True,
        help="generated_evaluation.json from the purpose=envelope_diagnostic run",
    )
    parser.add_argument("--output", help="Write the comparison JSON here")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.train_guard_evaluation).resolve()
    all_path = Path(args.all_guard_evaluation).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "would_read": [str(train_path), str(all_path)],
                    "would_write": str(Path(args.output).resolve())
                    if args.output
                    else None,
                },
                indent=2,
            )
        )
        return

    train_report = _read(train_path)
    all_report = _read(all_path)
    if train_report.get("dataset_id") != all_report.get("dataset_id"):
        raise ValueError("The two evaluations belong to different datasets")
    if train_report.get("selection_allowed") is False:
        raise ValueError("--train-guard-evaluation must be the selection-legal run")
    if all_report.get("selection_allowed") is not False:
        raise ValueError(
            "--all-guard-evaluation must come from an envelope_diagnostic run"
        )

    train_manifest = _read(Path(train_report["inputs"]["generated_root"]).parent
                           / "generation_manifest.json")
    all_manifest = _read(Path(all_report["inputs"]["generated_root"]).parent
                         / "generation_manifest.json")

    train_metrics = _flat_metrics(train_report["evaluation"])
    all_metrics = _flat_metrics(all_report["evaluation"])
    metric_rows = []
    for name in sorted(set(train_metrics) & set(all_metrics)):
        metric_rows.append(
            {
                "metric": name,
                "train_guard": train_metrics[name],
                "all_guard": all_metrics[name],
                "absolute_change": all_metrics[name] - train_metrics[name],
                "relative_change": _relative_change(
                    train_metrics[name], all_metrics[name]
                ),
                "noise_dominated": name in NOISE_DOMINATED_METRICS,
            }
        )

    largest = max(
        (
            row
            for row in metric_rows
            if row["relative_change"] is not None and not row["noise_dominated"]
        ),
        key=lambda row: abs(row["relative_change"]),
        default=None,
    )
    comparison = {
        "format": "flashsim_nf.guard_envelope_comparison",
        "format_version": 1,
        "dataset_id": train_report["dataset_id"],
        "reference_split": train_report.get("reference_split"),
        "note": (
            "The all-clean envelope contains validation and test rows. These numbers "
            "are a diagnostic; the train-guard run remains the selection artifact."
        ),
        "inputs": {
            "train_guard_evaluation": str(train_path),
            "all_guard_evaluation": str(all_path),
        },
        "generation": {
            "train_guard_seed": train_manifest.get("generation", {}).get("seed"),
            "all_guard_seed": all_manifest.get("generation", {}).get("seed"),
            "seeds_match": train_manifest.get("generation", {}).get("seed")
            == all_manifest.get("generation", {}).get("seed"),
            "preprocessing": train_manifest.get("preprocessing"),
            "ablation": train_manifest.get("ablation"),
        },
        "rejection": {
            "train_guard": _rejection(train_manifest),
            "all_guard": _rejection(all_manifest),
        },
        "envelope_bounds": {
            "train_guard": _envelope_bounds(train_manifest),
            "all_guard": _envelope_bounds(all_manifest),
        },
        "metrics": metric_rows,
        "largest_relative_change": largest,
    }

    if args.output:
        destination = Path(args.output).resolve()
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(comparison, indent=2, sort_keys=True))

    width = max(len(row["metric"]) for row in metric_rows) if metric_rows else 10
    print(f"{'metric':<{width}}{'train guard':>16}{'all guard':>16}{'relative':>12}")
    for row in metric_rows:
        relative = row["relative_change"]
        shown = "n/a" if relative is None else f"{relative:+.2%}"
        flag = "  (noise-dominated)" if row["noise_dominated"] else ""
        print(
            f"{row['metric']:<{width}}{row['train_guard']:>16.6g}"
            f"{row['all_guard']:>16.6g}{shown:>12}{flag}"
        )
    if largest is not None:
        print(
            f"\nlargest relative change (excluding noise-dominated): {largest['metric']} "
            f"{largest['relative_change']:+.2%}"
        )
    print(json.dumps({"output": args.output, "test_data_used": False}, indent=2))


if __name__ == "__main__":
    main()
