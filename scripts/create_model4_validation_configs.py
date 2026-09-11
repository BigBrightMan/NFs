#!/usr/bin/env python3
"""Resolve generated-validation configs for completed baseline runs.

With `--purpose envelope_diagnostic` the same runs are resolved against
`guard_all_ref.json` instead, producing a diagnostic sample bounded by the
observed envelope of all clean FLUKA splits. That sample measures how much the
envelope choice is worth; it carries validation and test information in its
rejection boundary and is refused by winner selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from flashsim_nf.generation import resolve_generation_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", default="/eos/user/t/tanansub/SWAN_projects/NFs_output"
    )
    parser.add_argument(
        "--guard-root", default="/eos/user/t/tanansub/SWAN_projects/NFs_data/guards"
    )
    parser.add_argument("--guard-version-directory", default="train_vs_all_clean_v4")
    parser.add_argument("--expected-count", type=int, default=12)
    parser.add_argument(
        "--stage", choices=("smoke", "production"), default="production"
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=("drop_ze", "drop_z_pz", "none"),
        default=["drop_ze"],
    )
    parser.add_argument(
        "--pipelines", nargs="+", choices=("A", "B", "C", "D", "E")
    )
    parser.add_argument("--generation-seed", type=int, default=1556)
    parser.add_argument(
        "--dataset-ids",
        nargs="+",
        help=(
            "Restrict to these campaigns. Useful for the envelope diagnostic: "
            "where the train and all-clean observed envelopes are identical, the "
            "diagnostic run reproduces the validation run exactly and carries no "
            "information."
        ),
    )
    parser.add_argument(
        "--purpose",
        choices=("validation", "envelope_diagnostic"),
        default="validation",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    successes = []
    for ablation in args.ablations:
        successes.extend(
            output_root.glob(
                f"campaigns/*/model4/preprocessing_*/{ablation}/{args.stage}/base/"
                "config_*/train_seed_42/_SUCCESS.json"
            )
        )
    successes = sorted(set(successes))
    records = []
    identities = set()
    candidates = []
    for success in successes:
        run = success.parent
        training = json.loads((run / "config_resolved.json").read_text())
        dataset_id = training["dataset"]["dataset_id"]
        pipeline = training["preprocessing"]["id"]
        ablation = training["model"]["ablation"]
        if args.pipelines and pipeline not in args.pipelines:
            continue
        if args.dataset_ids and dataset_id not in args.dataset_ids:
            continue
        identity = (dataset_id, pipeline, ablation)
        if identity in identities:
            raise RuntimeError(f"Duplicate completed baseline: {identity}")
        identities.add(identity)
        guard_name = (
            "guard_train_ref.json"
            if args.purpose == "validation"
            else "guard_all_ref.json"
        )
        guard = (
            Path(args.guard_root)
            / dataset_id
            / args.guard_version_directory
            / guard_name
        )
        candidates.append((run, dataset_id, pipeline, ablation, guard))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "purpose": args.purpose,
                    "would_read": [str(guard) for *_, guard in candidates],
                    "runs": [str(run) for run, *_ in candidates],
                },
                indent=2,
            )
        )
        return
    missing_guards = [
        str(guard) for _, _, _, _, guard in candidates if not guard.is_file()
    ]
    if missing_guards:
        formatted = "\n".join(f"- {path}" for path in missing_guards)
        raise FileNotFoundError(
            "Generated-validation guard preflight failed. Missing train-reference "
            f"artifacts:\n{formatted}"
        )
    if len(candidates) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} runs after filters, "
            f"found {len(candidates)}"
        )
    for run, dataset_id, pipeline, ablation, guard in candidates:
        config = resolve_generation_config(
            run_directory=run,
            purpose=args.purpose,
            guard_artifact=guard,
            generation_seed=args.generation_seed,
        )
        destination_root = output_root / (
            "resolved_configs/model4_generated_validation_guard_v4"
            if args.purpose == "validation"
            else "resolved_configs/model4_envelope_diagnostic_all_guard"
        )
        destination = destination_root / dataset_id
        if args.stage != "production":
            destination = destination / args.stage
        if ablation != "drop_ze":
            destination = destination / ablation
        destination = destination / (
            f"preprocess_{pipeline}_genseed{args.generation_seed}.yaml"
        )
        serialized = yaml.safe_dump(config, sort_keys=False)
        if destination.exists():
            if yaml.safe_load(destination.read_text()) != config:
                raise FileExistsError(
                    f"Existing config has different content: {destination}"
                )
            status = "verified_existing"
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(serialized)
            status = "created"
        records.append(
            {
                "dataset_id": dataset_id,
                "preprocessing": pipeline,
                "ablation": ablation,
                "run_directory": str(run),
                "config": str(destination),
                "status": status,
            }
        )
    print(json.dumps({"count": len(records), "configs": records}, indent=2))


if __name__ == "__main__":
    main()
