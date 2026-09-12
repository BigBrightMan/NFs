#!/usr/bin/env python3
"""Resolve MuonDIS delivery configs for the frozen winner of each campaign.

One config per campaign, built from that campaign's frozen `selected_model.json`.
`resolve_generation_config` refuses any run that is not the recorded winner, so this
cannot ship a candidate that validation did not select.

Delivery uses `guard_all_ref.json`, the envelope fitted on all clean FLUKA splits.
That envelope contains validation and test rows, which is why it is marked
`production_only` with `selection_allowed: false` and is permitted only after the
model is frozen. Nothing produced here may re-enter model selection or a test claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from flashsim_nf.config import load_yaml
from flashsim_nf.generation import resolve_generation_config


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", default="/eos/user/t/tanansub/SWAN_projects/NFs_output"
    )
    parser.add_argument(
        "--guard-root", default="/eos/user/t/tanansub/SWAN_projects/NFs_data/guards"
    )
    parser.add_argument("--guard-version-directory", default="train_vs_all_clean_v5")
    parser.add_argument("--number-events", type=int, default=100_000)
    parser.add_argument("--generation-seed", type=int, default=3556)
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--expected-count", type=int, default=4)
    parser.add_argument(
        "--config-output-directory",
        help="Defaults to <output-root>/resolved_configs/model4_muondis_delivery",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.number_events <= 0:
        raise ValueError("--number-events must be positive")
    output_root = Path(args.output_root).resolve()
    destination_root = (
        Path(args.config_output_directory).resolve()
        if args.config_output_directory
        else output_root / "resolved_configs/model4_muondis_delivery"
    )

    candidates = []
    for dataset_config in sorted((project / "configs/datasets").glob("fluka*.yaml")):
        dataset_id = str(load_yaml(dataset_config)["dataset_id"])
        if args.dataset_ids and dataset_id not in args.dataset_ids:
            continue
        frozen = (
            output_root
            / "campaigns"
            / dataset_id
            / "model4"
            / "model_selection"
            / "validation"
            / "selected_model.json"
        )
        if not frozen.is_file():
            raise FileNotFoundError(f"No frozen selection for {dataset_id}: {frozen}")
        selection = json.loads(frozen.read_text())
        if selection.get("status") != "frozen":
            raise ValueError(f"{dataset_id}: selection is not frozen")
        if selection.get("test_data_used") is not False:
            raise ValueError(f"{dataset_id}: selection claims test data was used")
        winner = selection["winner"]
        guard = (
            Path(args.guard_root)
            / dataset_id
            / args.guard_version_directory
            / "guard_all_ref.json"
        )
        candidates.append(
            {
                "dataset_id": dataset_id,
                "pipeline": winner["preprocessing"],
                "run": Path(winner["training_run"]),
                "frozen": frozen,
                "guard": guard,
            }
        )

    missing = [
        str(item["guard"]) for item in candidates if not item["guard"].is_file()
    ]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "MuonDIS delivery requires the all-clean FLUKA guard. Missing:\n"
            f"{formatted}"
        )
    if len(candidates) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} campaigns, found {len(candidates)}"
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "number_events": args.number_events,
                    "generation_seed": args.generation_seed,
                    "plans": [
                        {
                            "dataset_id": item["dataset_id"],
                            "pipeline": item["pipeline"],
                            "run": str(item["run"]),
                            "guard": str(item["guard"]),
                            "frozen_selection": str(item["frozen"]),
                        }
                        for item in candidates
                    ],
                },
                indent=2,
            )
        )
        return

    records = []
    for item in candidates:
        config = resolve_generation_config(
            run_directory=item["run"],
            purpose="muondis",
            guard_artifact=item["guard"],
            number_events=args.number_events,
            generation_seed=args.generation_seed,
            frozen_selection=item["frozen"],
        )
        destination = destination_root / item["dataset_id"] / (
            f"preprocess_{item['pipeline']}_muondis"
            f"_n{args.number_events}_genseed{args.generation_seed}.yaml"
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
                "dataset_id": item["dataset_id"],
                "preprocessing": item["pipeline"],
                "number_events": args.number_events,
                "config": str(destination),
                "output_directory": config["output"]["directory"],
                "status": status,
            }
        )

    print(
        json.dumps(
            {
                "count": len(records),
                "purpose": "muondis",
                "guard": "guard_all_ref (all clean FLUKA splits)",
                "selection_allowed": False,
                "configs": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
