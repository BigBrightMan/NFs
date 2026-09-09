#!/usr/bin/env python3
"""Resolve generated-validation configs for completed baseline runs."""

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
    parser.add_argument("--expected-count", type=int, default=12)
    parser.add_argument("--generation-seed", type=int, default=1556)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    successes = sorted(
        output_root.glob(
            "campaigns/*/model4/preprocessing_*/drop_ze/production/base/"
            "config_*/train_seed_42/_SUCCESS.json"
        )
    )
    if len(successes) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} completed baseline runs, "
            f"found {len(successes)}"
        )
    records = []
    identities = set()
    candidates = []
    for success in successes:
        run = success.parent
        training = json.loads((run / "config_resolved.json").read_text())
        dataset_id = training["dataset"]["dataset_id"]
        pipeline = training["preprocessing"]["id"]
        identity = (dataset_id, pipeline)
        if identity in identities:
            raise RuntimeError(f"Duplicate completed baseline: {identity}")
        identities.add(identity)
        guard = (
            Path(args.guard_root)
            / dataset_id
            / "train_vs_all_clean_v2"
            / "guard_train_ref.json"
        )
        candidates.append((run, dataset_id, pipeline, guard))
    missing_guards = [
        str(guard) for _, _, _, guard in candidates if not guard.is_file()
    ]
    if missing_guards:
        formatted = "\n".join(f"- {path}" for path in missing_guards)
        raise FileNotFoundError(
            "Generated-validation guard preflight failed. Missing train-reference "
            f"artifacts:\n{formatted}"
        )
    for run, dataset_id, pipeline, guard in candidates:
        config = resolve_generation_config(
            run_directory=run,
            purpose="validation",
            guard_artifact=guard,
            generation_seed=args.generation_seed,
        )
        destination = (
            output_root
            / "resolved_configs/model4_generated_validation"
            / dataset_id
            / f"preprocess_{pipeline}_genseed{args.generation_seed}.yaml"
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
                "run_directory": str(run),
                "config": str(destination),
                "status": status,
            }
        )
    print(json.dumps({"count": len(records), "configs": records}, indent=2))


if __name__ == "__main__":
    main()
