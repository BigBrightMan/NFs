#!/usr/bin/env python3
"""Small operator CLI for Model 4 generated-validation and selection."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from flashsim_nf.pipeline import execute_post_training_config


REPOSITORY = Path(__file__).resolve().parents[1]
ALL_YEARS = (2022, 2023, 2024, 2025)
ALL_PIPELINES = ("A", "B", "C")


def _default_output_root() -> Path:
    configured = os.environ.get("NFS_OUTPUT_ROOT")
    if configured:
        return Path(configured).resolve()
    cern = Path("/eos/user/t/tanansub/SWAN_projects/NFs_output")
    if cern.parent.exists():
        return cern
    return (REPOSITORY.parent / "NFs_output").resolve()


def _dataset_id(year: int) -> str:
    campaign = yaml.safe_load(
        (REPOSITORY / "configs" / "campaigns" / f"{year}.yaml").read_text()
    )
    dataset_path = REPOSITORY / campaign["dataset"]
    dataset = yaml.safe_load(dataset_path.read_text())
    return str(dataset["dataset_id"])


def _generation_config(
    *, output_root: Path, year: int, pipeline: str
) -> Path:
    dataset_id = _dataset_id(year)
    return (
        output_root
        / "resolved_configs"
        / "model4_generated_validation_guard_v3"
        / dataset_id
        / f"preprocess_{pipeline}_genseed1556.yaml"
    )


def _stage(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {"state": "missing-config", "config": config_path}
    config = yaml.safe_load(config_path.read_text())
    output = Path(config["output"]["directory"])
    generation = (output / "_SUCCESS.json").is_file()
    evaluation = (output / "evaluation" / "_SUCCESS.json").is_file()
    summary = (output / "post_training_summary.json").is_file()
    if generation and evaluation and summary:
        state = "complete"
    elif output.exists():
        state = "partial"
    else:
        state = "pending"
    return {
        "state": state,
        "config": config_path,
        "output": output,
        "evaluation": output / "evaluation" / "generated_evaluation.json",
    }


def _selection_path(output_root: Path, year: int) -> Path:
    return (
        output_root
        / "campaigns"
        / _dataset_id(year)
        / "model4"
        / "model_selection"
        / "validation"
        / "selected_model.json"
    )


def _print_status(args: argparse.Namespace) -> int:
    for year in args.years:
        for pipeline in args.pipelines:
            stage = _stage(
                _generation_config(
                    output_root=args.output_root, year=year, pipeline=pipeline
                )
            )
            print(f"{year} {pipeline}: {stage['state']}")
        selection = _selection_path(args.output_root, year)
        print(f"{year} winner: {'frozen' if selection.is_file() else 'not-frozen'}")
    return 0


def _run_generated_validation(args: argparse.Namespace) -> int:
    failures = 0
    for year in args.years:
        for pipeline in args.pipelines:
            config = _generation_config(
                output_root=args.output_root, year=year, pipeline=pipeline
            )
            stage = _stage(config)
            label = f"{year} Pipeline {pipeline}"
            if stage["state"] == "complete":
                print(f"SKIP {label}: complete")
                continue
            if stage["state"] == "missing-config":
                print(f"FAIL {label}: missing config {config}", file=sys.stderr)
                failures += 1
                continue
            if stage["state"] == "partial":
                print(
                    f"FAIL {label}: incomplete immutable output {stage['output']}",
                    file=sys.stderr,
                )
                failures += 1
                continue
            print(f"{'WOULD RUN' if args.dry_run else 'RUN'} {label}: {config}")
            if args.dry_run:
                continue
            try:
                execute_post_training_config(config)
            except Exception as error:  # keep independent candidates moving
                print(f"FAIL {label}: {type(error).__name__}: {error}", file=sys.stderr)
                failures += 1
            else:
                print(f"PASS {label}")
    return int(failures > 0)


def _select(args: argparse.Namespace) -> int:
    failures = 0
    for year in args.years:
        stages = [
            _stage(
                _generation_config(
                    output_root=args.output_root, year=year, pipeline=pipeline
                )
            )
            for pipeline in ALL_PIPELINES
        ]
        if any(stage["state"] != "complete" for stage in stages):
            message = f"{year}: A/B/C generated-validation is not complete"
            if args.all_ready:
                print(f"SKIP {message}")
                continue
            print(f"FAIL {message}", file=sys.stderr)
            failures += 1
            continue
        destination = _selection_path(args.output_root, year)
        if destination.exists():
            print(f"SKIP {year}: winner already frozen at {destination}")
            continue
        command = [
            sys.executable,
            str(REPOSITORY / "scripts" / "select_model4_validation_winner.py"),
            "--evaluations",
            *(str(stage["evaluation"]) for stage in stages),
            "--output",
            str(destination),
        ]
        print(f"{'WOULD FREEZE' if args.dry_run else 'FREEZE'} {year}")
        if not args.dry_run:
            subprocess.run(command, cwd=REPOSITORY, check=True)
    return int(failures > 0)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--years", nargs="+", type=int, choices=ALL_YEARS)
    parser.add_argument(
        "--output-root", type=Path, default=_default_output_root()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    _common(status)
    status.add_argument("--pipelines", nargs="+", choices=ALL_PIPELINES)
    status.set_defaults(handler=_print_status)

    generated = subparsers.add_parser("generated-validation")
    _common(generated)
    generated.add_argument("--pipelines", nargs="+", choices=ALL_PIPELINES)
    generated.add_argument("--dry-run", action="store_true")
    generated.set_defaults(handler=_run_generated_validation)

    select = subparsers.add_parser("select")
    selection_scope = select.add_mutually_exclusive_group(required=True)
    selection_scope.add_argument("--years", nargs="+", type=int, choices=ALL_YEARS)
    selection_scope.add_argument("--all-ready", action="store_true")
    select.add_argument("--output-root", type=Path, default=_default_output_root())
    select.add_argument("--dry-run", action="store_true")
    select.set_defaults(handler=_select)

    args = parser.parse_args()
    if getattr(args, "years", None) is None:
        args.years = list(ALL_YEARS)
    if getattr(args, "pipelines", None) is None:
        args.pipelines = list(ALL_PIPELINES)
    raise SystemExit(args.handler(args))


if __name__ == "__main__":
    main()
