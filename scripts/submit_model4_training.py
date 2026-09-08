#!/usr/bin/env python3
"""Preflight and submit exactly one resolved Model 4 training job."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from flashsim_nf.config import load_yaml
from flashsim_nf.execution import CondorResources, build_condor_submission
from flashsim_nf.training import preflight_model4_training


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--activate",
        default="/eos/user/t/tanansub/venvBBfs/bin/activate",
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--memory", default="12GB")
    parser.add_argument("--disk", default="10GB")
    parser.add_argument("--maximum-runtime", type=int, default=604800)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = preflight_model4_training(args.config)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    submission = build_condor_submission(
        config_path=config_path,
        config=config,
        project_root=project,
        activate_path=args.activate,
        resources=CondorResources(
            gpus=args.gpus,
            cpus=args.cpus,
            memory=args.memory,
            disk=args.disk,
            maximum_runtime_seconds=args.maximum_runtime,
        ),
    )
    print(
        json.dumps(
            {
                "preflight_status": report["status"],
                "training_allowed": report["training_allowed"],
                "run_id": submission.run_name,
                "run_directory": str(submission.run_directory),
                "log_directory": str(submission.log_directory),
                "dry_run": args.dry_run,
            },
            indent=2,
        ),
        flush=True,
    )
    print(shlex.join(submission.command), flush=True)
    if args.dry_run:
        return
    if shutil.which("condor_submit") is None:
        raise RuntimeError("condor_submit is unavailable")
    submission.log_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(submission.command, cwd=project, check=True)


if __name__ == "__main__":
    main()
