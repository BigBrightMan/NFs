#!/usr/bin/env python3
"""Submit one resolved post-training config to the CERN EOS schedd."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--project-root", default="/eos/user/t/tanansub/SWAN_projects/NFs"
    )
    parser.add_argument(
        "--activate", default="/eos/user/t/tanansub/venvBBfs/bin/activate"
    )
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--memory", default="8GB")
    parser.add_argument("--disk", default="10GB")
    parser.add_argument("--maximum-runtime", type=int, default=86_400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise TypeError("Post-training config must contain a mapping")
    output = Path(config["output"]["directory"])
    if output.exists():
        if (
            not (output / "generation_manifest.json").is_file()
            or not (output / "_SUCCESS.json").is_file()
        ):
            raise RuntimeError("Existing generation directory is incomplete")
        if (output / "post_training_summary.json").is_file():
            raise FileExistsError(
                f"Post-training pipeline is already complete: {output}"
            )
    run = Path(config["training_run"])
    training = json.loads((run / "config_resolved.json").read_text())
    output_root = Path(training["resolution"]["output_root"])
    dataset_id = config["dataset"]["dataset_id"]
    purpose = config["generation"]["purpose"]
    seed = config["generation"]["seed"]
    run_name = f"{training['resolution']['run_id']}-{purpose}-g{seed}"
    log_subdir = f"campaigns/{dataset_id}/model4/post_training/{purpose}"
    project = Path(args.project_root).resolve()
    submit_file = project / "condor/submit_model4_post_training.sub"
    log_directory = output_root / "condor_logs" / log_subdir
    command = [
        "condor_submit",
        f"project={project}",
        f"output_root={output_root}",
        f"activate={Path(args.activate).resolve()}",
        f"config={config_path}",
        f"run_name={run_name}",
        "gpus=1",
        f"cpus={args.cpus}",
        f"memory={args.memory}",
        f"disk={args.disk}",
        f"maximum_runtime={args.maximum_runtime}",
        f"log_subdir={log_subdir}",
        str(submit_file),
    ]
    print(" ".join(command))
    if not args.dry_run:
        log_directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
