#!/usr/bin/env python3
"""Submit an exact resolved generated-validation config matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configs = sorted(Path(args.config_root).glob("**/*.yaml"))
    if len(configs) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} configs, found {len(configs)}"
        )
    submitter = Path(__file__).with_name("submit_model4_post_training.py")
    for config in configs:
        command = [sys.executable, str(submitter), "--config", str(config)]
        if args.dry_run:
            command.append("--dry-run")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
