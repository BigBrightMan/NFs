#!/usr/bin/env python3
"""Create read-only diagnostics for one completed Model 4 training run."""

from __future__ import annotations

import argparse
import json

from flashsim_nf.evaluation import evaluate_training_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    report = evaluate_training_run(
        args.run_directory,
        output_directory=args.output_directory,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
