#!/usr/bin/env python3
"""Run a read-only full-data preflight for one resolved Model 4 config."""

from __future__ import annotations

import argparse
import json
import sys

from flashsim_nf.training import PreflightError, preflight_model4_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        report = preflight_model4_training(args.config)
    except (
        PreflightError,
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "fail",
                    "training_allowed": False,
                    "config": args.config,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
