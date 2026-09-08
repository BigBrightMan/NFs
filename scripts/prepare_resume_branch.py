#!/usr/bin/env python3
"""Prepare a new, non-overwriting run branched from a checkpoint."""

from __future__ import annotations

import argparse
import json

from flashsim_nf.training import prepare_resume_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--new-run-directory", required=True)
    parser.add_argument("--expected-epoch", type=int, default=450)
    args = parser.parse_args()
    result = prepare_resume_directory(
        args.checkpoint,
        args.new_run_directory,
        expected_epoch=args.expected_epoch,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
