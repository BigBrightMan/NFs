#!/usr/bin/env python3
"""Compute one campaign's validation-only FLUKA reference noise floor."""

from __future__ import annotations

import argparse
import json

from flashsim_nf.evaluation import compute_reference_noise_floor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--row-matched-rows", type=int)
    parser.add_argument("--random-seed", type=int, default=91556)
    args = parser.parse_args()
    report = compute_reference_noise_floor(
        dataset_id=args.dataset_id,
        train_root=args.train_root,
        validation_root=args.validation_root,
        output=args.output,
        repeats=args.repeats,
        row_matched_rows=args.row_matched_rows,
        random_seed=args.random_seed,
    )
    print(json.dumps({"status": report["status"], "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
