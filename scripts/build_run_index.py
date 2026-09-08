#!/usr/bin/env python3
"""Build the searchable CSV run index from run-local manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from flashsim_nf.manifest import build_run_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    destination = args.destination or args.output_root / "registry" / "runs.csv"
    rows = build_run_index(args.output_root, destination)
    print(f"runs={len(rows)}")
    print(f"index={destination}")


if __name__ == "__main__":
    main()
