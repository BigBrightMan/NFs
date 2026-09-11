#!/usr/bin/env python3
"""Fit and materialize Pipeline D for selected maintained datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashsim_nf.config import load_yaml
from flashsim_nf.preprocessing import prepare_pipeline_d


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("local", "cern"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    selected = set(args.dataset_ids or ())
    configs = sorted((project / "configs/datasets").glob("*.yaml"))
    if selected:
        configs = [
            path for path in configs if load_yaml(path)["dataset_id"] in selected
        ]
    reports = [
        prepare_pipeline_d(
            dataset_config=path,
            environment=args.environment,
            data_root=args.data_root,
            overwrite=args.overwrite,
        )
        for path in configs
    ]
    print(json.dumps({"number_datasets": len(reports), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
