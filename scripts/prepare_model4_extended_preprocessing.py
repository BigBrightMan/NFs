#!/usr/bin/env python3
"""Fit native extended-pipeline artifacts for maintained datasets.

Pipeline D covers both extended experiments: it is Pipeline A with `E` and `pz`
both on natural log, and Model 4 trains on only one of that pair per ablation.
Pipeline E remains selectable but is redundant with D under `drop_z_pz`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashsim_nf.config import load_yaml
from flashsim_nf.preprocessing import prepare_native_pipeline


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=("local", "cern"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--pipelines", nargs="+", choices=("D", "E"), default=["D"]
    )
    parser.add_argument("--dataset-ids", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    selected = set(args.dataset_ids or ())
    configs = sorted((project / "configs/datasets").glob("*.yaml"))
    if selected:
        configs = [
            path for path in configs if load_yaml(path)["dataset_id"] in selected
        ]
    reports = []
    for pipeline in args.pipelines:
        for path in configs:
            reports.append(
                prepare_native_pipeline(
                    pipeline=pipeline,
                    dataset_config=path,
                    environment=args.environment,
                    data_root=args.data_root,
                    overwrite=args.overwrite,
                )
            )
    print(json.dumps({"number_artifacts": len(reports), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
