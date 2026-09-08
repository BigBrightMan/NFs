from __future__ import annotations

import csv
import json
from pathlib import Path

from flashsim_nf.manifest import RunManifest, build_run_index, write_run_manifest


def test_manifest_and_index_round_trip(tmp_path: Path) -> None:
    run = tmp_path / "campaigns" / "dataset" / "model4" / "run"
    manifest = RunManifest(
        run_id="fs25-m4-A-lr_hi-s42-a31f48c2",
        display_name="High learning rate",
        status="completed",
        dataset_id="fluka2025_muons_horizontal",
        year=2025,
        preprocessing="A",
        trial_id="lr_hi",
        training_seed=42,
        config_hash="a31f48c2",
        resolved_config="config.yaml",
        source_commit="9df6e2d",
        result={"best_epoch": 134},
    )
    write_run_manifest(run / "manifest.json", manifest)

    payload = json.loads((run / "manifest.json").read_text())
    assert payload["run_id"] == manifest.run_id
    assert payload["result"] == {"best_epoch": 134}

    destination = tmp_path / "registry" / "runs.csv"
    rows = build_run_index(tmp_path, destination)
    assert len(rows) == 1
    with destination.open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert csv_rows[0]["config_hash"] == "a31f48c2"
