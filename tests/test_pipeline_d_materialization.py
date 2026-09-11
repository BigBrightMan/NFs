from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import uproot
import yaml

from flashsim_nf.experiments import (
    resolve_model4_training_config,
    write_resolved_training_config,
)
from flashsim_nf.preprocessing import FeaturePreprocessor, prepare_native_pipeline
from flashsim_nf.training import preflight_model4_training

PROJECT = Path(__file__).resolve().parents[1]


def _write_raw(path: Path, rows: int) -> None:
    index = np.arange(rows)
    arrays = {
        "run": np.ones(rows, dtype=np.int32),
        "event": index.astype(np.int64),
        "id": np.full(rows, 13, dtype=np.int32),
        "generation": np.ones(rows, dtype=np.int32),
        "x": -100.0 + index,
        "y": -50.0 + 2.0 * index,
        "z": 44000.0 + index,
        "E": 20.0 + index,
        "pz": 15.0 + index,
        "px": -2.0 + 0.1 * index,
        "py": -1.0 + 0.2 * index,
        "t": 1000.0 + 0.01 * index,
        "w": 0.01 + 0.001 * index,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as output:
        output["nt"] = arrays


@pytest.mark.parametrize("pipeline", ["D", "E"])
def test_native_pipeline_materializes_train_validation_only(
    tmp_path: Path, pipeline: str
) -> None:
    prepared = tmp_path / "legacy_prepared"
    _write_raw(prepared / "split/train_rawfeature.root", 20)
    _write_raw(prepared / "split/validation_rawfeature.root", 10)
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "fluka2025_test",
                "year": 2025,
                "selected_rows": 35,
                "dataset_fingerprint": "fingerprint",
                "split": {
                    "id": "split",
                    "counts": {"train": 20, "validation": 10, "test": 5},
                },
                "legacy_prepared_paths": {
                    "local": str(prepared),
                    "cern": str(prepared),
                },
            }
        )
    )
    (prepared / "split/split_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "fluka2025_test",
                "dataset_fingerprint": "fingerprint",
                "split_id": "split",
                "counts": {"train": 20, "validation": 10, "test": 5},
            }
        )
    )
    report = prepare_native_pipeline(
        pipeline=pipeline,
        dataset_config=dataset,
        environment="local",
        data_root=tmp_path / "NFs_data",
    )
    output = Path(report["output_directory"])
    assert (output / "8d/train_preprocessed.root").is_file()
    assert (output / "8d/validation_preprocessed.root").is_file()
    assert not (output / "8d/test_preprocessed.root").exists()
    assert (
        json.loads((output / "preparation_manifest.json").read_text())["test_loaded"]
        is False
    )
    artifact = FeaturePreprocessor.load(output / "preprocessor.json")
    assert artifact.name == pipeline
    assert artifact.feature_order[-1] == "w"

    resolved = resolve_model4_training_config(
        project_root=PROJECT,
        dataset_config=dataset,
        prepared_directory=prepared,
        preprocessing_directory=output,
        output_root=tmp_path / "NFs_output",
        preprocessing=pipeline,
        ablation="drop_z_pz",
        stage="smoke",
    )
    resolved_path = tmp_path / "resolved.yaml"
    write_resolved_training_config(resolved_path, resolved)
    preflight = preflight_model4_training(resolved_path)
    assert preflight["training_allowed"] is True
    assert preflight["model"]["input_dim"] == 6
    assert preflight["test_loaded"] is False
