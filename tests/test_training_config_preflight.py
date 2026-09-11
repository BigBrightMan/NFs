from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import uproot
import yaml

from flashsim_nf.experiments import (
    resolve_model4_training_config,
    write_resolved_training_config,
)
from flashsim_nf.training import PreflightError, preflight_model4_training

PROJECT = Path(__file__).resolve().parents[1]
CANONICAL_8D = ["x", "y", "z", "E", "pz", "px", "py", "t"]
MODEL_FEATURES = ["x", "y", "pz", "px", "py", "t"]


def _write_root(path: Path, *, rows: int, include_weight: bool) -> None:
    index = np.arange(rows, dtype=np.int64)
    values: dict[str, np.ndarray] = {
        "run": np.ones(rows, dtype=np.int32),
        "event": index,
        "id": np.full(rows, 13, dtype=np.int32),
        "generation": np.ones(rows, dtype=np.int32),
    }
    for offset, feature in enumerate(MODEL_FEATURES, start=1):
        values[feature] = index.astype(np.float64) + offset
    if include_weight:
        values["w"] = np.linspace(0.01, 0.2, rows, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as destination:
        destination["nt"] = values


def _fixture(tmp_path: Path, *, year: int = 2025) -> tuple[Path, Path, Path]:
    dataset_id = f"fluka{year}_test"
    rows = {"train": 7, "validation": 5, "test": 3}
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": dataset_id,
                "year": year,
                "selected_rows": sum(rows.values()),
                "dataset_fingerprint": f"fingerprint-{year}",
                "split": {"id": f"split-{year}", "counts": rows},
            }
        )
    )
    prepared = tmp_path / "prepared"
    pipeline = prepared / "preprocessing_B"
    (pipeline / "8d").mkdir(parents=True)
    (pipeline / "preprocessor.joblib").write_bytes(b"immutable fitted artifact")
    (pipeline / "preprocessing_parameters.json").write_text(
        json.dumps(
            {
                "pipeline": "B",
                "fitted": True,
                "feature_order": [*CANONICAL_8D, "w"],
            }
        )
    )
    (pipeline / "8d/feature_order.json").write_text(json.dumps(CANONICAL_8D))
    for split in ("train", "validation"):
        _write_root(
            pipeline / f"8d/{split}_preprocessed.root",
            rows=rows[split],
            include_weight=False,
        )
        _write_root(
            prepared / f"split/{split}_rawfeature.root",
            rows=rows[split],
            include_weight=True,
        )
    (prepared / "split/split_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "split_id": f"split-{year}",
                "dataset_fingerprint": f"fingerprint-{year}",
                "counts": rows,
            }
        )
    )
    return dataset_path, prepared, tmp_path / "outputs"


def _resolve(tmp_path: Path) -> dict:
    dataset, prepared, outputs = _fixture(tmp_path)
    return resolve_model4_training_config(
        project_root=PROJECT,
        dataset_config=dataset,
        prepared_directory=prepared,
        output_root=outputs,
        preprocessing="B",
        stage="production",
    )


def test_resolver_uses_dataset_counts_and_excludes_test_input(tmp_path: Path) -> None:
    config = _resolve(tmp_path)
    assert config["data"]["train"]["expected_rows"] == 7
    assert config["data"]["validation"]["expected_rows"] == 5
    assert "test" not in config["data"]
    assert config["data"]["feature_order"] == MODEL_FEATURES
    assert config["training"]["maximum_epochs"] == 500
    assert config["training"]["checkpoint_interval"] == 50
    assert config["training"]["early_stopping"]["enabled"] is False
    assert not Path(config["output"]["run_directory"]).exists()


def test_resolver_is_stable_and_resolved_config_never_overwrites(
    tmp_path: Path,
) -> None:
    config = _resolve(tmp_path)
    repeated = resolve_model4_training_config(
        project_root=PROJECT,
        dataset_config=tmp_path / "dataset.yaml",
        prepared_directory=tmp_path / "prepared",
        output_root=tmp_path / "outputs",
        preprocessing="B",
        stage="production",
    )
    assert config["resolution"]["config_hash"] == repeated["resolution"]["config_hash"]
    destination = tmp_path / "resolved.yaml"
    write_resolved_training_config(destination, config)
    with pytest.raises(FileExistsError, match="already exists"):
        write_resolved_training_config(destination, config)


def test_smoke_stage_defaults_to_five_epochs(tmp_path: Path) -> None:
    dataset, prepared, outputs = _fixture(tmp_path)
    config = resolve_model4_training_config(
        project_root=PROJECT,
        dataset_config=dataset,
        prepared_directory=prepared,
        output_root=outputs,
        preprocessing="B",
        stage="smoke",
    )
    assert config["training"]["maximum_epochs"] == 5


@pytest.mark.parametrize(
    ("ablation", "expected"),
    [
        ("drop_z_pz", ["x", "y", "E", "px", "py", "t"]),
        ("none", CANONICAL_8D),
    ],
)
def test_resolver_supports_alternative_ablation_feature_orders(
    tmp_path: Path, ablation: str, expected: list[str]
) -> None:
    dataset, prepared, outputs = _fixture(tmp_path)
    config = resolve_model4_training_config(
        project_root=PROJECT,
        dataset_config=dataset,
        prepared_directory=prepared,
        output_root=outputs,
        preprocessing="B",
        ablation=ablation,
        stage="smoke",
    )
    assert config["model"]["ablation"] == ablation
    assert config["data"]["feature_order"] == expected


def test_2022_resolver_requires_tclean_identity(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "fluka2022_muons_down",
                "year": 2022,
                "selected_rows": 3,
                "dataset_fingerprint": "bad",
                "split": {
                    "id": "split",
                    "counts": {"train": 1, "validation": 1, "test": 1},
                },
            }
        )
    )
    with pytest.raises(ValueError, match="t-clean"):
        resolve_model4_training_config(
            project_root=PROJECT,
            dataset_config=dataset,
            prepared_directory=tmp_path / "prepared",
            output_root=tmp_path / "outputs",
            preprocessing="B",
            stage="production",
        )


def test_preflight_fully_scans_train_validation_and_is_read_only(
    tmp_path: Path,
) -> None:
    config = _resolve(tmp_path)
    config_path = tmp_path / "resolved.yaml"
    write_resolved_training_config(config_path, config)
    report = preflight_model4_training(config_path)
    assert report["training_allowed"] is True
    assert report["train"]["rows"] == 7
    assert report["validation"]["rows"] == 5
    assert report["test_loaded"] is False
    assert report["read_only"] is True
    assert not Path(config["output"]["run_directory"]).exists()


def test_preflight_rejects_test_input_and_existing_run(tmp_path: Path) -> None:
    config = _resolve(tmp_path)
    unsafe = deepcopy(config)
    unsafe["data"]["test"] = {"model_space_path": "/tmp/test.root"}
    unsafe_path = tmp_path / "unsafe.yaml"
    unsafe_path.write_text(yaml.safe_dump(unsafe, sort_keys=False))
    with pytest.raises(PreflightError, match="must not contain a test"):
        preflight_model4_training(unsafe_path)

    config_path = tmp_path / "resolved.yaml"
    write_resolved_training_config(config_path, config)
    Path(config["output"]["run_directory"]).mkdir(parents=True)
    with pytest.raises(PreflightError, match="already exists"):
        preflight_model4_training(config_path)


def test_preflight_rejects_changed_maintained_source(tmp_path: Path) -> None:
    config = _resolve(tmp_path)
    source = Path(config["resolution"]["sources"]["dataset"]["path"])
    source.write_text(source.read_text() + "\nstatus: changed\n")
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(PreflightError, match="SHA-256 mismatch"):
        preflight_model4_training(config_path)
