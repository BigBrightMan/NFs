from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import uproot
import yaml

from flashsim_nf.training.model4 import train_model4_from_config


def test_example_drop_z_e_feature_contract() -> None:
    project = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (project / "configs/jobs/model4_training.example.yaml").read_text()
    )
    assert config["model"]["ablation"] == "drop_ze"
    assert config["data"]["feature_order"] == ["x", "y", "pz", "px", "py", "t"]
    assert "w" not in config["data"]["feature_order"]


def _write_root(path: Path, *, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    index = np.arange(rows, dtype=np.int64)
    with uproot.recreate(path) as destination:
        destination["nt"] = {
            "run": np.ones(rows, dtype=np.int32),
            "event": index,
            "id": np.full(rows, 13, dtype=np.int32),
            "generation": np.ones(rows, dtype=np.int32),
            "x": rng.normal(size=rows),
            "y": rng.normal(size=rows),
            "w": rng.uniform(0.01, 0.2, size=rows),
        }


def _config(tmp_path: Path, run: Path, *, maximum_epochs: int) -> Path:
    train = tmp_path / "train.root"
    validation = tmp_path / "validation.root"
    _write_root(train, rows=16, seed=1)
    _write_root(validation, rows=8, seed=2)
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "fluka_test",
                "split_id": "frozen",
                "dataset_fingerprint": "fingerprint",
                "counts": {"train": 16, "validation": 8, "test": 8},
            }
        )
    )
    preprocessor = tmp_path / "preprocessor.json"
    preprocessor.write_text("{}\n")
    digest = hashlib.sha256(preprocessor.read_bytes()).hexdigest()
    config = {
        "dataset": {
            "dataset_id": "fluka_test",
            "split_id": "frozen",
            "dataset_fingerprint": "fingerprint",
        },
        "preprocessing": {
            "id": "A",
            "fit_split": "train",
            "artifact": str(preprocessor),
            "artifact_sha256": digest,
        },
        "data": {
            "tree_name": "nt",
            "feature_order": ["x", "y"],
            "split_manifest_path": str(manifest),
            "train": {
                "model_space_path": str(train),
                "raw_weight_path": str(train),
                "expected_rows": 16,
            },
            "validation": {
                "model_space_path": str(validation),
                "raw_weight_path": str(validation),
                "expected_rows": 8,
            },
        },
        "model": {
            "family": "model4",
            "architecture": "rq_spline",
            "ablation": "drop_ze",
            "objective": "weighted_nll",
            "weight_is_input_feature": False,
            "num_transforms": 1,
            "hidden_features": 8,
            "num_blocks": 1,
            "num_bins": 4,
        },
        "training": {
            "learning_rate": 0.001,
            "batch_size": 8,
            "maximum_epochs": maximum_epochs,
            "checkpoint_interval": 1,
            "early_stopping": {"enabled": False, "patience": 2},
            "device": "cpu",
            "seed": 42,
        },
        "output": {"run_directory": str(run)},
    }
    path = tmp_path / f"config_{run.name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def test_model4_training_and_full_state_resume_vertical_slice(tmp_path: Path) -> None:
    first_run = tmp_path / "first_run"
    first_config = _config(tmp_path, first_run, maximum_epochs=2)
    first = train_model4_from_config(first_config)
    assert first["first_epoch"] == 1
    assert first["last_epoch"] == 2
    assert first["best_epoch"] in {1, 2}
    checkpoint = first_run / "checkpoints/epoch_0002_model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["optimizer_state_dict"]
    assert payload["scheduler_state_dict"]
    assert payload["data_loader_generator_state"] is not None
    resolved = json.loads((first_run / "config_resolved.json").read_text())
    train_weights = resolved["training_provenance"]["train"]["weight_summary"]
    validation_weights = resolved["training_provenance"]["validation"][
        "weight_summary"
    ]
    assert train_weights["rows"] == 16
    assert validation_weights["rows"] == 8
    assert 0 < train_weights["effective_sample_size"] <= 16
    assert 0 < validation_weights["effective_sample_size"] <= 8

    resumed_run = tmp_path / "resumed_run"
    resumed_config = _config(tmp_path, resumed_run, maximum_epochs=3)
    config = yaml.safe_load(resumed_config.read_text())
    config["resume"] = {
        "mode": "branch",
        "source_checkpoint": str(checkpoint),
        "expected_epoch": 2,
    }
    resumed_config.write_text(yaml.safe_dump(config, sort_keys=False))
    resumed = train_model4_from_config(resumed_config)
    assert resumed["first_epoch"] == 3
    assert resumed["last_epoch"] == 3
    assert (resumed_run / "checkpoints/last_model.pt").is_file()
    assert (first_run / "checkpoints/last_model.pt").is_file()
