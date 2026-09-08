"""Resolve one immutable, path-explicit Model 4 training configuration."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..config import config_hash, deep_merge, load_yaml
from ..naming import build_run_id, validate_slug

ABLATION_DROPS = {"drop_ze": {"z", "E"}}
CANONICAL_8D = ("x", "y", "z", "E", "pz", "px", "py", "t")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_directory(path: str | Path, *, name: str) -> Path:
    value = Path(path)
    if not value.is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return value.resolve()


def _source(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _trial(
    baseline: dict[str, Any],
    tuning_space: dict[str, Any],
    trial_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trial_id = validate_slug(trial_id, field="trial_id")
    matches = [
        value
        for value in tuning_space.get("trials", [])
        if value.get("trial_id") == trial_id and value.get("enabled", True)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one enabled tuning trial {trial_id!r}")
    trial = matches[0]
    overrides = trial.get("overrides", {})
    if not isinstance(overrides, Mapping):
        raise TypeError("Tuning trial overrides must be a mapping")
    return deep_merge(baseline, overrides), trial


def training_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the path-independent scientific identity of a resolved run."""

    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    experiment = config["experiment"]
    return {
        "dataset": {
            "dataset_id": dataset["dataset_id"],
            "year": dataset["year"],
            "dataset_fingerprint": dataset["dataset_fingerprint"],
            "split_id": dataset["split_id"],
            "split_counts": dataset["split_counts"],
        },
        "preprocessing": {
            "id": preprocessing["id"],
            "fit_split": preprocessing["fit_split"],
            "dimension": preprocessing["dimension"],
            "artifact_sha256": preprocessing["artifact_sha256"],
            "metadata_sha256": preprocessing["metadata_sha256"],
            "feature_order_sha256": preprocessing["feature_order_sha256"],
        },
        "data": {
            "tree_name": config["data"]["tree_name"],
            "feature_order": config["data"]["feature_order"],
            "split_manifest_sha256": config["data"]["split_manifest_sha256"],
        },
        "model": config["model"],
        "training": config["training"],
        "experiment": {
            "stage": experiment["stage"],
            "trial_id": experiment["trial_id"],
            "type": experiment["type"],
            "data_roles": experiment["data_roles"],
        },
    }


def resolve_model4_training_config(
    *,
    project_root: str | Path,
    dataset_config: str | Path,
    prepared_directory: str | Path,
    output_root: str | Path,
    preprocessing: str,
    stage: str,
    trial_id: str = "base",
    training_seed: int = 42,
    maximum_epochs: int | None = None,
    batch_size: int | None = None,
    checkpoint_interval: int | None = None,
    early_stopping: bool | None = None,
    model_config: str | Path = "configs/models/model4/baseline.yaml",
    tuning_space: str | Path = "configs/models/model4/tuning_space.yaml",
) -> dict[str, Any]:
    """Compose maintained metadata and real paths without scanning ROOT data."""

    project = Path(project_root).resolve()
    prepared = _absolute_directory(prepared_directory, name="prepared_directory")
    outputs = _absolute_directory(output_root, name="output_root")
    pipeline = preprocessing.upper()
    if pipeline not in {"A", "B", "C"}:
        raise ValueError("preprocessing must be A, B, or C")
    if stage not in {"smoke", "production"}:
        raise ValueError("stage must be smoke or production")
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")

    dataset_path = Path(dataset_config)
    if not dataset_path.is_absolute():
        dataset_path = project / dataset_path
    model_path = Path(model_config)
    if not model_path.is_absolute():
        model_path = project / model_path
    tuning_path = Path(tuning_space)
    if not tuning_path.is_absolute():
        tuning_path = project / tuning_path
    preprocessing_path = project / f"configs/preprocessing/{pipeline}.yaml"
    dataset = load_yaml(dataset_path)
    baseline = load_yaml(model_path)
    space = load_yaml(tuning_path)
    preprocessing_definition = load_yaml(preprocessing_path)
    selected, trial = _trial(baseline, space, trial_id)

    dataset_id = str(dataset["dataset_id"])
    year = int(dataset["year"])
    if year == 2022 and dataset_id != "fluka2022_muons_down_tclean_v1":
        raise ValueError("2022 Model 4 training requires the t-clean dataset identity")
    split = dataset["split"]
    counts = {
        name: int(split["counts"][name])
        for name in ("train", "validation", "test")
    }
    if sum(counts.values()) != int(dataset["selected_rows"]):
        raise ValueError("Dataset selected_rows does not equal frozen split counts")

    model = {
        "family": "model4",
        "architecture": "rq_spline",
        "ablation": "drop_ze",
        "objective": "weighted_nll",
        "weight_is_input_feature": False,
        "num_transforms": 8,
        "hidden_features": 64,
        "num_blocks": 2,
        "use_residual_blocks": True,
        "permutation": "reverse",
        "base_distribution": "standard_normal",
        "dropout_probability": 0.0,
        "use_batch_norm": False,
        "num_bins": 8,
        "tails": "linear",
        "tail_bound": 8.0,
        **selected["model"],
    }
    ablation = str(model["ablation"])
    if ablation not in ABLATION_DROPS:
        raise ValueError(f"Unsupported Model 4 ablation: {ablation}")

    preprocessing_directory = prepared / f"preprocessing_{pipeline}"
    artifact = preprocessing_directory / "preprocessor.joblib"
    metadata = preprocessing_directory / "preprocessing_parameters.json"
    feature_order_path = preprocessing_directory / "8d" / "feature_order.json"
    split_manifest = prepared / "split" / "split_manifest.json"
    source_feature_order = json.loads(feature_order_path.read_text())
    if tuple(source_feature_order) != CANONICAL_8D:
        raise ValueError("Prepared 8d feature order does not match the canonical order")
    feature_order = [
        name for name in source_feature_order if name not in ABLATION_DROPS[ablation]
    ]

    training_defaults = {
        "optimizer": "AdamW",
        "learning_rate": 5.0e-4,
        "weight_decay": 1.0e-6,
        "batch_size": 2048,
        "maximum_epochs": 500,
        "checkpoint_interval": 50,
        "early_stopping": {"enabled": False, "patience": 20},
        "scheduler": {"type": "ReduceLROnPlateau", "factor": 0.5, "patience": 5},
        "mixed_precision": False,
        "gradient_clip_norm": 5.0,
        "num_workers": 4,
        "device": "auto",
    }
    training = deep_merge(training_defaults, selected.get("training", {}))
    if "max_epochs" in training:
        training["maximum_epochs"] = training.pop("max_epochs")
    training["seed"] = int(training_seed)
    if maximum_epochs is not None:
        training["maximum_epochs"] = int(maximum_epochs)
    elif stage == "smoke":
        training["maximum_epochs"] = 5
    if batch_size is not None:
        training["batch_size"] = int(batch_size)
    if checkpoint_interval is not None:
        training["checkpoint_interval"] = int(checkpoint_interval)
    if early_stopping is not None:
        training["early_stopping"]["enabled"] = bool(early_stopping)

    resolved: dict[str, Any] = {
        "dataset": {
            "dataset_id": dataset_id,
            "year": year,
            "dataset_fingerprint": str(dataset["dataset_fingerprint"]),
            "split_id": str(split["id"]),
            "split_counts": counts,
        },
        "preprocessing": {
            "id": pipeline,
            "display_name": preprocessing_definition["display_name"],
            "fit_split": "train",
            "dimension": "8d",
            "transforms": preprocessing_definition["transforms"],
            "artifact": str(artifact.resolve()),
            "artifact_sha256": sha256_file(artifact),
            "metadata": str(metadata.resolve()),
            "metadata_sha256": sha256_file(metadata),
            "feature_order_artifact": str(feature_order_path.resolve()),
            "feature_order_sha256": sha256_file(feature_order_path),
        },
        "data": {
            "tree_name": "nt",
            "feature_order": feature_order,
            "split_manifest_path": str(split_manifest.resolve()),
            "split_manifest_sha256": sha256_file(split_manifest),
            "train": {
                "model_space_path": str(
                    (preprocessing_directory / "8d/train_preprocessed.root").resolve()
                ),
                "raw_weight_path": str(
                    (prepared / "split/train_rawfeature.root").resolve()
                ),
                "expected_rows": counts["train"],
            },
            "validation": {
                "model_space_path": str(
                    (
                        preprocessing_directory
                        / "8d/validation_preprocessed.root"
                    ).resolve()
                ),
                "raw_weight_path": str(
                    (prepared / "split/validation_rawfeature.root").resolve()
                ),
                "expected_rows": counts["validation"],
            },
        },
        "model": model,
        "training": training,
        "experiment": {
            "stage": stage,
            "trial_id": trial_id,
            "display_name": str(trial["display_name"]),
            "type": "fresh_smoke" if stage == "smoke" else "fresh_production",
            "data_roles": {
                "fit": "train",
                "selection": "validation",
                "test": "reporting_only_not_loaded",
            },
        },
    }
    identity = config_hash(training_identity(resolved))
    run_id = build_run_id(
        year=year,
        model="m4",
        preprocessing=pipeline,
        trial_id=trial_id,
        training_seed=training_seed,
        config_hash=identity,
    )
    run_directory = (
        outputs
        / "campaigns"
        / dataset_id
        / "model4"
        / f"preprocessing_{pipeline}"
        / ablation
        / stage
        / trial_id
        / f"config_{identity}"
        / f"train_seed_{training_seed}"
    )
    if run_directory.exists():
        raise FileExistsError(f"Resolved run already exists: {run_directory}")
    resolved["output"] = {"run_directory": str(run_directory)}
    resolved["resolution"] = {
        "run_id": run_id,
        "config_hash": identity,
        "prepared_directory": str(prepared),
        "output_root": str(outputs),
        "sources": {
            "dataset": _source(dataset_path),
            "preprocessing": _source(preprocessing_path),
            "model": _source(model_path),
            "tuning_space": _source(tuning_path),
        },
    }
    return resolved


def write_resolved_training_config(
    path: str | Path, config: Mapping[str, Any]
) -> Path:
    """Atomically write one new resolved YAML and refuse every overwrite."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Resolved config already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(yaml.safe_dump(dict(config), sort_keys=False))
    temporary.replace(destination)
    return destination.resolve()
