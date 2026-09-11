"""Read-only fail-closed validation for resolved Model 4 training configs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import config_hash, load_yaml
from ..data import Model4RootAdapter, RootSplitSpec
from ..experiments.training_config import (
    ABLATION_DROPS,
    sha256_file,
    training_identity,
)
from ..models.flows import FlowConfig
from ..models.model4 import sample_weight_summary
from ..naming import build_run_id
from .state import TrainingPolicy

CANONICAL_8D = ("x", "y", "z", "E", "pz", "px", "py", "t")
CANONICAL_9D = (*CANONICAL_8D, "w")


class PreflightError(ValueError):
    """Raised when a resolved config is unsafe to start."""


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"{name} must be a mapping")
    return value


def _absolute_file(value: Any, *, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        raise PreflightError(f"{name} must be an absolute path")
    if not path.is_file():
        raise PreflightError(f"{name} is missing: {path}")
    return path.resolve()


def _matching_hash(path: Path, expected: Any, *, name: str) -> str:
    observed = sha256_file(path)
    if observed != str(expected):
        raise PreflightError(
            f"{name} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _json(path: Path, *, name: str) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"Cannot read {name}: {path}") from error


def _root_spec(config: Mapping[str, Any], split: str) -> RootSplitSpec:
    dataset = config["dataset"]
    data = config["data"]
    split_config = data[split]
    return RootSplitSpec(
        dataset_id=str(dataset["dataset_id"]),
        split=split,
        model_space_path=Path(split_config["model_space_path"]),
        raw_weight_path=Path(split_config["raw_weight_path"]),
        feature_order=tuple(data["feature_order"]),
        expected_rows=int(split_config["expected_rows"]),
        split_manifest_path=Path(data["split_manifest_path"]),
        expected_split_id=str(dataset["split_id"]),
        expected_dataset_fingerprint=str(dataset["dataset_fingerprint"]),
        tree_name=str(data["tree_name"]),
    )


def preflight_model4_training(config_path: str | Path) -> dict[str, Any]:
    """Fully scan train/validation inputs without creating any run output."""

    source = Path(config_path).resolve()
    config = load_yaml(source)
    for section in (
        "dataset",
        "preprocessing",
        "data",
        "model",
        "training",
        "experiment",
        "output",
        "resolution",
    ):
        _require_mapping(config.get(section), name=section)

    dataset = config["dataset"]
    preprocessing = config["preprocessing"]
    data = config["data"]
    model = config["model"]
    training = config["training"]
    output = config["output"]
    resolution = config["resolution"]
    checks: list[str] = []

    dataset_id = str(dataset.get("dataset_id", ""))
    if (
        not dataset_id
        or not dataset.get("dataset_fingerprint")
        or not dataset.get("split_id")
    ):
        raise PreflightError("Dataset identity, fingerprint, and split ID are required")
    if int(dataset["year"]) == 2022 and dataset_id != "fluka2022_muons_down_tclean_v1":
        raise PreflightError("2022 Model 4 training requires the t-clean dataset")
    counts = _require_mapping(dataset.get("split_counts"), name="dataset.split_counts")
    for split in ("train", "validation", "test"):
        if int(counts.get(split, 0)) <= 0:
            raise PreflightError(f"dataset.split_counts.{split} must be positive")
    if "test" in data:
        raise PreflightError("Resolved training data must not contain a test input")
    if set(data) & {"test_path", "test_file", "test_root"}:
        raise PreflightError("Resolved training config contains a test path")
    for split in ("train", "validation"):
        split_config = _require_mapping(data.get(split), name=f"data.{split}")
        if int(split_config.get("expected_rows", 0)) != int(counts[split]):
            raise PreflightError(f"data.{split}.expected_rows disagrees with manifest")
        for key in ("model_space_path", "raw_weight_path"):
            _absolute_file(split_config.get(key), name=f"data.{split}.{key}")
    checks.append("dataset_identity_and_split_roles")

    sources = _require_mapping(resolution.get("sources"), name="resolution.sources")
    for source_name in ("dataset", "preprocessing", "model", "tuning_space"):
        record = _require_mapping(
            sources.get(source_name), name=f"source.{source_name}"
        )
        path = _absolute_file(record.get("path"), name=f"source.{source_name}.path")
        _matching_hash(path, record.get("sha256"), name=f"source.{source_name}")
    checks.append("maintained_source_hashes")

    pipeline = str(preprocessing.get("id", "")).upper()
    if pipeline not in {"A", "B", "C", "D", "E"}:
        raise PreflightError("preprocessing.id must be A, B, C, D, or E")
    if preprocessing.get("fit_split") != "train":
        raise PreflightError("Preprocessing artifact must be fit on train only")
    if preprocessing.get("dimension") != "8d":
        raise PreflightError("Model 4 requires the prepared 8d representation")
    artifact = _absolute_file(
        preprocessing.get("artifact"), name="preprocessor artifact"
    )
    metadata_path = _absolute_file(
        preprocessing.get("metadata"), name="preprocessor metadata"
    )
    feature_order_path = _absolute_file(
        preprocessing.get("feature_order_artifact"), name="feature-order artifact"
    )
    _matching_hash(artifact, preprocessing.get("artifact_sha256"), name="preprocessor")
    _matching_hash(
        metadata_path,
        preprocessing.get("metadata_sha256"),
        name="preprocessor metadata",
    )
    _matching_hash(
        feature_order_path,
        preprocessing.get("feature_order_sha256"),
        name="feature-order artifact",
    )
    metadata = _json(metadata_path, name="preprocessor metadata")
    composed = metadata.get("format") == "flashsim_nf.composed_preprocessor"
    fitted = (
        metadata.get("fitted_on") == "train"
        if composed
        else metadata.get("fitted") is True
    )
    if metadata.get("pipeline") != pipeline or not fitted:
        raise PreflightError(
            "Preprocessor metadata has wrong pipeline or is not fitted"
        )
    if tuple(metadata.get("feature_order", ())) != CANONICAL_9D:
        raise PreflightError("Preprocessor metadata does not use canonical 9d order")
    if tuple(_json(feature_order_path, name="feature-order artifact")) != CANONICAL_8D:
        raise PreflightError("Prepared feature-order artifact is not canonical 8d")
    ablation = str(model.get("ablation", ""))
    if ablation not in ABLATION_DROPS:
        raise PreflightError(f"Unsupported Model 4 ablation: {ablation}")
    expected_features = tuple(
        name for name in CANONICAL_8D if name not in ABLATION_DROPS[ablation]
    )
    if tuple(data.get("feature_order", ())) != expected_features:
        raise PreflightError(
            f"Model 4 {ablation} feature order is not canonical: {expected_features}"
        )
    checks.append("preprocessor_identity_and_feature_contract")

    manifest_path = _absolute_file(
        data.get("split_manifest_path"), name="data.split_manifest_path"
    )
    _matching_hash(
        manifest_path, data.get("split_manifest_sha256"), name="split manifest"
    )
    checks.append("split_manifest_hash")

    if model.get("family") != "model4" or model.get("architecture") != "rq_spline":
        raise PreflightError("Only the Model 4 RQ-spline backend is allowed")
    if model.get("objective") != "weighted_nll":
        raise PreflightError("Model 4 requires objective=weighted_nll")
    if model.get("weight_is_input_feature") is not False:
        raise PreflightError("FLUKA weight must not be an NF input feature")
    flow = FlowConfig.from_mapping(model, input_dim=len(expected_features))
    checks.append("model_contract")

    TrainingPolicy.from_config(training)
    if training.get("optimizer") != "AdamW":
        raise PreflightError("training.optimizer must be AdamW")
    for name in ("learning_rate", "batch_size", "gradient_clip_norm"):
        if float(training.get(name, 0)) <= 0:
            raise PreflightError(f"training.{name} must be positive")
    scheduler = _require_mapping(training.get("scheduler"), name="training.scheduler")
    if scheduler.get("type") != "ReduceLROnPlateau":
        raise PreflightError("training.scheduler.type must be ReduceLROnPlateau")
    checks.append("training_policy")

    output_root = Path(str(resolution.get("output_root", "")))
    run_directory = Path(str(output.get("run_directory", "")))
    if not output_root.is_absolute() or not run_directory.is_absolute():
        raise PreflightError("Output root and run directory must be absolute")
    try:
        run_directory.relative_to(output_root)
    except ValueError as error:
        raise PreflightError(
            "Run directory is outside the declared output root"
        ) from error
    if run_directory.exists():
        raise PreflightError(f"Run directory already exists: {run_directory}")
    checks.append("new_output_destination")

    observed_hash = config_hash(training_identity(config))
    if observed_hash != resolution.get("config_hash"):
        raise PreflightError("Resolved scientific config hash is inconsistent")
    expected_run_id = build_run_id(
        year=int(dataset["year"]),
        model="m4",
        preprocessing=pipeline,
        trial_id=str(config["experiment"]["trial_id"]),
        training_seed=int(training["seed"]),
        config_hash=observed_hash,
    )
    if expected_run_id != resolution.get("run_id"):
        raise PreflightError("Resolved run ID is inconsistent")
    checks.append("immutable_run_identity")

    train, validation = Model4RootAdapter().load_training_pair(
        _root_spec(config, "train"), _root_spec(config, "validation")
    )
    checks.append("full_train_validation_root_scan")
    return {
        "status": "pass",
        "training_allowed": True,
        "config": str(source),
        "run_id": expected_run_id,
        "config_hash": observed_hash,
        "run_directory": str(run_directory),
        "checks": checks,
        "model": flow.to_dict(),
        "train": {
            "rows": int(train.features.shape[0]),
            "features": int(train.features.shape[1]),
            "weight_summary": sample_weight_summary(train.weights),
        },
        "validation": {
            "rows": int(validation.features.shape[0]),
            "features": int(validation.features.shape[1]),
            "weight_summary": sample_weight_summary(validation.weights),
        },
        "test_loaded": False,
        "read_only": True,
    }
