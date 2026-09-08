"""Fail-closed ROOT adapters for Model 4 training data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_METADATA = ("run", "event", "id", "generation")


@dataclass(frozen=True)
class RootSplitSpec:
    dataset_id: str
    split: str
    model_space_path: Path
    raw_weight_path: Path
    feature_order: tuple[str, ...]
    expected_rows: int
    split_manifest_path: Path
    expected_split_id: str
    expected_dataset_fingerprint: str
    tree_name: str = "nt"
    weight_branch: str = "w"
    feature_dtype: str = "float32"
    weight_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if self.expected_rows <= 0:
            raise ValueError("expected_rows must be positive")
        if not self.feature_order or len(self.feature_order) != len(
            set(self.feature_order)
        ):
            raise ValueError("feature_order must be non-empty and unique")
        if "w" in self.feature_order:
            raise ValueError("Model 4 weight must not be an input feature")
        for name, value in (
            ("feature_dtype", self.feature_dtype),
            ("weight_dtype", self.weight_dtype),
        ):
            if value not in {"float32", "float64"}:
                raise ValueError(f"{name} must be float32 or float64")


@dataclass(frozen=True)
class RootSplit:
    features: np.ndarray
    weights: np.ndarray
    metadata: dict[str, np.ndarray]
    provenance: dict[str, Any]


def _uproot():
    try:
        import uproot
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ROOT data loading requires the optional uproot dependency"
        ) from error
    return uproot


def _read_arrays(path: Path, tree_name: str, branches: tuple[str, ...]):
    uproot = _uproot()
    if not path.is_file():
        raise FileNotFoundError(f"ROOT file is missing: {path}")
    with uproot.open(path) as source:
        if tree_name not in source:
            raise KeyError(f"TTree {tree_name!r} not found in {path}")
        tree = source[tree_name]
        available = {str(name) for name in tree.keys()}
        missing = sorted(set(branches) - available)
        if missing:
            raise ValueError(f"Missing ROOT branches in {path}:{tree_name}: {missing}")
        arrays = tree.arrays(list(branches), library="np")
        rows = int(tree.num_entries)
    if isinstance(arrays, np.ndarray) and arrays.dtype.names:
        return {name: arrays[name] for name in branches}, rows
    return {name: arrays[name] for name in branches}, rows


class Model4RootAdapter:
    """Load aligned model-space kinematics and raw FLUKA weights."""

    def load(self, spec: RootSplitSpec) -> RootSplit:
        manifest = self._validate_manifest(spec)
        model_branches = (*SOURCE_METADATA, *spec.feature_order)
        raw_branches = (*SOURCE_METADATA, spec.weight_branch)
        model, model_rows = _read_arrays(
            spec.model_space_path, spec.tree_name, model_branches
        )
        raw, raw_rows = _read_arrays(spec.raw_weight_path, spec.tree_name, raw_branches)
        if model_rows != spec.expected_rows or raw_rows != spec.expected_rows:
            raise ValueError(
                "ROOT row count mismatch: "
                f"model={model_rows}, raw={raw_rows}, expected={spec.expected_rows}"
            )
        metadata: dict[str, np.ndarray] = {}
        for name in SOURCE_METADATA:
            if not np.array_equal(model[name], raw[name]):
                raise ValueError(f"Model-space and raw rows are misaligned in {name!r}")
            metadata[name] = np.asarray(model[name]).copy()
        features = np.column_stack([model[name] for name in spec.feature_order]).astype(
            np.dtype(spec.feature_dtype), copy=False
        )
        weights = np.asarray(
            raw[spec.weight_branch], dtype=np.dtype(spec.weight_dtype)
        ).reshape(-1)
        if not np.isfinite(features).all():
            raise ValueError("Model-space features contain non-finite values")
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Model 4 requires finite, strictly positive FLUKA weights")
        provenance = {
            "dataset_id": spec.dataset_id,
            "split": spec.split,
            "split_id": spec.expected_split_id,
            "dataset_fingerprint": spec.expected_dataset_fingerprint,
            "rows": spec.expected_rows,
            "feature_order": list(spec.feature_order),
            "weight_is_input_feature": False,
            "model_space_path": str(spec.model_space_path.resolve()),
            "raw_weight_path": str(spec.raw_weight_path.resolve()),
            "split_manifest_path": str(spec.split_manifest_path.resolve()),
            "manifest": manifest,
        }
        return RootSplit(
            features=features,
            weights=weights,
            metadata=metadata,
            provenance=provenance,
        )

    def load_training_pair(
        self, train: RootSplitSpec, validation: RootSplitSpec
    ) -> tuple[RootSplit, RootSplit]:
        if train.split != "train" or validation.split != "validation":
            raise ValueError("Training may load only train and validation splits")
        if train.dataset_id != validation.dataset_id:
            raise ValueError("Train and validation dataset IDs differ")
        if train.expected_split_id != validation.expected_split_id:
            raise ValueError("Train and validation split IDs differ")
        if train.feature_order != validation.feature_order:
            raise ValueError("Train and validation feature orders differ")
        return self.load(train), self.load(validation)

    @staticmethod
    def _validate_manifest(spec: RootSplitSpec) -> dict[str, Any]:
        if not spec.split_manifest_path.is_file():
            raise FileNotFoundError(
                f"Split manifest is missing: {spec.split_manifest_path}"
            )
        manifest = json.loads(spec.split_manifest_path.read_text())
        expected = {
            "dataset_id": spec.dataset_id,
            "split_id": spec.expected_split_id,
            "dataset_fingerprint": spec.expected_dataset_fingerprint,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError(
                    f"Split manifest {key} mismatch: {manifest.get(key)!r} != {value!r}"
                )
        count = manifest.get("counts", {}).get(spec.split)
        if int(count) != spec.expected_rows:
            raise ValueError(f"Split manifest row count mismatch for {spec.split}")
        return manifest


def split_spec_to_dict(spec: RootSplitSpec) -> dict[str, Any]:
    payload = asdict(spec)
    for key in ("model_space_path", "raw_weight_path", "split_manifest_path"):
        payload[key] = str(payload[key])
    payload["feature_order"] = list(payload["feature_order"])
    return payload
