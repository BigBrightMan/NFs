"""Weighted robust reference guards with explicit train/all-data roles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data import Model4RootAdapter, RootSplitSpec

PHYSICAL_FEATURES = ("x", "y", "z", "px", "py", "pz", "E", "t")
HARD_SUPPORT_FEATURES = PHYSICAL_FEATURES
FEATURE_TRANSFORMS = {
    "x": "identity",
    "y": "identity",
    "z": "identity",
    "px": "signed_log1p",
    "py": "signed_log1p",
    "pz": "log",
    "E": "log",
    "t": "signed_log1p",
}
FORMAT_NAME = "flashsim_nf.robust_reference_guard"
FORMAT_VERSION = 2


@dataclass(frozen=True)
class ReferenceData:
    features: dict[str, np.ndarray]
    weights: np.ndarray
    label: str

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if len(weights) == 0:
            raise ValueError("Reference data must not be empty")
        if not np.isfinite(weights).all() or np.any(weights <= 0.0):
            raise ValueError("Reference weights must be finite and positive")
        missing = sorted(set(PHYSICAL_FEATURES) - set(self.features))
        if missing:
            raise ValueError(f"Reference data is missing features: {missing}")
        for name in PHYSICAL_FEATURES:
            values = np.asarray(self.features[name], dtype=np.float64).reshape(-1)
            if len(values) != len(weights):
                raise ValueError(f"Feature {name!r} and weight row counts differ")

    @property
    def rows(self) -> int:
        return len(self.weights)


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(values.sum(dtype=np.float64))
    return float(total * total / np.square(values).sum(dtype=np.float64))


def weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray | list[float] | tuple[float, ...],
    weights: np.ndarray,
) -> np.ndarray:
    """Return deterministic weighted empirical quantiles."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("Weighted-quantile values and weights must align")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Weighted-quantile values must be non-empty and finite")
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Weighted-quantile weights must be finite and positive")
    if np.any((quantiles < 0.0) | (quantiles > 1.0)):
        raise ValueError("Quantiles must lie in [0, 1]")
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    positions = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / np.sum(
        ordered_weights
    )
    return np.interp(
        quantiles,
        positions,
        ordered_values,
        left=ordered_values[0],
        right=ordered_values[-1],
    )


def _transform(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if name == "identity":
        return values
    if name == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    if name == "log":
        result = np.full(values.shape, np.nan, dtype=np.float64)
        positive = values > 0.0
        result[positive] = np.log(values[positive])
        return result
    raise ValueError(f"Unknown robust-guard transform: {name}")


def _inverse_bound(value: float, name: str) -> float:
    if name == "identity":
        return float(value)
    if name == "signed_log1p":
        return float(np.sign(value) * np.expm1(abs(value)))
    if name == "log":
        return float(np.exp(value))
    raise ValueError(f"Unknown robust-guard transform: {name}")


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _concatenate(
    splits: dict[str, ReferenceData], source_splits: tuple[str, ...]
) -> ReferenceData:
    missing = sorted(set(source_splits) - set(splits))
    if missing:
        raise ValueError(f"Missing requested guard-fit splits: {missing}")
    return ReferenceData(
        features={
            name: np.concatenate(
                [splits[split].features[name] for split in source_splits]
            )
            for name in PHYSICAL_FEATURES
        },
        weights=np.concatenate([splits[split].weights for split in source_splits]),
        label="+".join(source_splits),
    )


def fit_reference_guard(
    splits: dict[str, ReferenceData],
    *,
    dataset_id: str,
    dataset_fingerprint: str,
    split_id: str,
    fit_scope: str,
    iqr_multiplier: float = 3.0,
    lower_quantile: float = 0.0001,
    upper_quantile: float = 0.9999,
) -> dict[str, Any]:
    """Fit a weighted catastrophic-tail guard with an immutable usage role."""

    scopes = {
        "train": ("train",),
        "all_clean_splits": ("train", "validation", "test"),
    }
    if fit_scope not in scopes:
        raise ValueError(f"fit_scope must be one of {sorted(scopes)}")
    if not dataset_id or not dataset_fingerprint or not split_id:
        raise ValueError("Dataset ID, fingerprint, and split ID are required")
    if not np.isfinite(iqr_multiplier) or iqr_multiplier <= 0.0:
        raise ValueError("iqr_multiplier must be finite and positive")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Guard quantiles must satisfy 0 < lower < upper < 1")
    source_splits = scopes[fit_scope]
    fitted = _concatenate(splits, source_splits)
    bounds: dict[str, Any] = {}
    for feature in PHYSICAL_FEATURES:
        transform = FEATURE_TRANSFORMS[feature]
        transformed = _transform(fitted.features[feature], transform)
        finite = np.isfinite(transformed)
        if not finite.all():
            raise ValueError(
                f"Cannot fit {feature!r}: transformed reference values are non-finite"
            )
        q_lower, q1, median, q3, q_upper = weighted_quantile(
            transformed,
            (lower_quantile, 0.25, 0.5, 0.75, upper_quantile),
            fitted.weights,
        )
        iqr = float(q3 - q1)
        lower_fence = float(q1 - iqr_multiplier * iqr)
        upper_fence = float(q3 + iqr_multiplier * iqr)
        lower = float(min(lower_fence, q_lower))
        upper = float(max(upper_fence, q_upper))
        bounds[feature] = {
            "transform": transform,
            "lower": lower,
            "upper": upper,
            "physical_lower": _inverse_bound(lower, transform),
            "physical_upper": _inverse_bound(upper, transform),
            "weighted_q_lower": float(q_lower),
            "weighted_q1": float(q1),
            "weighted_median": float(median),
            "weighted_q3": float(q3),
            "weighted_q_upper": float(q_upper),
            "weighted_iqr": iqr,
            "lower_iqr_fence": lower_fence,
            "upper_iqr_fence": upper_fence,
        }
    hard_support_bounds = {
        feature: {
            "physical_lower": float(np.min(fitted.features[feature])),
            "physical_upper": float(np.max(fitted.features[feature])),
        }
        for feature in HARD_SUPPORT_FEATURES
    }
    selection_allowed = fit_scope == "train"
    artifact: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "split_id": split_id,
        "fit_scope": fit_scope,
        "source_splits": list(source_splits),
        "usage_role": (
            "validation_and_selection" if selection_allowed else "production_only"
        ),
        "selection_allowed": selection_allowed,
        "fit_rows": fitted.rows,
        "fit_sum_w": float(np.sum(fitted.weights, dtype=np.float64)),
        "fit_effective_sample_size": effective_sample_size(fitted.weights),
        "settings": {
            "weighting": "original_fluka_weight",
            "iqr_multiplier": float(iqr_multiplier),
            "lower_quantile": float(lower_quantile),
            "upper_quantile": float(upper_quantile),
            "bound_rule": "wider_of_weighted_iqr_fence_and_weighted_quantiles",
            "raw_minmax_used": True,
            "raw_minmax_usage": "hard_support_all_physical_features",
            "robust_bound_action": "diagnostic_only",
        },
        "hard_support": {
            "contract_id": "per_dataset_all_features_raw_minmax_v1",
            "rule": "raw_observed_minmax",
            "weighting": "unweighted_support",
            "fit_scope": fit_scope,
            "source_splits": list(source_splits),
            "feature_bounds": hard_support_bounds,
        },
        "feature_bounds": bounds,
    }
    artifact["artifact_sha256"] = _canonical_hash(artifact)
    return artifact


def _rejection_masks(
    data: ReferenceData, artifact: dict[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reasons: dict[str, np.ndarray] = {}
    rejected = np.zeros(data.rows, dtype=bool)
    for feature, bound in artifact["feature_bounds"].items():
        transformed = _transform(data.features[feature], str(bound["transform"]))
        outside = (~np.isfinite(transformed)) | (
            (transformed < float(bound["lower"]))
            | (transformed > float(bound["upper"]))
        )
        reasons[f"diagnostic_robust_{feature}_outside"] = outside
    hard_support = artifact.get("hard_support")
    if not isinstance(hard_support, dict):
        raise ValueError("Reference guard has no per-dataset hard-support contract")
    support_bounds = hard_support.get("feature_bounds")
    if not isinstance(support_bounds, dict):
        raise ValueError("Reference guard has no hard-support feature bounds")
    for feature in HARD_SUPPORT_FEATURES:
        if feature not in support_bounds:
            raise ValueError(
                f"Reference guard has no hard-support bound for {feature!r}"
            )
        bound = support_bounds[feature]
        values = np.asarray(data.features[feature], dtype=np.float64)
        outside = (~np.isfinite(values)) | (
            (values < float(bound["physical_lower"]))
            | (values > float(bound["physical_upper"]))
        )
        reasons[f"hard_support_{feature}_outside"] = outside
        rejected |= outside
    return rejected, reasons


def guard_rejection_masks(
    data: ReferenceData, artifact: dict[str, Any]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return inclusive and per-feature masks after validating guard identity."""

    if artifact.get("format") != FORMAT_NAME:
        raise ValueError("Unsupported robust reference guard format")
    if int(artifact.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("Unsupported robust reference guard version")
    expected = artifact.get("artifact_sha256")
    unhashed = dict(artifact)
    unhashed.pop("artifact_sha256", None)
    if expected != _canonical_hash(unhashed):
        raise ValueError("Robust reference guard SHA-256 mismatch")
    return _rejection_masks(data, artifact)


def evaluate_guard(data: ReferenceData, artifact: dict[str, Any]) -> dict[str, Any]:
    """Evaluate inclusive rejection and non-exclusive feature reasons."""

    rejected, reasons = _rejection_masks(data, artifact)
    total_weight = float(np.sum(data.weights, dtype=np.float64))
    rejected_weight = float(np.sum(data.weights[rejected], dtype=np.float64))
    return {
        "label": data.label,
        "rows": data.rows,
        "sum_w": total_weight,
        "effective_sample_size": effective_sample_size(data.weights),
        "rejected_rows": int(rejected.sum()),
        "rejected_row_fraction": float(rejected.mean()),
        "rejected_sum_w": rejected_weight,
        "rejected_weight_fraction": rejected_weight / total_weight,
        "reasons": {
            name: {
                "rows": int(mask.sum()),
                "row_fraction": float(mask.mean()),
                "sum_w": float(np.sum(data.weights[mask], dtype=np.float64)),
                "weight_fraction": float(
                    np.sum(data.weights[mask], dtype=np.float64) / total_weight
                ),
            }
            for name, mask in reasons.items()
        },
    }


def compare_reference_guards(
    splits: dict[str, ReferenceData],
    *,
    train_guard: dict[str, Any],
    all_guard: dict[str, Any],
    generated: ReferenceData | None = None,
) -> dict[str, Any]:
    """Compare both guards on every reference split and the same proposals."""

    for key in ("dataset_id", "dataset_fingerprint", "split_id"):
        if train_guard.get(key) != all_guard.get(key):
            raise ValueError(f"Guard {key} mismatch")
    if train_guard.get("fit_scope") != "train":
        raise ValueError("train_guard must use fit_scope=train")
    if all_guard.get("fit_scope") != "all_clean_splits":
        raise ValueError("all_guard must use fit_scope=all_clean_splits")
    report: dict[str, Any] = {
        "dataset_id": train_guard["dataset_id"],
        "dataset_fingerprint": train_guard["dataset_fingerprint"],
        "split_id": train_guard["split_id"],
        "selection_guard_sha256": train_guard["artifact_sha256"],
        "production_guard_sha256": all_guard["artifact_sha256"],
        "reference": {
            split: {
                "train_ref": evaluate_guard(data, train_guard),
                "all_ref": evaluate_guard(data, all_guard),
            }
            for split, data in splits.items()
        },
        "bounds": {},
    }
    for feature in PHYSICAL_FEATURES:
        train_bound = train_guard["feature_bounds"][feature]
        all_bound = all_guard["feature_bounds"][feature]
        report["bounds"][feature] = {
            "train_physical_lower": train_bound["physical_lower"],
            "train_physical_upper": train_bound["physical_upper"],
            "all_physical_lower": all_bound["physical_lower"],
            "all_physical_upper": all_bound["physical_upper"],
        }
    report["hard_support"] = {}
    for feature in HARD_SUPPORT_FEATURES:
        train_bound = train_guard["hard_support"]["feature_bounds"][feature]
        all_bound = all_guard["hard_support"]["feature_bounds"][feature]
        report["hard_support"][feature] = {
            "train_physical_lower": train_bound["physical_lower"],
            "train_physical_upper": train_bound["physical_upper"],
            "all_physical_lower": all_bound["physical_lower"],
            "all_physical_upper": all_bound["physical_upper"],
        }
    if generated is not None:
        train_rejected, _ = _rejection_masks(generated, train_guard)
        all_rejected, _ = _rejection_masks(generated, all_guard)
        union = train_rejected | all_rejected
        intersection = train_rejected & all_rejected
        report["generated_same_proposals"] = {
            "train_ref": evaluate_guard(generated, train_guard),
            "all_ref": evaluate_guard(generated, all_guard),
            "rejected_by_both": int(intersection.sum()),
            "rejected_by_train_only": int((train_rejected & ~all_rejected).sum()),
            "rejected_by_all_only": int((all_rejected & ~train_rejected).sum()),
            "rejection_jaccard": (
                float(intersection.sum() / union.sum()) if union.any() else 1.0
            ),
        }
    return report


def load_reference_splits(
    dataset_config: dict[str, Any], prepared_directory: str | Path
) -> dict[str, ReferenceData]:
    """Load and validate train/validation/test physical reference ROOT files."""

    prepared = Path(prepared_directory)
    split = dataset_config["split"]
    adapter = Model4RootAdapter()
    result: dict[str, ReferenceData] = {}
    for name in ("train", "validation", "test"):
        raw = prepared / "split" / f"{name}_rawfeature.root"
        loaded = adapter.load(
            RootSplitSpec(
                dataset_id=str(dataset_config["dataset_id"]),
                split=name,
                model_space_path=raw,
                raw_weight_path=raw,
                feature_order=PHYSICAL_FEATURES,
                expected_rows=int(split["counts"][name]),
                split_manifest_path=prepared / "split" / "split_manifest.json",
                expected_split_id=str(split["id"]),
                expected_dataset_fingerprint=str(dataset_config["dataset_fingerprint"]),
                feature_dtype="float64",
                weight_dtype="float64",
            )
        )
        result[name] = ReferenceData(
            features={
                feature: loaded.features[:, index].astype(np.float64, copy=False)
                for index, feature in enumerate(PHYSICAL_FEATURES)
            },
            weights=loaded.weights.astype(np.float64, copy=False),
            label=name,
        )
    return result


def load_generated_root(path: str | Path, *, tree_name: str = "nt") -> ReferenceData:
    """Load one pre-rejection proposal ROOT file; generated rows have unit weight."""

    try:
        import uproot
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Generated ROOT loading requires uproot") from error
    source_path = Path(path)
    with uproot.open(source_path) as source:
        tree = source[tree_name]
        missing = sorted(set(PHYSICAL_FEATURES) - {str(key) for key in tree.keys()})
        if missing:
            raise ValueError(f"Generated ROOT is missing features: {missing}")
        arrays = tree.arrays(list(PHYSICAL_FEATURES), library="np")
        rows = int(tree.num_entries)
    if isinstance(arrays, np.ndarray) and arrays.dtype.names:
        features = {name: arrays[name] for name in PHYSICAL_FEATURES}
    else:
        features = {name: arrays[name] for name in PHYSICAL_FEATURES}
    return ReferenceData(
        features=features,
        weights=np.ones(rows, dtype=np.float64),
        label=str(source_path.resolve()),
    )
