"""Weighted global and tail evaluation in common physical coordinates."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, wasserstein_distance

from ..manifest import write_json_atomic
from ..models.model4 import sample_weight_summary
from .plots import write_generated_evaluation_plots

FEATURES_8D = ("x", "y", "z", "E", "pz", "px", "py", "t")


@dataclass(frozen=True)
class EvaluationSettings:
    tail_quantiles: tuple[float, ...] = (0.99, 0.999, 0.9999)
    ccdf_quantiles: tuple[float, ...] = (0.9, 0.95, 0.99, 0.995, 0.999)
    energy_sample_size: int = 1000
    energy_repeats: int = 5
    sliced_wasserstein_projections: int = 64
    minimum_tail_ess: float = 20.0
    random_seed: int = 1556
    c2st_max_rows_per_class: int = 50_000

    def __post_init__(self) -> None:
        if not self.tail_quantiles or any(
            value <= 0.5 or value >= 1.0 for value in self.tail_quantiles
        ):
            raise ValueError("tail_quantiles must lie strictly between 0.5 and 1")
        if not self.ccdf_quantiles or any(
            value <= 0.5 or value >= 1.0 for value in self.ccdf_quantiles
        ):
            raise ValueError("ccdf_quantiles must lie strictly between 0.5 and 1")
        if (
            min(
                self.energy_sample_size,
                self.energy_repeats,
                self.sliced_wasserstein_projections,
                self.c2st_max_rows_per_class,
            )
            <= 0
        ):
            raise ValueError("Monte Carlo evaluation settings must be positive")
        if self.minimum_tail_ess <= 0:
            raise ValueError("minimum_tail_ess must be positive")


def _validate(values: np.ndarray, weights: np.ndarray, *, name: str) -> None:
    if values.ndim != 2 or values.shape[1] != len(FEATURES_8D):
        raise ValueError(f"{name} must have shape (rows, {len(FEATURES_8D)})")
    if weights.shape != (len(values),):
        raise ValueError(f"{name} weights are not aligned")
    if len(values) < 4 or not np.isfinite(values).all():
        raise ValueError(f"{name} requires at least four finite rows")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError(f"{name} weights must be finite and positive")


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or len(values) == 0:
        raise ValueError("Weighted quantiles require aligned non-empty arrays")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("Weighted quantiles require finite inputs")
    if np.any(weights <= 0):
        raise ValueError("Weighted quantiles require positive weights")
    requested = np.asarray(quantiles, dtype=np.float64)
    if np.any(requested < 0) or np.any(requested > 1):
        raise ValueError("Quantiles must lie in [0, 1]")
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    return np.interp(requested, cumulative, ordered_values)


def _weighted_moments(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    normalized = weights / weights.sum()
    mean = np.sum(matrix * normalized[:, None], axis=0)
    centered = matrix - mean
    covariance = (centered * normalized[:, None]).T @ centered
    return mean, 0.5 * (covariance + covariance.T)


def _weighted_correlation(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    _, covariance = _weighted_moments(matrix, weights)
    scale = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denominator = np.outer(scale, scale)
    correlation = np.divide(
        covariance,
        denominator,
        out=np.zeros_like(covariance),
        where=denominator > np.finfo(float).eps,
    )
    np.fill_diagonal(correlation, 1.0)
    return np.clip(correlation, -1.0, 1.0)


def _correlation_metrics(
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
) -> dict[str, Any]:
    pearson_reference = _weighted_correlation(reference, reference_weights)
    pearson_generated = _weighted_correlation(generated, generated_weights)
    reference_ranks = np.column_stack(
        [rankdata(reference[:, index]) for index in range(reference.shape[1])]
    )
    generated_ranks = np.column_stack(
        [rankdata(generated[:, index]) for index in range(generated.shape[1])]
    )
    spearman_reference = _weighted_correlation(reference_ranks, reference_weights)
    spearman_generated = _weighted_correlation(generated_ranks, generated_weights)
    return {
        "pearson": {
            "reference": pearson_reference.tolist(),
            "generated": pearson_generated.tolist(),
            "difference": (pearson_generated - pearson_reference).tolist(),
            "mean_absolute_difference": float(
                np.mean(np.abs(pearson_generated - pearson_reference))
            ),
        },
        "spearman": {
            "reference": spearman_reference.tolist(),
            "generated": spearman_generated.tolist(),
            "difference": (spearman_generated - spearman_reference).tolist(),
            "mean_absolute_difference": float(
                np.mean(np.abs(spearman_generated - spearman_reference))
            ),
        },
    }


def _c2st(
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    settings: EvaluationSettings,
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("C2ST evaluation requires scikit-learn") from error
    rng = np.random.default_rng(settings.random_seed + 2)
    rows = min(len(reference), len(generated), settings.c2st_max_rows_per_class)
    reference_indices = rng.choice(
        len(reference),
        size=rows,
        replace=True,
        p=reference_weights / reference_weights.sum(),
    )
    generated_indices = rng.choice(
        len(generated),
        size=rows,
        replace=True,
        p=generated_weights / generated_weights.sum(),
    )
    values = np.vstack([reference[reference_indices], generated[generated_indices]])
    labels = np.concatenate([np.zeros(rows), np.ones(rows)])
    indices = np.arange(len(values))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.3,
        random_state=settings.random_seed + 2,
        stratify=labels,
    )
    classifier = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=31,
        learning_rate=0.1,
        random_state=settings.random_seed + 2,
    )
    classifier.fit(
        values[train_indices],
        labels[train_indices],
    )
    probability = classifier.predict_proba(values[test_indices])[:, 1]
    auc = float(roc_auc_score(labels[test_indices], probability))
    return {
        "classifier": "weighted_hist_gradient_boosting",
        "rows_per_class": rows,
        "sampling": "weighted_resampling_with_replacement",
        "test_fraction": 0.3,
        "roc_auc": auc,
        "distance_from_ideal_half": abs(auc - 0.5),
        "interpretation": "0.5 is indistinguishable; larger separation is worse",
    }


def _psd_square_root(matrix: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ (eigenvectors.T)


def frechet_gaussian_distance(
    first_mean: np.ndarray,
    first_covariance: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
) -> float:
    mean_term = float(np.sum(np.square(first_mean - second_mean)))
    first_root = _psd_square_root(first_covariance)
    middle_root = _psd_square_root(first_root @ second_covariance @ first_root)
    covariance_term = float(
        np.trace(first_covariance + second_covariance - 2.0 * middle_root)
    )
    return max(mean_term + covariance_term, 0.0)


def weighted_ks_statistic(
    first: np.ndarray,
    second: np.ndarray,
    first_weights: np.ndarray,
    second_weights: np.ndarray,
) -> float:
    first_order = np.argsort(first)
    second_order = np.argsort(second)
    first_values = first[first_order]
    second_values = second[second_order]
    first_cdf = np.cumsum(first_weights[first_order]) / first_weights.sum()
    second_cdf = np.cumsum(second_weights[second_order]) / second_weights.sum()
    points = np.sort(np.concatenate((first_values, second_values)))
    first_indices = np.searchsorted(first_values, points, side="right") - 1
    second_indices = np.searchsorted(second_values, points, side="right") - 1
    first_at = np.where(first_indices >= 0, first_cdf[np.maximum(first_indices, 0)], 0)
    second_at = np.where(
        second_indices >= 0, second_cdf[np.maximum(second_indices, 0)], 0
    )
    return float(np.max(np.abs(first_at - second_at)))


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum(dtype=np.float64))
    return total * total / float(np.square(weights).sum(dtype=np.float64))


def _weighted_mass(mask: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[mask].sum(dtype=np.float64) / weights.sum(dtype=np.float64))


def _conditional_mean(
    values: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> float | None:
    if not np.any(mask):
        return None
    return float(np.average(values[mask], weights=weights[mask]))


def _conditional_wasserstein(
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    reference_mask: np.ndarray,
    generated_mask: np.ndarray,
    scale: float,
) -> float | None:
    if not np.any(reference_mask) or not np.any(generated_mask):
        return None
    distance = wasserstein_distance(
        reference[reference_mask],
        generated[generated_mask],
        u_weights=reference_weights[reference_mask],
        v_weights=generated_weights[generated_mask],
    )
    return float(distance / scale)


def _tail_ess(mask: np.ndarray, weights: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return _effective_sample_size(weights[mask])


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _tail_metrics(
    train: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    train_weights: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    settings: EvaluationSettings,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for index, feature in enumerate(FEATURES_8D):
        train_values = train[:, index]
        reference_values = reference[:, index]
        generated_values = generated[:, index]
        train_core = weighted_quantile(train_values, train_weights, (0.001, 0.999))
        robust_width = max(float(train_core[1] - train_core[0]), np.finfo(float).eps)
        feature_levels: dict[str, Any] = {}
        for quantile in settings.tail_quantiles:
            lower_probability = 1.0 - quantile
            lower, upper = weighted_quantile(
                train_values, train_weights, (lower_probability, quantile)
            )
            reference_quantiles = weighted_quantile(
                reference_values,
                reference_weights,
                (lower_probability, quantile),
            )
            generated_quantiles = weighted_quantile(
                generated_values,
                generated_weights,
                (lower_probability, quantile),
            )
            reference_lower = reference_values <= lower
            generated_lower = generated_values <= lower
            reference_upper = reference_values >= upper
            generated_upper = generated_values >= upper
            reference_lower_mass = _weighted_mass(reference_lower, reference_weights)
            generated_lower_mass = _weighted_mass(generated_lower, generated_weights)
            reference_upper_mass = _weighted_mass(reference_upper, reference_weights)
            generated_upper_mass = _weighted_mass(generated_upper, generated_weights)
            reference_lower_ess = _tail_ess(reference_lower, reference_weights)
            generated_lower_ess = _tail_ess(generated_lower, generated_weights)
            reference_upper_ess = _tail_ess(reference_upper, reference_weights)
            generated_upper_ess = _tail_ess(generated_upper, generated_weights)
            feature_levels[f"q{quantile:.4f}"] = {
                "train_thresholds": {"lower": float(lower), "upper": float(upper)},
                "quantile_error_over_train_core_width": {
                    "lower": float(
                        (generated_quantiles[0] - reference_quantiles[0]) / robust_width
                    ),
                    "upper": float(
                        (generated_quantiles[1] - reference_quantiles[1]) / robust_width
                    ),
                },
                "lower": {
                    "reference_mass": reference_lower_mass,
                    "generated_mass": generated_lower_mass,
                    "generated_over_reference_mass": _ratio(
                        generated_lower_mass, reference_lower_mass
                    ),
                    "reference_ess": reference_lower_ess,
                    "generated_ess": generated_lower_ess,
                    "conditional_mean_reference": _conditional_mean(
                        reference_values, reference_weights, reference_lower
                    ),
                    "conditional_mean_generated": _conditional_mean(
                        generated_values, generated_weights, generated_lower
                    ),
                    "conditional_normalized_wasserstein": _conditional_wasserstein(
                        reference_values,
                        generated_values,
                        reference_weights,
                        generated_weights,
                        reference_lower,
                        generated_lower,
                        robust_width,
                    ),
                    "insufficient_statistics": min(
                        reference_lower_ess, generated_lower_ess
                    )
                    < settings.minimum_tail_ess,
                },
                "upper": {
                    "reference_mass": reference_upper_mass,
                    "generated_mass": generated_upper_mass,
                    "generated_over_reference_mass": _ratio(
                        generated_upper_mass, reference_upper_mass
                    ),
                    "reference_ess": reference_upper_ess,
                    "generated_ess": generated_upper_ess,
                    "conditional_mean_reference": _conditional_mean(
                        reference_values, reference_weights, reference_upper
                    ),
                    "conditional_mean_generated": _conditional_mean(
                        generated_values, generated_weights, generated_upper
                    ),
                    "conditional_normalized_wasserstein": _conditional_wasserstein(
                        reference_values,
                        generated_values,
                        reference_weights,
                        generated_weights,
                        reference_upper,
                        generated_upper,
                        robust_width,
                    ),
                    "insufficient_statistics": min(
                        reference_upper_ess, generated_upper_ess
                    )
                    < settings.minimum_tail_ess,
                },
            }

        ccdf_thresholds = weighted_quantile(
            train_values, train_weights, settings.ccdf_quantiles
        )
        ccdf_rows = []
        log_ratios = []
        for probability, threshold in zip(settings.ccdf_quantiles, ccdf_thresholds):
            reference_mass = _weighted_mass(
                reference_values >= threshold, reference_weights
            )
            generated_mass = _weighted_mass(
                generated_values >= threshold, generated_weights
            )
            ratio = _ratio(generated_mass, reference_mass)
            if ratio is not None and ratio > 0:
                log_ratios.append(float(np.log10(ratio)))
            ccdf_rows.append(
                {
                    "train_quantile": probability,
                    "threshold": float(threshold),
                    "reference_mass": reference_mass,
                    "generated_mass": generated_mass,
                    "generated_over_reference_mass": ratio,
                }
            )
        report[feature] = {
            "levels": feature_levels,
            "upper_ccdf": {
                "points": ccdf_rows,
                "rms_log10_mass_ratio": (
                    float(np.sqrt(np.mean(np.square(log_ratios))))
                    if log_ratios
                    else None
                ),
            },
        }
    return report


def _energy_distance(
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    settings: EvaluationSettings,
) -> dict[str, Any]:
    rng = np.random.default_rng(settings.random_seed)
    reference_probability = reference_weights / reference_weights.sum()
    generated_probability = generated_weights / generated_weights.sum()
    estimates = []
    size = settings.energy_sample_size
    for _ in range(settings.energy_repeats):
        x1 = reference[rng.choice(len(reference), size=size, p=reference_probability)]
        x2 = reference[rng.choice(len(reference), size=size, p=reference_probability)]
        y1 = generated[rng.choice(len(generated), size=size, p=generated_probability)]
        y2 = generated[rng.choice(len(generated), size=size, p=generated_probability)]
        estimate = float(
            2.0 * cdist(x1, y1).mean() - cdist(x1, x2).mean() - cdist(y1, y2).mean()
        )
        estimates.append(estimate)
    return {
        "method": "independent weighted Monte Carlo resampling",
        "sample_size_per_draw": size,
        "repeats": settings.energy_repeats,
        "seed": settings.random_seed,
        "raw_estimates": estimates,
        "mean_raw": float(np.mean(estimates)),
        "mean_clipped_nonnegative": max(float(np.mean(estimates)), 0.0),
        "standard_deviation": float(np.std(estimates, ddof=0)),
    }


def _sliced_wasserstein(
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    settings: EvaluationSettings,
) -> dict[str, Any]:
    rng = np.random.default_rng(settings.random_seed + 1)
    directions = rng.normal(
        size=(settings.sliced_wasserstein_projections, reference.shape[1])
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    distances = [
        float(
            wasserstein_distance(
                reference @ direction,
                generated @ direction,
                u_weights=reference_weights,
                v_weights=generated_weights,
            )
        )
        for direction in directions
    ]
    return {
        "projections": settings.sliced_wasserstein_projections,
        "seed": settings.random_seed + 1,
        "mean": float(np.mean(distances)),
        "standard_deviation": float(np.std(distances, ddof=0)),
        "maximum": float(np.max(distances)),
    }


def evaluate_generated_arrays(
    *,
    train: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    train_weights: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    settings: EvaluationSettings,
) -> dict[str, Any]:
    """Evaluate generated and reference samples with train-defined scales/tails."""

    train = np.asarray(train, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    train_weights = np.asarray(train_weights, dtype=np.float64)
    reference_weights = np.asarray(reference_weights, dtype=np.float64)
    generated_weights = np.asarray(generated_weights, dtype=np.float64)
    _validate(train, train_weights, name="train")
    _validate(reference, reference_weights, name="reference")
    _validate(generated, generated_weights, name="generated")

    train_mean, train_covariance = _weighted_moments(train, train_weights)
    scale = np.sqrt(np.clip(np.diag(train_covariance), 0.0, None))
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    standardized_reference = (reference - train_mean) / scale
    standardized_generated = (generated - train_mean) / scale
    reference_mean, reference_covariance = _weighted_moments(
        standardized_reference, reference_weights
    )
    generated_mean, generated_covariance = _weighted_moments(
        standardized_generated, generated_weights
    )

    marginal = {}
    for index, feature in enumerate(FEATURES_8D):
        train_core = weighted_quantile(train[:, index], train_weights, (0.001, 0.999))
        width = max(float(train_core[1] - train_core[0]), np.finfo(float).eps)
        distance = float(
            wasserstein_distance(
                reference[:, index],
                generated[:, index],
                u_weights=reference_weights,
                v_weights=generated_weights,
            )
        )
        marginal[feature] = {
            "weighted_ks": weighted_ks_statistic(
                reference[:, index],
                generated[:, index],
                reference_weights,
                generated_weights,
            ),
            "weighted_wasserstein": distance,
            "normalized_weighted_wasserstein": distance / width,
            "train_core_width_q001_q999": width,
        }

    return {
        "feature_order": list(FEATURES_8D),
        "settings": asdict(settings),
        "weight_summaries": {
            "train": sample_weight_summary(train_weights),
            "reference": sample_weight_summary(reference_weights),
            "generated": sample_weight_summary(generated_weights),
        },
        "global_multivariate": {
            "standardization": "weighted train mean and standard deviation",
            "frechet_gaussian_distance": frechet_gaussian_distance(
                reference_mean,
                reference_covariance,
                generated_mean,
                generated_covariance,
            ),
            "covariance_frobenius_distance": float(
                np.linalg.norm(generated_covariance - reference_covariance, ord="fro")
            ),
            "energy_distance": _energy_distance(
                standardized_reference,
                standardized_generated,
                reference_weights,
                generated_weights,
                settings,
            ),
            "sliced_wasserstein": _sliced_wasserstein(
                standardized_reference,
                standardized_generated,
                reference_weights,
                generated_weights,
                settings,
            ),
            "c2st": _c2st(
                standardized_reference,
                standardized_generated,
                reference_weights,
                generated_weights,
                settings,
            ),
        },
        "correlations": _correlation_metrics(
            reference,
            generated,
            reference_weights,
            generated_weights,
        ),
        "marginal": marginal,
        "tails": _tail_metrics(
            train,
            reference,
            generated,
            train_weights,
            reference_weights,
            generated_weights,
            settings,
        ),
    }


def _load_root(
    source: str | Path,
    *,
    tree_name: str,
    generated_weight_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import uproot
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Generated evaluation requires uproot") from error
    root_path = Path(source)
    if not root_path.is_file():
        raise FileNotFoundError(root_path)
    with uproot.open(root_path) as root_file:
        if tree_name not in root_file:
            raise KeyError(f"Missing tree {tree_name!r}: {root_path}")
        tree = root_file[tree_name]
        branches = {str(name) for name in tree.keys()}
        missing = sorted(set(FEATURES_8D) - branches)
        if missing:
            raise ValueError(f"Missing physical features in {root_path}: {missing}")
        requested = list(FEATURES_8D)
        use_branch = generated_weight_mode != "uniform" and "w" in branches
        if use_branch:
            requested.append("w")
        arrays = tree.arrays(requested, library="np")
    if isinstance(arrays, np.ndarray) and arrays.dtype.names:
        values = {name: arrays[name] for name in requested}
    else:
        values = {name: arrays[name] for name in requested}
    matrix = np.column_stack([values[name] for name in FEATURES_8D])
    if use_branch:
        weights = np.asarray(values["w"], dtype=np.float64)
        source_label = "ROOT branch w"
    else:
        weights = np.ones(len(matrix), dtype=np.float64)
        source_label = "uniform"
    return matrix, weights, source_label


def evaluate_generated_root_files(
    *,
    dataset_id: str,
    train_reference_root: str | Path,
    reference_root: str | Path,
    generated_root: str | Path,
    output_directory: str | Path,
    reference_split: str = "validation",
    tree_name: str = "nt",
    generated_weight_mode: str = "auto",
    settings: EvaluationSettings | None = None,
) -> dict[str, Any]:
    """Load physical ROOT files, evaluate them, and write one immutable report."""

    if reference_split not in {"validation", "test"}:
        raise ValueError("reference_split must be validation or test")
    if generated_weight_mode not in {"auto", "branch", "uniform"}:
        raise ValueError("generated_weight_mode must be auto, branch, or uniform")
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"Evaluation output already exists: {destination}")
    train, train_weights, train_weight_source = _load_root(
        train_reference_root, tree_name=tree_name, generated_weight_mode="branch"
    )
    reference, reference_weights, reference_weight_source = _load_root(
        reference_root, tree_name=tree_name, generated_weight_mode="branch"
    )
    if train_weight_source != "ROOT branch w":
        raise ValueError("Training reference ROOT must contain the FLUKA w branch")
    if reference_weight_source != "ROOT branch w":
        raise ValueError("Evaluation reference ROOT must contain the FLUKA w branch")
    generated, generated_weights, generated_weight_source = _load_root(
        generated_root,
        tree_name=tree_name,
        generated_weight_mode=generated_weight_mode,
    )
    if generated_weight_mode == "branch" and generated_weight_source != "ROOT branch w":
        raise ValueError("Generated ROOT has no w branch but branch mode was required")
    evaluation = evaluate_generated_arrays(
        train=train,
        reference=reference,
        generated=generated,
        train_weights=train_weights,
        reference_weights=reference_weights,
        generated_weights=generated_weights,
        settings=settings or EvaluationSettings(),
    )
    report = {
        "status": "complete",
        "dataset_id": dataset_id,
        "reference_split": reference_split,
        "inputs": {
            "train_reference_root": str(Path(train_reference_root).resolve()),
            "reference_root": str(Path(reference_root).resolve()),
            "generated_root": str(Path(generated_root).resolve()),
            "tree_name": tree_name,
            "weight_sources": {
                "train": train_weight_source,
                "reference": reference_weight_source,
                "generated": generated_weight_source,
            },
        },
        "evaluation": evaluation,
    }
    generation_manifest_path = Path(generated_root).parent / "generation_manifest.json"
    generation_manifest = (
        json.loads(generation_manifest_path.read_text())
        if generation_manifest_path.is_file()
        else {}
    )
    partial = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(partial)
    try:
        temporary_plots = write_generated_evaluation_plots(
            reference=reference,
            generated=generated,
            reference_weights=reference_weights,
            generated_weights=generated_weights,
            report=report,
            output_directory=partial,
            context={
                "year": generation_manifest.get("dataset", {}).get("year", "Unknown"),
                "preprocessing": generation_manifest.get("preprocessing", "?"),
                "reference_split": reference_split,
            },
        )
        report["plots"] = [
            str((destination / Path(path).relative_to(partial)).resolve())
            for path in temporary_plots
        ]
        write_json_atomic(partial / "generated_evaluation.json", report)
        write_json_atomic(
            partial / "_SUCCESS.json",
            {"stage": "generated_evaluation", "reference_split": reference_split},
        )
        partial.replace(destination)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return report
