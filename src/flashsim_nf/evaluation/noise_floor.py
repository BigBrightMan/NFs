"""FLUKA-vs-FLUKA reference noise floor for Model 4 validation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import write_json_atomic
from .generated import FEATURES_8D, EvaluationSettings, evaluate_generated_arrays
from .quality import selection_metrics


def _read(
    path: Path, *, tree_name: str
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    try:
        import uproot
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Noise-floor evaluation requires uproot") from error
    metadata_names = ("run", "event", "id", "generation")
    requested = (*FEATURES_8D, "w", *metadata_names)
    with uproot.open(path) as source:
        arrays = source[tree_name].arrays(list(requested), library="np")
    matrix = np.column_stack([arrays[name] for name in FEATURES_8D]).astype(np.float64)
    weights = np.asarray(arrays["w"], dtype=np.float64)
    metadata = {name: np.asarray(arrays[name]) for name in metadata_names}
    return matrix, weights, metadata


def _stable_half(metadata: dict[str, np.ndarray]) -> np.ndarray:
    """Deterministic 64-bit mix of immutable row identity; True selects half A."""

    value = np.zeros(len(metadata["event"]), dtype=np.uint64)
    for index, name in enumerate(("run", "event", "id", "generation"), start=1):
        component = np.asarray(metadata[name], dtype=np.int64).view(np.uint64)
        value ^= component + np.uint64(0x9E3779B97F4A7C15 * index % 2**64)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
    return (value & np.uint64(1)) == 0


def _summary(records: list[dict[str, float]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in records[0]:
        values = np.asarray([record[name] for record in records], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=0)),
            "q05": float(np.quantile(values, 0.05)),
            "q50": float(np.quantile(values, 0.50)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return result


def _noise_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    metrics = selection_metrics(evaluation)
    metrics["covariance_frobenius_distance"] = float(
        evaluation["global_multivariate"]["covariance_frobenius_distance"]
    )
    return metrics


def compute_reference_noise_floor(
    *,
    dataset_id: str,
    train_root: str | Path,
    validation_root: str | Path,
    output: str | Path,
    repeats: int = 20,
    row_matched_rows: int | None = None,
    random_seed: int = 91556,
    tree_name: str = "nt",
) -> dict[str, Any]:
    """Evaluate two independent views of the same weighted FLUKA distribution."""

    if repeats < 2:
        raise ValueError("repeats must be at least two")
    train, train_weights, _ = _read(Path(train_root), tree_name=tree_name)
    validation, validation_weights, metadata = _read(
        Path(validation_root), tree_name=tree_name
    )
    half = _stable_half(metadata)
    if min(int(half.sum()), int((~half).sum())) < 4:
        raise ValueError("Stable validation half split is too small")
    base_settings = EvaluationSettings(random_seed=random_seed)
    half_evaluation = evaluate_generated_arrays(
        train=train,
        reference=validation[half],
        generated=validation[~half],
        train_weights=train_weights,
        reference_weights=validation_weights[half],
        generated_weights=validation_weights[~half],
        settings=base_settings,
    )
    target_rows = int(row_matched_rows or len(validation))
    rng = np.random.default_rng(random_seed)
    probability = validation_weights / validation_weights.sum()
    repeated: list[dict[str, float]] = []
    for repeat in range(repeats):
        left = rng.choice(
            len(validation), size=target_rows, replace=True, p=probability
        )
        right = rng.choice(
            len(validation), size=target_rows, replace=True, p=probability
        )
        uniform = np.ones(target_rows, dtype=np.float64)
        evaluation = evaluate_generated_arrays(
            train=train,
            reference=validation[left],
            generated=validation[right],
            train_weights=train_weights,
            reference_weights=uniform,
            generated_weights=uniform,
            settings=EvaluationSettings(random_seed=random_seed + repeat + 1),
        )
        repeated.append(_noise_metrics(evaluation))
    report = {
        "status": "complete",
        "format": "flashsim_nf.reference_noise_floor",
        "format_version": 1,
        "dataset_id": dataset_id,
        "reference_split": "validation",
        "test_data_used": False,
        "method": {
            "stable_half_split": "hash(run,event,id,generation)",
            "repeated_sampling": "independent weighted bootstrap with replacement",
            "repeats": repeats,
            "rows_per_sample": target_rows,
            "random_seed": random_seed,
        },
        "inputs": {
            "train_root": str(Path(train_root).resolve()),
            "validation_root": str(Path(validation_root).resolve()),
        },
        "stable_half": {
            "rows_a": int(half.sum()),
            "rows_b": int((~half).sum()),
            "metrics": _noise_metrics(half_evaluation),
        },
        "bootstrap": {"summary": _summary(repeated), "raw_metrics": repeated},
    }
    destination = Path(output).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    write_json_atomic(destination, report)
    return report
