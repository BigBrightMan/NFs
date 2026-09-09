"""Read-only diagnostics for one completed Model 4 training run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..manifest import write_json_atomic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_history(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Training history is empty")
    columns: dict[str, np.ndarray] = {}
    for name in rows[0]:
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(values).all():
            raise ValueError(f"Training history column {name!r} is not finite")
        columns[name] = values
    for required in ("epoch", "train_loss", "validation_loss", "learning_rate"):
        if required not in columns:
            raise ValueError(f"Training history is missing {required!r}")
    epochs = columns["epoch"]
    if np.any(np.diff(epochs) != 1):
        raise ValueError("Training history epochs are not consecutive")
    return columns


def _first_column(
    columns: dict[str, np.ndarray], *names: str
) -> np.ndarray | None:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def _optional_stat(
    columns: dict[str, np.ndarray], names: tuple[str, ...], operation: str
) -> float | None:
    values = _first_column(columns, *names)
    if values is None:
        return None
    if operation == "maximum":
        return float(np.max(values))
    if operation == "median":
        return float(np.median(values))
    raise ValueError(operation)


def _plot_diagnostics(
    *,
    columns: dict[str, np.ndarray],
    config: dict[str, Any],
    best_epoch: int,
    destination: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Training plots require matplotlib") from error

    epochs = columns["epoch"]
    train = columns["train_loss"]
    validation = columns["validation_loss"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(epochs, train, label="Train weighted NLL")
    axes[0, 0].plot(epochs, validation, label="Validation weighted NLL")
    axes[0, 0].axvline(best_epoch, color="tab:green", linestyle="--", label="Best")
    axes[0, 0].set_ylabel("Weighted NLL")
    axes[0, 0].legend()

    axes[0, 1].axhline(0.0, color="0.5", linewidth=1)
    axes[0, 1].plot(epochs, validation - train, color="tab:purple")
    axes[0, 1].set_ylabel("Validation - train NLL")

    axes[1, 0].plot(epochs, columns["learning_rate"], color="tab:orange")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Learning rate")
    axes[1, 0].set_xlabel("Epoch")

    diagnostic_axis = axes[1, 1]
    plotted = False
    for names, label, color in (
        (("train_gradient_norm", "gradient_norm"), "Gradient norm", "tab:red"),
        (
            ("train_batch_weight_mean_cv", "batch_weight_mean_cv"),
            "Batch weight CV",
            "tab:blue",
        ),
        (
            ("train_number_nonfinite_batches", "number_nonfinite_batches"),
            "Non-finite batches",
            "tab:brown",
        ),
    ):
        values = _first_column(columns, *names)
        if values is not None:
            diagnostic_axis.plot(epochs, values, label=label, color=color)
            plotted = True
    diagnostic_axis.set_xlabel("Epoch")
    diagnostic_axis.set_ylabel("Optimization diagnostics")
    if plotted:
        diagnostic_axis.legend()
    else:
        diagnostic_axis.text(0.5, 0.5, "No optional diagnostics recorded", ha="center")

    dataset = config.get("dataset", {})
    preprocessing = config.get("preprocessing", {})
    model = config.get("model", {})
    year = dataset.get("year", "Unknown year")
    pipeline = preprocessing.get("id", "?")
    ablation = model.get("ablation", "unknown")
    figure.suptitle(
        f"{year} data · Model 4 · Preprocess {pipeline} · "
        f"{ablation} · Training validation"
    )
    figure.text(
        0.012,
        0.988,
        "FlashSim in progress",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#1f5da8",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def evaluate_training_run(
    run_directory: str | Path,
    output_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one completed run and write immutable training diagnostics."""

    run = Path(run_directory).resolve()
    destination = (
        Path(output_directory).resolve()
        if output_directory is not None
        else run / "training_validation"
    )
    if destination.exists():
        raise FileExistsError(f"Training diagnostics already exist: {destination}")
    summary_path = run / "training_summary.json"
    config_path = run / "config_resolved.json"
    success_path = run / "_SUCCESS.json"
    for path in (summary_path, config_path, success_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text())
    success = json.loads(success_path.read_text())
    if summary.get("status") != "complete" or success.get("stage") != "training":
        raise ValueError("Run does not have a complete training contract")

    columns = _load_history(run / "training" / "training_history.csv")
    epochs = columns["epoch"].astype(np.int64)
    last_epoch = int(epochs[-1])
    best_index = int(np.argmin(columns["validation_loss"]))
    best_epoch = int(epochs[best_index])
    if int(summary["last_epoch"]) != last_epoch:
        raise ValueError("training_summary last_epoch disagrees with history")
    if int(summary["best_epoch"]) != best_epoch:
        raise ValueError("training_summary best_epoch disagrees with history")

    checkpoint_directory = run / "checkpoints"
    checkpoints = {}
    for name in ("best_model.pt", "last_model.pt"):
        path = checkpoint_directory / name
        checkpoints[name] = {
            "exists": path.is_file(),
            "path": str(path),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    if not all(item["exists"] for item in checkpoints.values()):
        raise FileNotFoundError(
            "Completed run is missing best_model.pt or last_model.pt"
        )
    periodic = sorted(checkpoint_directory.glob("epoch_*_model.pt"))

    train = columns["train_loss"]
    validation = columns["validation_loss"]
    learning_rate = columns["learning_rate"]
    report = {
        "status": "complete",
        "stage": "training_validation",
        "run_directory": str(run),
        "run_identity": {
            "dataset_id": config.get("dataset", {}).get("dataset_id"),
            "year": config.get("dataset", {}).get("year"),
            "preprocessing": config.get("preprocessing", {}).get("id"),
            "ablation": config.get("model", {}).get("ablation"),
            "training_seed": config.get("training", {}).get("seed"),
        },
        "epochs": {
            "first": int(epochs[0]),
            "last": last_epoch,
            "count": int(len(epochs)),
            "best": best_epoch,
        },
        "loss": {
            "initial_train": float(train[0]),
            "initial_validation": float(validation[0]),
            "last_train": float(train[-1]),
            "last_validation": float(validation[-1]),
            "best_validation": float(validation[best_index]),
            "train_at_best": float(train[best_index]),
            "gap_at_best": float(validation[best_index] - train[best_index]),
            "last_gap": float(validation[-1] - train[-1]),
            "validation_improvement": float(validation[0] - validation[best_index]),
        },
        "optimization": {
            "initial_learning_rate": float(learning_rate[0]),
            "final_learning_rate": float(learning_rate[-1]),
            "minimum_learning_rate": float(np.min(learning_rate)),
            "learning_rate_changes": int(np.count_nonzero(np.diff(learning_rate))),
            "maximum_gradient_norm": _optional_stat(
                columns, ("train_gradient_norm", "gradient_norm"), "maximum"
            ),
            "median_gradient_norm": _optional_stat(
                columns, ("train_gradient_norm", "gradient_norm"), "median"
            ),
            "maximum_batch_weight_mean_cv": _optional_stat(
                columns,
                ("train_batch_weight_mean_cv", "batch_weight_mean_cv"),
                "maximum",
            ),
            "total_nonfinite_batches": None,
        },
        "checkpoints": {
            **checkpoints,
            "periodic": [str(path) for path in periodic],
        },
        "test_set_used": False,
    }
    nonfinite = _first_column(
        columns, "train_number_nonfinite_batches", "number_nonfinite_batches"
    )
    if nonfinite is not None:
        report["optimization"]["total_nonfinite_batches"] = int(np.sum(nonfinite))
    _plot_diagnostics(
        columns=columns,
        config=config,
        best_epoch=best_epoch,
        destination=destination / "training_diagnostics.png",
    )
    write_json_atomic(destination / "training_diagnostics.json", report)
    write_json_atomic(
        destination / "_SUCCESS.json",
        {"stage": "training_validation", "best_epoch": best_epoch},
    )
    return report
