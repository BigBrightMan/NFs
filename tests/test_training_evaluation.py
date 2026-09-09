from __future__ import annotations

import csv
import json

import pytest

from flashsim_nf.evaluation.training import evaluate_training_run


def _write_run(run) -> None:
    (run / "training").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    config = {
        "dataset": {"dataset_id": "fluka_test", "year": 2025},
        "preprocessing": {"id": "B"},
        "model": {"ablation": "drop_ze"},
        "training": {"seed": 42},
    }
    (run / "config_resolved.json").write_text(json.dumps(config))
    (run / "training_summary.json").write_text(
        json.dumps({"status": "complete", "last_epoch": 3, "best_epoch": 2})
    )
    (run / "_SUCCESS.json").write_text(
        json.dumps({"stage": "training", "last_epoch": 3})
    )
    with (run / "training" / "training_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
                "learning_rate",
                "train_gradient_norm",
                "train_batch_weight_mean_cv",
                "train_number_nonfinite_batches",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "epoch": 1,
                    "train_loss": 7,
                    "validation_loss": 8,
                    "learning_rate": 0.001,
                    "train_gradient_norm": 4,
                    "train_batch_weight_mean_cv": 0.05,
                    "train_number_nonfinite_batches": 0,
                },
                {
                    "epoch": 2,
                    "train_loss": 5,
                    "validation_loss": 5.5,
                    "learning_rate": 0.001,
                    "train_gradient_norm": 3,
                    "train_batch_weight_mean_cv": 0.04,
                    "train_number_nonfinite_batches": 0,
                },
                {
                    "epoch": 3,
                    "train_loss": 4.8,
                    "validation_loss": 5.7,
                    "learning_rate": 0.0005,
                    "train_gradient_norm": 2,
                    "train_batch_weight_mean_cv": 0.03,
                    "train_number_nonfinite_batches": 0,
                },
            ]
        )
    (run / "checkpoints" / "best_model.pt").write_bytes(b"best")
    (run / "checkpoints" / "last_model.pt").write_bytes(b"last")
    (run / "checkpoints" / "epoch_0002_model.pt").write_bytes(b"periodic")


def test_training_diagnostics_are_complete_and_immutable(tmp_path) -> None:
    run = tmp_path / "run"
    _write_run(run)
    report = evaluate_training_run(run)
    output = run / "training_validation"
    assert report["epochs"]["best"] == 2
    assert report["loss"]["gap_at_best"] == 0.5
    assert report["optimization"]["learning_rate_changes"] == 1
    assert report["test_set_used"] is False
    assert (output / "training_diagnostics.png").is_file()
    assert (output / "training_diagnostics.json").is_file()
    assert (output / "_SUCCESS.json").is_file()
    with pytest.raises(FileExistsError):
        evaluate_training_run(run)
