"""Orchestrate immutable post-training generation and evaluation stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..evaluation import EvaluationSettings, evaluate_generated_root_files
from ..evaluation.training import evaluate_training_run
from ..generation import generate_model4_from_config
from ..manifest import write_json_atomic


def execute_post_training_config(config_path: str | Path) -> dict[str, Any]:
    """Execute diagnostics, generation, and validation/test evaluation."""

    source = Path(config_path).resolve()
    config = yaml.safe_load(source.read_text())
    if not isinstance(config, dict):
        raise TypeError("Post-training config must contain a mapping")
    run = Path(config["training_run"])
    training_config = json.loads((run / "config_resolved.json").read_text())
    training_validation = run / "training_validation"
    if not training_validation.exists():
        evaluate_training_run(run)
    generation_directory = Path(config["output"]["directory"])
    generation_manifest = generation_directory / "generation_manifest.json"
    generation_success = generation_directory / "_SUCCESS.json"
    if generation_directory.exists():
        if not generation_manifest.is_file() or not generation_success.is_file():
            raise RuntimeError(
                "Generation directory exists without a complete immutable stage"
            )
        generation = json.loads(generation_manifest.read_text())
    else:
        generation = generate_model4_from_config(source)
    purpose = config["generation"]["purpose"]
    evaluation = None
    if purpose in {"validation", "test"}:
        prepared = Path(training_config["resolution"]["prepared_directory"])
        evaluation_directory = generation_directory / "evaluation"
        if evaluation_directory.exists():
            evaluation_success = evaluation_directory / "_SUCCESS.json"
            evaluation_report = evaluation_directory / "generated_evaluation.json"
            if not evaluation_success.is_file() or not evaluation_report.is_file():
                raise RuntimeError(
                    "Evaluation directory exists without a complete immutable stage"
                )
            evaluation = json.loads(evaluation_report.read_text())
        else:
            evaluation = evaluate_generated_root_files(
                dataset_id=config["dataset"]["dataset_id"],
                train_reference_root=prepared / "split/train_rawfeature.root",
                reference_root=prepared / "split" / f"{purpose}_rawfeature.root",
                generated_root=generation["outputs"]["w1"],
                output_directory=evaluation_directory,
                reference_split=purpose,
                generated_weight_mode="uniform",
                settings=EvaluationSettings(
                    random_seed=int(config["generation"]["seed"])
                ),
            )
    summary = {
        "status": "complete",
        "purpose": purpose,
        "run_directory": str(run.resolve()),
        "generation_config": str(source),
        "generation_manifest": str(
            Path(config["output"]["directory"]) / "generation_manifest.json"
        ),
        "evaluation": (
            str(Path(config["output"]["directory"]) / "evaluation")
            if evaluation
            else None
        ),
        "test_data_used": purpose == "test",
    }
    summary_path = generation_directory / "post_training_summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing != summary:
            raise RuntimeError("Existing post-training summary does not match")
    else:
        write_json_atomic(summary_path, summary)
    return summary
