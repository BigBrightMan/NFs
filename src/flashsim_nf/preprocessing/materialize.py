"""Materialize native NFs preprocessing artifacts without changing frozen splits."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_yaml
from ..data import write_root_arrays
from ..experiments.training_config import sha256_file
from ..manifest import write_json_atomic
from .pipelines import FeaturePreprocessor, build_preprocessor

PHYSICAL_8D = ("x", "y", "z", "E", "pz", "px", "py", "t")
FIT_9D = (*PHYSICAL_8D, "w")
SOURCE_METADATA = ("run", "event", "id", "generation")


def _read_raw(path: Path, *, tree_name: str = "nt") -> dict[str, np.ndarray]:
    try:
        import uproot
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Preparing native pipelines requires uproot") from error
    requested = (*SOURCE_METADATA, *FIT_9D)
    with uproot.open(path) as source:
        tree = source[tree_name]
        arrays = tree.arrays(list(requested), library="np")
        rows = int(tree.num_entries)
    values = {name: np.asarray(arrays[name]) for name in requested}
    if any(len(value) != rows for value in values.values()):
        raise ValueError(f"Misaligned raw ROOT branches: {path}")
    matrix = np.column_stack([values[name] for name in FIT_9D])
    if not np.isfinite(matrix).all() or np.any(values["w"] <= 0):
        raise ValueError(f"Invalid raw values: {path}")
    return values


def prepare_native_pipeline(
    *,
    pipeline: str,
    dataset_config: str | Path,
    environment: str,
    data_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit a native experimental pipeline on train and transform train/validation."""

    if environment not in {"local", "cern"}:
        raise ValueError("environment must be local or cern")
    pipeline = str(pipeline).upper()
    if pipeline not in {"D", "E"}:
        raise ValueError("Native materialization currently supports D or E")
    dataset_path = Path(dataset_config).resolve()
    dataset = load_yaml(dataset_path)
    dataset_id = str(dataset["dataset_id"])
    prepared = Path(dataset["legacy_prepared_paths"][environment]).resolve()
    output = (
        Path(data_root).resolve()
        / "campaigns"
        / dataset_id
        / "prepared_data"
        / f"preprocessing_{pipeline}"
    )
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(partial)
    raw_paths = {
        split: prepared / "split" / f"{split}_rawfeature.root"
        for split in ("train", "validation")
    }
    for path in raw_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    counts = dataset["split"]["counts"]
    values = {split: _read_raw(path) for split, path in raw_paths.items()}
    for split in values:
        if len(values[split]["w"]) != int(counts[split]):
            raise ValueError(f"{split} row count does not match frozen dataset config")
    preprocessor: FeaturePreprocessor = build_preprocessor(pipeline, FIT_9D)
    preprocessor.fit(np.column_stack([values["train"][name] for name in FIT_9D]))
    partial.mkdir(parents=True, exist_ok=False)
    try:
        artifact = partial / "preprocessor.json"
        preprocessor.save(artifact)
        dimension = partial / "8d"
        dimension.mkdir()
        (dimension / "feature_order.json").write_text(
            json.dumps(list(PHYSICAL_8D), indent=2) + "\n"
        )
        output_files: dict[str, str] = {}
        for split, raw in values.items():
            transformed = preprocessor.transform(
                np.column_stack([raw[name] for name in FIT_9D])
            )
            arrays = {name: raw[name] for name in SOURCE_METADATA}
            arrays.update(
                {
                    name: transformed[:, index].astype(np.float32)
                    for index, name in enumerate(PHYSICAL_8D)
                }
            )
            path = write_root_arrays(dimension / f"{split}_preprocessed.root", arrays)
            output_files[split] = str(output / "8d" / path.name)
        manifest = {
            "status": "complete",
            "format": "flashsim_nf.preprocessing_materialization",
            "format_version": 1,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset["dataset_fingerprint"],
            "split_id": dataset["split"]["id"],
            "pipeline": pipeline,
            "fit_split": "train",
            "transformed_splits": ["train", "validation"],
            "test_loaded": False,
            "feature_order": list(PHYSICAL_8D),
            "source_dataset_config": {
                "path": str(dataset_path),
                "sha256": sha256_file(dataset_path),
            },
            "raw_inputs": {key: str(value) for key, value in raw_paths.items()},
            "outputs": output_files,
        }
        write_json_atomic(partial / "preparation_manifest.json", manifest)
        write_json_atomic(
            partial / "_SUCCESS.json",
            {"stage": f"preprocessing_{pipeline}", "dataset_id": dataset_id},
        )
        if output.exists():
            backup = output.with_name(f"{output.name}.before_overwrite")
            if backup.exists():
                raise FileExistsError(backup)
            output.replace(backup)
            manifest["replaced_output_backup"] = str(backup)
        partial.replace(output)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    manifest["output_directory"] = str(output)
    return manifest


def prepare_pipeline_d(
    *,
    dataset_config: str | Path,
    environment: str,
    data_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Backward-compatible wrapper for Pipeline D materialization."""

    return prepare_native_pipeline(
        pipeline="D",
        dataset_config=dataset_config,
        environment=environment,
        data_root=data_root,
        overwrite=overwrite,
    )
