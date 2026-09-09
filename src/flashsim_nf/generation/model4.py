"""Complete Model 4 inference from an immutable NFs training run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..data import write_root_arrays
from ..guards import PHYSICAL_FEATURES, ReferenceData, guard_rejection_masks
from ..manifest import write_json_atomic
from ..models.flows import FlowConfig, build_flow
from ..preprocessing import load_legacy_preprocessor
from ..reconstruction import ScoringPlane, fit_scoring_plane, reconstruct_drop_z_e

FORMAT = "flashsim_nf.model4_generation_config"
VERSION = 1


def _validate_guard_role(guard: dict[str, Any], *, purpose: str) -> None:
    """Enforce the reference scope allowed for each scientific stage."""

    if purpose in {"validation", "test"}:
        expected = {
            "fit_scope": "train",
            "source_splits": ["train"],
            "usage_role": "validation_and_selection",
            "selection_allowed": True,
        }
        role = "train-reference"
    elif purpose == "muondis":
        expected = {
            "fit_scope": "all_clean_splits",
            "source_splits": ["train", "validation", "test"],
            "usage_role": "production_only",
            "selection_allowed": False,
        }
        role = "all-clean-FLUKA production"
    else:  # pragma: no cover - guarded by the public resolver
        raise ValueError(f"Unsupported generation purpose: {purpose}")
    mismatches = {
        key: {"expected": value, "observed": guard.get(key)}
        for key, value in expected.items()
        if guard.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{purpose} requires the {role} guard; role mismatch: {mismatches}"
        )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weight_summary(paths: list[str | Path], tree_name: str) -> dict[str, Any]:
    try:
        import uproot
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Reference normalization requires uproot") from error
    rows = 0
    total = 0.0
    square_total = 0.0
    minimum = math.inf
    maximum = -math.inf
    files: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        with uproot.open(path) as source:
            values = np.asarray(
                source[tree_name]["w"].array(library="np"), dtype=np.float64
            )
        if len(values) == 0 or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Invalid reference weights: {path}")
        item = {"path": str(path), "rows": len(values), "sum_w": float(values.sum())}
        files.append(item)
        rows += len(values)
        total += float(values.sum(dtype=np.float64))
        square_total += float(np.square(values).sum(dtype=np.float64))
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    effective = total * total / square_total
    return {
        "rows": rows,
        "sum_w": total,
        "mean_w": total / rows,
        "minimum_w": minimum,
        "maximum_w": maximum,
        "effective_sample_size": effective,
        "effective_sample_fraction": effective / rows,
        "files": files,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON mapping: {path}")
    return value


def resolve_generation_config(
    *,
    run_directory: str | Path,
    purpose: str,
    guard_artifact: str | Path,
    number_events: int | None = None,
    generation_seed: int | None = None,
    batch_size: int = 50_000,
    maximum_attempt_multiplier: float = 2.0,
    device: str = "auto",
    frozen_selection: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve checkpoint, inverse transform, physics, guard, and output paths."""

    run = Path(run_directory).resolve()
    if purpose not in {"validation", "test", "muondis"}:
        raise ValueError("purpose must be validation, test, or muondis")
    if purpose in {"test", "muondis"} and frozen_selection is None:
        raise ValueError(
            "Final test and MuonDIS generation require a frozen-selection artifact"
        )
    resolved_path = run / "config_resolved.json"
    success_path = run / "_SUCCESS.json"
    checkpoint = run / "checkpoints" / "best_model.pt"
    for path in (resolved_path, success_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    training = _read_json(resolved_path)
    dataset = training["dataset"]
    preprocessing = training["preprocessing"]
    if training["model"].get("ablation") != "drop_ze":
        raise ValueError("Generation currently supports Model 4 drop_ze only")
    if training["model"].get("objective") != "weighted_nll":
        raise ValueError("Checkpoint is not a Model 4 weighted-density run")
    if training["model"].get("weight_is_input_feature") is not False:
        raise ValueError("Model 4 generation forbids w as an NF feature")
    prepared = Path(training["resolution"]["prepared_directory"]).resolve()
    split_counts = dataset["split_counts"]
    default_rows = split_counts.get(purpose)
    rows = int(number_events if number_events is not None else default_rows or 0)
    if rows <= 0:
        raise ValueError("number_events must be supplied and positive")
    seed_defaults = {"validation": 1556, "test": 2556, "muondis": 3556}
    seed = int(
        generation_seed if generation_seed is not None else seed_defaults[purpose]
    )
    if purpose == "validation":
        reference_splits = ["validation"]
    elif purpose == "test":
        reference_splits = ["test"]
    else:
        reference_splits = ["train", "validation", "test"]
    references = [
        prepared / "split" / f"{name}_rawfeature.root" for name in reference_splits
    ]
    for path in references:
        if not path.is_file():
            raise FileNotFoundError(path)
    guard_path = Path(guard_artifact).resolve()
    guard = _read_json(guard_path)
    if guard.get("dataset_id") != dataset["dataset_id"]:
        raise ValueError("Guard dataset_id does not match the training run")
    if guard.get("dataset_fingerprint") != dataset["dataset_fingerprint"]:
        raise ValueError("Guard dataset fingerprint does not match the training run")
    if guard.get("split_id") != dataset["split_id"]:
        raise ValueError("Guard split ID does not match the training run")
    _validate_guard_role(guard, purpose=purpose)
    plane = fit_scoring_plane(training["data"]["train"]["raw_weight_path"])
    output = run / "samples" / purpose / f"generation_seed_{seed}" / "guard_reject_v2"
    if output.exists():
        raise FileExistsError(f"Generation output already exists: {output}")
    frozen = Path(frozen_selection).resolve() if frozen_selection else None
    if frozen and not frozen.is_file():
        raise FileNotFoundError(frozen)
    if frozen:
        selection = _read_json(frozen)
        if selection.get("status") != "frozen":
            raise ValueError("Selection artifact is not frozen")
        if selection.get("test_data_used") is not False:
            raise ValueError("Selection artifact must not use test data")
        if selection.get("winner", {}).get("training_run") != str(run):
            raise ValueError(
                "Selected winner does not match the requested training run"
            )
    return {
        "format": FORMAT,
        "format_version": VERSION,
        "dataset": dataset,
        "training_run": str(run),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "preprocessing": {
            "id": preprocessing["id"],
            "metadata": preprocessing["metadata"],
            "metadata_sha256": preprocessing["metadata_sha256"],
            "feature_order": training["data"]["feature_order"],
        },
        "model": training["model"],
        "reconstruction": {
            "type": "drop_z_e",
            "muon_mass_gev": 0.1056583755,
            "scoring_plane": plane.to_dict(),
        },
        "guard": {
            "id": "guard_reject_v2",
            "artifact": str(guard_path),
            "artifact_sha256": _sha256(guard_path),
            "maximum_rejection_fraction": 0.05,
            "detector_bounds": None,
            "detector_bounds_status": "disabled_pending_authoritative_bounds",
        },
        "generation": {
            "purpose": purpose,
            "number_events": rows,
            "seed": seed,
            "batch_size": int(batch_size),
            "maximum_attempt_multiplier": float(maximum_attempt_multiplier),
            "device": device,
            "tree_name": "nt",
        },
        "normalization": {
            "scope": "+".join(reference_splits),
            "reference_weight_files": [str(path.resolve()) for path in references],
        },
        "frozen_selection": (
            {"path": str(frozen), "sha256": _sha256(frozen)} if frozen else None
        ),
        "output": {"directory": str(output)},
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("format") != FORMAT or config.get("format_version") != VERSION:
        raise ValueError("Unsupported Model 4 generation config")
    artifacts = (
        ("checkpoint", config["checkpoint"]["path"], config["checkpoint"]["sha256"]),
        (
            "preprocessing",
            config["preprocessing"]["metadata"],
            config["preprocessing"]["metadata_sha256"],
        ),
        ("guard", config["guard"]["artifact"], config["guard"]["artifact_sha256"]),
    )
    for name, path, expected in artifacts:
        if _sha256(path) != expected:
            raise ValueError(f"{name} artifact SHA-256 mismatch")
    selection = config.get("frozen_selection")
    if selection and _sha256(selection["path"]) != selection["sha256"]:
        raise ValueError("Frozen-selection artifact SHA-256 mismatch")
    guard = _read_json(Path(config["guard"]["artifact"]))
    _validate_guard_role(guard, purpose=config["generation"]["purpose"])


def generate_model4_from_config(config_path: str | Path) -> dict[str, Any]:
    """Sample, invert, reconstruct, reject, and write physical Model 4 ROOT files."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Model 4 generation requires torch and nflows") from error
    config = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(config, dict):
        raise TypeError("Generation config must contain a mapping")
    _validate_config(config)
    requested = int(config["generation"]["number_events"])
    batch_size = int(config["generation"]["batch_size"])
    maximum_attempts = math.ceil(
        requested * float(config["generation"]["maximum_attempt_multiplier"])
    )
    requested_device = str(config["generation"].get("device", "auto"))
    device = torch.device(
        "cuda"
        if requested_device == "auto" and torch.cuda.is_available()
        else "cpu"
        if requested_device == "auto"
        else requested_device
    )
    seed = int(config["generation"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    checkpoint = torch.load(
        config["checkpoint"]["path"], map_location=device, weights_only=False
    )
    if checkpoint.get("format") != "flashsim_nf.full_training_state":
        raise ValueError("Unsupported checkpoint format")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        raise ValueError("Checkpoint has no embedded resolved training config")
    if checkpoint_config.get("dataset") != config["dataset"]:
        raise ValueError("Checkpoint dataset identity does not match generation config")
    checkpoint_preprocessing = checkpoint_config.get("preprocessing", {})
    for key in ("id", "metadata_sha256"):
        if checkpoint_preprocessing.get(key) != config["preprocessing"].get(key):
            raise ValueError(f"Checkpoint preprocessing {key} mismatch")
    if checkpoint_config.get("data", {}).get("feature_order") != config[
        "preprocessing"
    ].get("feature_order"):
        raise ValueError("Checkpoint feature order does not match generation config")
    feature_order = tuple(config["preprocessing"]["feature_order"])
    model = build_flow(
        FlowConfig.from_mapping(config["model"], input_dim=len(feature_order))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    preprocessor = load_legacy_preprocessor(
        config["preprocessing"]["metadata"],
        feature_order=feature_order,
        expected_pipeline=config["preprocessing"]["id"],
    )
    plane = ScoringPlane(**config["reconstruction"]["scoring_plane"])
    guard = _read_json(Path(config["guard"]["artifact"]))
    accepted_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in PHYSICAL_FEATURES
    }
    attempted = 0
    accepted = 0
    reason_counts: dict[str, int] = {}
    with torch.no_grad():
        while accepted < requested and attempted < maximum_attempts:
            proposal_rows = min(batch_size, maximum_attempts - attempted)
            model_space = model.sample(proposal_rows).detach().cpu().numpy()
            physical_selected = preprocessor.inverse_transform(model_space)
            physical = reconstruct_drop_z_e(
                physical_selected,
                feature_order=feature_order,
                scoring_plane=plane,
                muon_mass_gev=float(config["reconstruction"]["muon_mass_gev"]),
            )
            matrix = np.column_stack([physical[name] for name in PHYSICAL_FEATURES])
            finite = np.isfinite(matrix).all(axis=1)
            domain = finite & (physical["E"] > 10.0) & (physical["pz"] > 0.0)
            reference = ReferenceData(
                features=physical,
                weights=np.ones(proposal_rows, dtype=np.float64),
                label="generated_proposals",
            )
            robust_rejected, reasons = guard_rejection_masks(reference, guard)
            keep = domain & ~robust_rejected
            remaining = requested - accepted
            accepted_positions = np.flatnonzero(keep)
            consumed = (
                int(accepted_positions[remaining - 1]) + 1
                if len(accepted_positions) >= remaining
                else proposal_rows
            )
            keep = keep[:consumed]
            finite = finite[:consumed]
            reason_counts["nonfinite"] = reason_counts.get("nonfinite", 0) + int(
                (~finite).sum()
            )
            reason_counts["E_le_10"] = reason_counts.get("E_le_10", 0) + int(
                (finite & (physical["E"][:consumed] <= 10.0)).sum()
            )
            reason_counts["pz_le_0"] = reason_counts.get("pz_le_0", 0) + int(
                (finite & (physical["pz"][:consumed] <= 0.0)).sum()
            )
            for name, mask in reasons.items():
                reason_counts[name] = reason_counts.get(name, 0) + int(
                    mask[:consumed].sum()
                )
            positions = np.flatnonzero(keep)
            for name in PHYSICAL_FEATURES:
                accepted_parts[name].append(physical[name][positions])
            attempted += consumed
            accepted += len(positions)
    if accepted != requested:
        raise RuntimeError(
            f"Accepted {accepted}/{requested} rows within {maximum_attempts} proposals"
        )
    rejection_fraction = (attempted - accepted) / attempted
    if rejection_fraction > float(config["guard"]["maximum_rejection_fraction"]):
        raise RuntimeError(
            f"Generation rejection {rejection_fraction:.3%} exceeds configured maximum"
        )
    features = {name: np.concatenate(parts) for name, parts in accepted_parts.items()}
    physical_matrix = np.column_stack([features[name] for name in PHYSICAL_FEATURES])
    if not np.isfinite(physical_matrix).all():
        raise FloatingPointError("Accepted generated output contains non-finite values")
    if np.any(features["E"] <= 10.0) or np.any(features["pz"] <= 0.0):
        raise ValueError("Accepted generated output violates E>10 or pz>0")
    mass = float(config["reconstruction"]["muon_mass_gev"])
    mass_shell = (
        np.square(features["E"])
        - np.square(features["px"])
        - np.square(features["py"])
        - np.square(features["pz"])
        - mass**2
    )
    plane_residual = (
        features["z"]
        - plane.intercept
        - plane.x * features["x"]
        - plane.y * features["y"]
    )
    contract = {
        "finite": True,
        "E_gt_10": True,
        "pz_gt_0": True,
        "maximum_absolute_mass_shell_residual_gev2": float(np.max(np.abs(mass_shell))),
        "maximum_absolute_scoring_plane_residual": float(
            np.max(np.abs(plane_residual))
        ),
    }
    if contract["maximum_absolute_mass_shell_residual_gev2"] > 1.0e-6:
        raise ValueError("Generated output violates the mass-shell contract")
    if contract["maximum_absolute_scoring_plane_residual"] > 1.0e-8:
        raise ValueError("Generated output violates the scoring-plane contract")
    metadata = {
        "run": np.ones(requested, dtype=np.int32),
        "event": np.arange(requested, dtype=np.int64),
        "id": np.full(requested, 13, dtype=np.int32),
        "generation": np.full(requested, seed, dtype=np.int32),
    }
    base_arrays = {**metadata, **features}
    output = Path(config["output"]["directory"])
    if output.exists():
        raise FileExistsError(output)
    year = config["dataset"]["year"]
    prefix = (
        f"FS_generate_{year}_model4_"
        f"preprocess_{config['preprocessing']['id']}_drop_z_E_"
        f"genseed{seed}_guard_reject_v2"
    )
    reference = _weight_summary(
        config["normalization"]["reference_weight_files"],
        config["generation"]["tree_name"],
    )
    constant = float(reference["sum_w"]) / requested
    base_path = output / f"{prefix}_kinematics.root"
    w1_path = output / f"{prefix}_w1.root"
    constant_path = output / f"{prefix}_global_c.root"
    manifest = {
        "status": "complete",
        "dataset": config["dataset"],
        "model": "Model 4 weighted-density NF",
        "preprocessing": config["preprocessing"]["id"],
        "ablation": "drop_z_E",
        "training_run": config["training_run"],
        "checkpoint": config["checkpoint"],
        "generation": config["generation"],
        "guard": config["guard"],
        "reconstruction": config["reconstruction"],
        "physical_contract": contract,
        "rejection": {
            "attempted_rows": attempted,
            "accepted_rows": requested,
            "rejected_rows": attempted - requested,
            "rejection_fraction": rejection_fraction,
            "reason_counts_nonexclusive": reason_counts,
        },
        "normalization": {
            **config["normalization"],
            "reference_weight_summary": reference,
            "global_constant_weight": constant,
        },
        "outputs": {
            "kinematics": str(base_path.resolve()),
            "w1": str(w1_path.resolve()),
            "global_c": str(constant_path.resolve()),
            "same_kinematics_in_weight_exports": True,
        },
    }
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if partial.exists():
        raise FileExistsError(partial)
    partial.mkdir(parents=True, exist_ok=False)
    try:
        write_root_arrays(partial / base_path.name, base_arrays)
        write_root_arrays(
            partial / w1_path.name,
            {**base_arrays, "w": np.ones(requested, dtype=np.float64)},
        )
        write_root_arrays(
            partial / constant_path.name,
            {**base_arrays, "w": np.full(requested, constant, dtype=np.float64)},
        )
        write_json_atomic(partial / "generation_manifest.json", manifest)
        write_json_atomic(
            partial / "_SUCCESS.json",
            {"stage": "generation", "rows": requested},
        )
        partial.replace(output)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return manifest
