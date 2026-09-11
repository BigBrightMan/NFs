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
from ..preprocessing import load_preprocessor
from ..reconstruction import (
    ScoringPlane,
    fit_scoring_plane,
    reconstruct_drop_z_e,
    reconstruct_drop_z_pz,
    reconstruct_full_8d,
)

FORMAT = "flashsim_nf.model4_generation_config"
VERSION = 1

# `E` is reconstructed as sqrt(px^2 + py^2 + pz^2 + m^2), so the residual
# E^2 - px^2 - py^2 - pz^2 - m^2 is zero in exact arithmetic and is pure float64
# rounding in practice. That rounding scales with E^2 (about 2.5 * eps * E^2), so an
# absolute threshold silently becomes a hard ceiling on generated energy: 1.0e-6 GeV^2
# is exceeded above roughly 42 TeV by rounding alone, with no physics violation.
# Compare the residual against its own magnitude scale instead. The relative bound is
# about four thousand times the float64 noise floor, so it still catches any genuine
# mass-shell violation, which is O(1) relative.
MASS_SHELL_RELATIVE_TOLERANCE = 1.0e-12
SCORING_PLANE_RELATIVE_TOLERANCE = 1.0e-12
SCORING_PLANE_ABSOLUTE_FLOOR = 1.0e-8


def _empirical_envelope_excursion(
    features: dict[str, np.ndarray], guard: dict[str, Any]
) -> dict[str, Any]:
    """Report how far accepted rows travelled beyond the observed FLUKA envelope.

    When the guard is train-fitted the observed min/max is `diagnostic_only`, so
    nothing bounds the generated support from above. That is the intended policy —
    a finite-sample envelope is not a physical limit — but it means an unbounded
    inverse transform can place rows far outside anything FLUKA produced. Record the
    excursion so the manifest carries the evidence instead of leaving it implicit.
    """

    support = guard.get("empirical_support") or {}
    bounds = support.get("feature_bounds") or {}
    report: dict[str, Any] = {
        "action": support.get("action"),
        "contract_id": support.get("contract_id"),
        "features": {},
    }
    worst_name: str | None = None
    worst_ratio = 0.0
    for name, bound in bounds.items():
        values = features.get(name)
        if values is None:
            continue
        lower = float(bound["physical_lower"])
        upper = float(bound["physical_upper"])
        width = upper - lower
        observed_low = float(np.min(values))
        observed_high = float(np.max(values))
        above = max(0.0, observed_high - upper)
        below = max(0.0, lower - observed_low)
        entry = {
            "envelope_lower": lower,
            "envelope_upper": upper,
            "generated_minimum": observed_low,
            "generated_maximum": observed_high,
            "rows_above_envelope": int(np.count_nonzero(values > upper)),
            "rows_below_envelope": int(np.count_nonzero(values < lower)),
        }
        if width > 0.0:
            entry["excursion_above_in_envelope_widths"] = above / width
            entry["excursion_below_in_envelope_widths"] = below / width
            ratio = max(above, below) / width
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_name = name
        if upper > 0.0:
            entry["generated_maximum_over_envelope_upper"] = observed_high / upper
        report["features"][name] = entry
    report["worst_feature"] = worst_name
    report["worst_excursion_in_envelope_widths"] = worst_ratio
    return report


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
    elif purpose in {"muondis", "envelope_diagnostic"}:
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
    physical_contract = guard.get("physical_contract")
    if not isinstance(physical_contract, dict):
        raise ValueError(f"{purpose} guard has no explicit physical contract")
    if physical_contract.get("contract_id") != "model4_drop_ze_physics_v1":
        raise ValueError("Unsupported guard physical contract")
    empirical_support = guard.get("empirical_support")
    if not isinstance(empirical_support, dict):
        raise ValueError("Guard has no empirical observed-envelope contract")
    if empirical_support.get("rule") != "raw_observed_minmax":
        raise ValueError("Guard empirical envelope must use raw observed min/max")
    if empirical_support.get("contract_id") != "per_dataset_observed_envelope_v2":
        raise ValueError("Unsupported empirical observed-envelope contract")
    if empirical_support.get("fit_scope") != guard.get("fit_scope"):
        raise ValueError("Empirical-envelope scope does not match its reference role")
    action = empirical_support.get("action")
    if action not in {"operational_reject", "diagnostic_only"}:
        raise ValueError(
            f"Unsupported empirical-envelope action {action!r}; expected "
            "'operational_reject' or 'diagnostic_only'"
        )
    if purpose in {"muondis", "envelope_diagnostic"} and action != "operational_reject":
        raise ValueError(f"{purpose} requires empirical-envelope action=operational_reject")
    # For validation and test the envelope action is a recorded policy choice, not a
    # leakage control: `fit_scope == "train"` above is what keeps validation and test
    # information out of the guard. Both actions are therefore permitted here, and the
    # generation manifest records which one was in force.
    bounds = empirical_support.get("feature_bounds")
    required_features = {"x", "y", "z", "px", "py", "pz", "E", "t"}
    if not isinstance(bounds, dict) or set(bounds) != required_features:
        raise ValueError("Guard empirical envelope must contain all eight features")


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
    if purpose not in {"validation", "test", "muondis", "envelope_diagnostic"}:
        raise ValueError(
            "purpose must be validation, test, muondis, or envelope_diagnostic"
        )
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
    ablation = str(training["model"].get("ablation"))
    if ablation not in {"drop_ze", "drop_z_pz", "none"}:
        raise ValueError(f"Unsupported Model 4 ablation: {ablation}")
    if training["model"].get("objective") != "weighted_nll":
        raise ValueError("Checkpoint is not a Model 4 weighted-density run")
    if training["model"].get("weight_is_input_feature") is not False:
        raise ValueError("Model 4 generation forbids w as an NF feature")
    prepared = Path(training["resolution"]["prepared_directory"]).resolve()
    split_counts = dataset["split_counts"]
    default_rows = split_counts.get(
        "validation" if purpose == "envelope_diagnostic" else purpose
    )
    rows = int(number_events if number_events is not None else default_rows or 0)
    if rows <= 0:
        raise ValueError("number_events must be supplied and positive")
    # `envelope_diagnostic` deliberately reuses the validation seed so the flow
    # proposal stream is identical to the matching validation run and the only
    # difference between the two samples is which observed envelope rejected.
    seed_defaults = {
        "validation": 1556,
        "test": 2556,
        "muondis": 3556,
        "envelope_diagnostic": 1556,
    }
    seed = int(
        generation_seed if generation_seed is not None else seed_defaults[purpose]
    )
    if purpose in {"validation", "envelope_diagnostic"}:
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
    support_contract = str(guard["empirical_support"]["contract_id"])
    output = (
        run
        / "samples"
        / purpose
        / f"generation_seed_{seed}"
        / "guard_reject_v2"
        / support_contract
    )
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
            "artifact": preprocessing.get("artifact", preprocessing["metadata"]),
            "artifact_sha256": preprocessing.get(
                "artifact_sha256", preprocessing["metadata_sha256"]
            ),
            "feature_order": training["data"]["feature_order"],
        },
        "model": training["model"],
        "reconstruction": {
            "type": ablation,
            "muon_mass_gev": 0.1056583755,
            "scoring_plane": plane.to_dict(),
        },
        "guard": {
            "id": "guard_reject_v2",
            "artifact": str(guard_path),
            "artifact_sha256": _sha256(guard_path),
            "maximum_rejection_fraction": 0.05,
            "physical_contract": guard["physical_contract"],
            "empirical_envelope": guard["empirical_support"]["feature_bounds"],
            "empirical_envelope_action": guard["empirical_support"]["action"],
            "empirical_envelope_scope": guard["empirical_support"]["fit_scope"],
            "empirical_envelope_contract": support_contract,
        },
        "generation": {
            "purpose": purpose,
            # An envelope_diagnostic sample is generated under the all-clean guard,
            # whose bounds contain validation and test information. It exists only to
            # measure how much the envelope choice matters and must never reach model
            # selection or a test claim.
            "selection_allowed": purpose == "validation",
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
            "preprocessing artifact",
            config["preprocessing"]["artifact"],
            config["preprocessing"]["artifact_sha256"],
        ),
        (
            "preprocessing metadata",
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
    preprocessor = load_preprocessor(
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
    diagnostic_counts: dict[str, int] = {}
    with torch.no_grad():
        while accepted < requested and attempted < maximum_attempts:
            proposal_rows = min(batch_size, maximum_attempts - attempted)
            model_space = model.sample(proposal_rows).detach().cpu().numpy()
            physical_selected = preprocessor.inverse_transform(model_space)
            reconstruction = config["reconstruction"]["type"]
            if reconstruction == "drop_ze":
                physical = reconstruct_drop_z_e(
                    physical_selected,
                    feature_order=feature_order,
                    scoring_plane=plane,
                    muon_mass_gev=float(config["reconstruction"]["muon_mass_gev"]),
                )
            elif reconstruction == "drop_z_pz":
                physical = reconstruct_drop_z_pz(
                    physical_selected,
                    feature_order=feature_order,
                    scoring_plane=plane,
                    muon_mass_gev=float(config["reconstruction"]["muon_mass_gev"]),
                )
            elif reconstruction == "none":
                physical = reconstruct_full_8d(
                    physical_selected, feature_order=feature_order
                )
            else:  # pragma: no cover - validated when resolving the config
                raise ValueError(f"Unknown reconstruction: {reconstruction}")
            matrix = np.column_stack([physical[name] for name in PHYSICAL_FEATURES])
            finite = np.isfinite(matrix).all(axis=1)
            domain = finite & (physical["E"] > 10.0) & (physical["pz"] > 0.0)
            reference = ReferenceData(
                features=physical,
                weights=np.ones(proposal_rows, dtype=np.float64),
                label="generated_proposals",
            )
            guard_rejected, reasons = guard_rejection_masks(reference, guard)
            keep = domain & ~guard_rejected
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
                count = int(mask[:consumed].sum())
                destination = (
                    diagnostic_counts
                    if name.startswith("diagnostic_")
                    else reason_counts
                )
                destination[name] = destination.get(name, 0) + count
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
    mass_shell_scale = np.maximum(np.square(features["E"]), 1.0)
    mass_shell_relative = np.abs(mass_shell) / mass_shell_scale
    plane_scale = np.maximum(np.abs(features["z"]), 1.0)
    plane_relative = np.abs(plane_residual) / plane_scale
    worst_mass_shell = int(np.argmax(mass_shell_relative))
    worst_plane = int(np.argmax(plane_relative))
    contract = {
        "finite": True,
        "E_gt_10": True,
        "pz_gt_0": True,
        "maximum_absolute_mass_shell_residual_gev2": float(np.max(np.abs(mass_shell))),
        "maximum_relative_mass_shell_residual": float(
            mass_shell_relative[worst_mass_shell]
        ),
        "mass_shell_relative_tolerance": MASS_SHELL_RELATIVE_TOLERANCE,
        "energy_at_worst_mass_shell_residual_gev": float(
            features["E"][worst_mass_shell]
        ),
        "maximum_absolute_scoring_plane_residual": float(
            np.max(np.abs(plane_residual))
        ),
        "maximum_relative_scoring_plane_residual": float(plane_relative[worst_plane]),
        "scoring_plane_relative_tolerance": SCORING_PLANE_RELATIVE_TOLERANCE,
        "maximum_generated_energy_gev": float(np.max(features["E"])),
        "maximum_generated_pz_gev": float(np.max(features["pz"])),
    }
    if contract["maximum_relative_mass_shell_residual"] > MASS_SHELL_RELATIVE_TOLERANCE:
        raise ValueError(
            "Generated output violates the mass-shell contract: relative residual "
            f"{contract['maximum_relative_mass_shell_residual']:.3e} exceeds "
            f"{MASS_SHELL_RELATIVE_TOLERANCE:.3e} "
            f"(absolute {contract['maximum_absolute_mass_shell_residual_gev2']:.3e} GeV^2 "
            f"at E = {contract['energy_at_worst_mass_shell_residual_gev']:.6g} GeV)"
        )
    plane_absolute = contract["maximum_absolute_scoring_plane_residual"]
    if (
        contract["maximum_relative_scoring_plane_residual"]
        > SCORING_PLANE_RELATIVE_TOLERANCE
        and plane_absolute > SCORING_PLANE_ABSOLUTE_FLOOR
    ):
        raise ValueError(
            "Generated output violates the scoring-plane contract: relative residual "
            f"{contract['maximum_relative_scoring_plane_residual']:.3e} exceeds "
            f"{SCORING_PLANE_RELATIVE_TOLERANCE:.3e} "
            f"(absolute {plane_absolute:.3e})"
        )
    envelope_excursion = _empirical_envelope_excursion(features, guard)
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
    ablation = str(config["model"]["ablation"])
    display_ablation = {
        "drop_ze": "drop_z_E",
        "drop_z_pz": "drop_z_pz",
        "none": "full_8d",
    }[ablation]
    prefix = (
        f"FS_generate_{year}_model4_"
        f"preprocess_{config['preprocessing']['id']}_{display_ablation}_"
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
        "ablation": display_ablation,
        "training_run": config["training_run"],
        "checkpoint": config["checkpoint"],
        "generation": config["generation"],
        "guard": config["guard"],
        "reconstruction": config["reconstruction"],
        "physical_contract": contract,
        "empirical_envelope_excursion": envelope_excursion,
        "selection_allowed": bool(config["generation"].get("selection_allowed", False)),
        "rejection": {
            "attempted_rows": attempted,
            "accepted_rows": requested,
            "rejected_rows": attempted - requested,
            "rejection_fraction": rejection_fraction,
            "reason_counts_nonexclusive": reason_counts,
        },
        "diagnostics": {
            "do_not_affect_acceptance": True,
            "reason_counts_nonexclusive": diagnostic_counts,
            "interpretation": (
                "Robust-tail and validation observed-envelope exceedances are "
                "reported, not treated as proof of physical impossibility."
            ),
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
