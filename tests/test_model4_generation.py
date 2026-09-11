import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from flashsim_nf.generation import (
    generate_model4_from_config,
    resolve_generation_config,
)
from flashsim_nf.guards import PHYSICAL_FEATURES, ReferenceData, fit_reference_guard
from flashsim_nf.models.flows import FlowConfig, build_flow

torch = pytest.importorskip("torch")
uproot = pytest.importorskip("uproot")
pytest.importorskip("nflows")


def _write_raw(path: Path, rows: int) -> None:
    x = np.linspace(-10.0, 10.0, rows)
    y = np.linspace(10.0, -10.0, rows)
    pz = np.linspace(5.0, 100.0, rows)
    px = np.linspace(-2.0, 2.0, rows)
    py = np.linspace(2.0, -2.0, rows)
    mass = 0.1056583755
    values = {
        "run": np.ones(rows, dtype=np.int32),
        "event": np.arange(rows, dtype=np.int64),
        "id": np.full(rows, 13, dtype=np.int32),
        "generation": np.ones(rows, dtype=np.int32),
        "x": x,
        "y": y,
        "z": 100.0 + 2.0 * x - 3.0 * y,
        "E": np.sqrt(pz**2 + px**2 + py**2 + mass**2),
        "pz": pz,
        "px": px,
        "py": py,
        "t": np.linspace(4.0, 5.0, rows),
        "w": np.full(rows, 0.25),
    }
    with uproot.recreate(path) as sink:
        sink["nt"] = values


def test_resolve_and_generate_model4_pair(tmp_path):
    dataset_id = "fluka2025_muons_horizontal"
    prepared = tmp_path / "prepared"
    split = prepared / "split"
    split.mkdir(parents=True)
    for name, rows in (("train", 40), ("validation", 12), ("test", 10)):
        _write_raw(split / f"{name}_rawfeature.root", rows)
    metadata = prepared / "preprocessing_B" / "preprocessing_parameters.json"
    metadata.parent.mkdir()
    parameters = {
        "version": 3,
        "pipeline": "B",
        "feature_order": ["x", "y", "z", "E", "pz", "px", "py", "t", "w"],
        "fitted": True,
        "parameters": {
            "x": {"kind": "identity", "final_mean": 0.0, "final_std": 1.0},
            "y": {"kind": "identity", "final_mean": 0.0, "final_std": 1.0},
            "pz": {"kind": "log", "final_mean": 3.0, "final_std": 0.05},
            "px": {
                "kind": "scaled_arcsinh",
                "scale": 1.0,
                "final_mean": 0.0,
                "final_std": 0.1,
            },
            "py": {
                "kind": "scaled_arcsinh",
                "scale": 1.0,
                "final_mean": 0.0,
                "final_std": 0.1,
            },
            "t": {"kind": "identity", "final_mean": 4.5, "final_std": 0.1},
        },
    }
    metadata.write_text(json.dumps(parameters))
    feature_order = ["x", "y", "pz", "px", "py", "t"]
    model_config = FlowConfig(
        input_dim=6, num_transforms=1, hidden_features=8, num_blocks=1
    )
    model = build_flow(model_config)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    import hashlib

    metadata_hash = hashlib.sha256(metadata.read_bytes()).hexdigest()
    resolved = {
        "dataset": {
            "dataset_id": dataset_id,
            "year": 2025,
            "dataset_fingerprint": "fingerprint",
            "split_id": "split-id",
            "split_counts": {"train": 40, "validation": 12, "test": 10},
        },
        "preprocessing": {
            "id": "B",
            "metadata": str(metadata),
            "metadata_sha256": metadata_hash,
        },
        "data": {
            "feature_order": feature_order,
            "train": {"raw_weight_path": str(split / "train_rawfeature.root")},
        },
        "model": {
            "family": "model4",
            "ablation": "drop_ze",
            "objective": "weighted_nll",
            "weight_is_input_feature": False,
            **model_config.to_dict(),
        },
        "resolution": {"prepared_directory": str(prepared)},
    }
    (run / "config_resolved.json").write_text(json.dumps(resolved))
    (run / "_SUCCESS.json").write_text("{}")
    torch.save(
        {
            "format": "flashsim_nf.full_training_state",
            "model_state_dict": model.state_dict(),
            "config": resolved,
        },
        run / "checkpoints" / "best_model.pt",
    )
    arrays = {}
    with uproot.open(split / "train_rawfeature.root") as source:
        values = source["nt"].arrays(list(PHYSICAL_FEATURES), library="np")
        arrays = {name: values[name] for name in PHYSICAL_FEATURES}
    guard = fit_reference_guard(
        {"train": ReferenceData(arrays, np.ones(40), "train")},
        dataset_id=dataset_id,
        dataset_fingerprint="fingerprint",
        split_id="split-id",
        fit_scope="train",
        iqr_multiplier=100.0,
        lower_quantile=0.001,
        upper_quantile=0.999,
    )
    guard_path = tmp_path / "guard.json"
    guard_path.write_text(json.dumps(guard))
    generation = resolve_generation_config(
        run_directory=run,
        purpose="validation",
        guard_artifact=guard_path,
        number_events=8,
        generation_seed=1556,
    )
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(yaml.safe_dump(generation))
    report = generate_model4_from_config(config_path)
    with (
        uproot.open(report["outputs"]["w1"]) as unit,
        uproot.open(report["outputs"]["global_c"]) as constant,
    ):
        first = unit["nt"].arrays(list(PHYSICAL_FEATURES), library="np")
        second = constant["nt"].arrays(list(PHYSICAL_FEATURES), library="np")
        assert all(
            np.array_equal(first[name], second[name]) for name in PHYSICAL_FEATURES
        )
        assert np.allclose(constant["nt"]["w"].array(library="np"), 12 * 0.25 / 8)
    assert report["reconstruction"]["scoring_plane"]["fit_rows"] == 40
    assert report["guard"]["empirical_envelope_action"] == "diagnostic_only"
    assert report["diagnostics"]["do_not_affect_acceptance"] is True

    split_references = {}
    for split_name, rows in (("train", 40), ("validation", 12), ("test", 10)):
        with uproot.open(split / f"{split_name}_rawfeature.root") as source:
            split_values = source["nt"].arrays(list(PHYSICAL_FEATURES), library="np")
            split_references[split_name] = ReferenceData(
                {name: split_values[name] for name in PHYSICAL_FEATURES},
                np.ones(rows),
                split_name,
            )
    all_guard = fit_reference_guard(
        split_references,
        dataset_id=dataset_id,
        dataset_fingerprint="fingerprint",
        split_id="split-id",
        fit_scope="all_clean_splits",
        iqr_multiplier=100.0,
        lower_quantile=0.001,
        upper_quantile=0.999,
    )
    all_guard_path = tmp_path / "guard_all.json"
    all_guard_path.write_text(json.dumps(all_guard))
    selection_path = tmp_path / "selected_model.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "frozen",
                "test_data_used": False,
                "winner": {"training_run": str(run.resolve())},
            }
        )
    )

    with pytest.raises(ValueError, match="test requires the train-reference guard"):
        resolve_generation_config(
            run_directory=run,
            purpose="test",
            guard_artifact=all_guard_path,
            frozen_selection=selection_path,
        )
    test_config = resolve_generation_config(
        run_directory=run,
        purpose="test",
        guard_artifact=guard_path,
        frozen_selection=selection_path,
    )
    assert test_config["normalization"]["scope"] == "test"

    with pytest.raises(
        ValueError, match="muondis requires the all-clean-FLUKA production guard"
    ):
        resolve_generation_config(
            run_directory=run,
            purpose="muondis",
            guard_artifact=guard_path,
            number_events=100,
            frozen_selection=selection_path,
        )
    production_config = resolve_generation_config(
        run_directory=run,
        purpose="muondis",
        guard_artifact=all_guard_path,
        number_events=100,
        frozen_selection=selection_path,
    )
    assert production_config["normalization"]["scope"] == "train+validation+test"
    assert production_config["guard"]["empirical_envelope_action"] == (
        "operational_reject"
    )


def test_mass_shell_tolerance_is_scale_free_across_the_energy_range():
    """The reconstructed mass-shell residual is float64 noise that grows as E^2.

    `E = sqrt(px^2 + py^2 + pz^2 + m^2)` makes the residual exactly zero in real
    arithmetic, so whatever survives is rounding. Its absolute size scales with E^2,
    which turned the old fixed 1.0e-6 GeV^2 threshold into an implicit ceiling near
    42 TeV. The relative residual must stay flat instead.
    """

    from flashsim_nf.generation.model4 import MASS_SHELL_RELATIVE_TOLERANCE

    mass = 0.1056583755
    px, py = 2.5, -1.75
    for pz in (1.0e2, 1.0e3, 5.126e3, 4.2e4, 1.5e5, 5.0e5):
        energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
        residual = energy**2 - px**2 - py**2 - pz**2 - mass**2
        relative = abs(residual) / max(energy**2, 1.0)
        assert relative <= MASS_SHELL_RELATIVE_TOLERANCE, (
            f"pz={pz:g} GeV gives relative residual {relative:.3e}"
        )


def test_empirical_envelope_excursion_reports_unbounded_generation():
    """A diagnostic-only envelope must still leave excursion evidence behind.

    With a train-fitted guard the observed min/max is `diagnostic_only`, so nothing
    rejects rows above it. The manifest has to record how far past FLUKA's observed
    support the accepted sample went, or the extrapolation is invisible.
    """

    from flashsim_nf.generation.model4 import _empirical_envelope_excursion

    guard = {
        "empirical_support": {
            "action": "diagnostic_only",
            "contract_id": "per_dataset_observed_envelope_v2",
            "feature_bounds": {
                "pz": {"physical_lower": 10.0, "physical_upper": 5_126.0},
                "t": {"physical_lower": 1_395.0, "physical_upper": 1_398.0},
            },
        }
    }
    features = {
        "pz": np.array([50.0, 4_000.0, 500_000.0]),
        "t": np.array([1_396.0, 1_397.0, 1_397.5]),
    }
    report = _empirical_envelope_excursion(features, guard)

    assert report["action"] == "diagnostic_only"
    assert report["worst_feature"] == "pz"
    assert report["features"]["pz"]["rows_above_envelope"] == 1
    assert report["features"]["pz"]["generated_maximum_over_envelope_upper"] > 97.0
    assert report["features"]["t"]["rows_above_envelope"] == 0
    assert report["features"]["t"]["excursion_above_in_envelope_widths"] == 0.0


def test_empirical_envelope_excursion_is_zero_inside_support():
    from flashsim_nf.generation.model4 import _empirical_envelope_excursion

    guard = {
        "empirical_support": {
            "action": "diagnostic_only",
            "contract_id": "per_dataset_observed_envelope_v2",
            "feature_bounds": {
                "pz": {"physical_lower": 10.0, "physical_upper": 5_126.0}
            },
        }
    }
    features = {"pz": np.array([11.0, 2_000.0, 5_125.0])}
    report = _empirical_envelope_excursion(features, guard)
    assert report["worst_excursion_in_envelope_widths"] == 0.0
    assert report["features"]["pz"]["rows_above_envelope"] == 0
    assert report["features"]["pz"]["rows_below_envelope"] == 0


def test_envelope_diagnostic_requires_the_all_clean_guard():
    """The diagnostic purpose is the only validation-reference path allowed the
    all-clean guard, and it must refuse the train guard."""

    from flashsim_nf.generation.model4 import _validate_guard_role

    base = {
        "physical_contract": {"contract_id": "model4_drop_ze_physics_v1"},
        "empirical_support": {
            "rule": "raw_observed_minmax",
            "contract_id": "per_dataset_observed_envelope_v2",
            "action": "operational_reject",
            "feature_bounds": {
                name: {"physical_lower": 0.0, "physical_upper": 1.0}
                for name in ("x", "y", "z", "px", "py", "pz", "E", "t")
            },
        },
    }
    train_guard = {
        **base,
        "fit_scope": "train",
        "source_splits": ["train"],
        "usage_role": "validation_and_selection",
        "selection_allowed": True,
    }
    train_guard["empirical_support"] = {
        **base["empirical_support"],
        "fit_scope": "train",
    }
    all_guard = {
        **base,
        "fit_scope": "all_clean_splits",
        "source_splits": ["train", "validation", "test"],
        "usage_role": "production_only",
        "selection_allowed": False,
    }
    all_guard["empirical_support"] = {
        **base["empirical_support"],
        "fit_scope": "all_clean_splits",
    }

    _validate_guard_role(all_guard, purpose="envelope_diagnostic")
    _validate_guard_role(train_guard, purpose="validation")
    with pytest.raises(ValueError, match="role mismatch"):
        _validate_guard_role(train_guard, purpose="envelope_diagnostic")
    with pytest.raises(ValueError, match="role mismatch"):
        _validate_guard_role(all_guard, purpose="validation")


def test_envelope_diagnostic_is_never_selection_allowed(tmp_path):
    """A diagnostic sample must be stamped so winner selection can refuse it."""

    import subprocess
    import sys

    payload = {
        "format": "flashsim_nf.generated_evaluation",
        "format_version": 3,
        "status": "complete",
        "dataset_id": "fluka2022_muons_down_tclean_v1",
        "reference_split": "validation",
        "selection_allowed": False,
        "generation_purpose": "envelope_diagnostic",
        "inputs": {"generated_root": str(tmp_path / "g.root")},
        "evaluation": {},
    }
    evaluation_path = tmp_path / "generated_evaluation.json"
    evaluation_path.write_text(json.dumps(payload))
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts"
                / "select_model4_validation_winner.py"
            ),
            "--evaluations",
            str(evaluation_path),
            "--output",
            str(tmp_path / "selected_model.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "diagnostic evaluation" in (result.stderr + result.stdout)
