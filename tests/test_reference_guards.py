from __future__ import annotations

import numpy as np
import pytest

from flashsim_nf.guards import (
    EMPIRICAL_SUPPORT_FEATURES,
    PHYSICAL_FEATURES,
    ReferenceData,
    compare_reference_guards,
    evaluate_guard,
    fit_reference_guard,
    weighted_quantile,
)


def _data(label: str, x: np.ndarray) -> ReferenceData:
    values = np.asarray(x, dtype=np.float64)
    rows = len(values)
    base = np.linspace(-1.0, 1.0, rows)
    return ReferenceData(
        features={
            "x": values,
            "y": base,
            "z": 44870.0 + 0.1 * base,
            "px": 2.0 * base,
            "py": -base,
            "pz": 10.0 + np.arange(rows, dtype=np.float64),
            "E": 11.0 + np.arange(rows, dtype=np.float64),
            "t": 1496.0 + base,
        },
        weights=np.linspace(1.0, 2.0, rows),
        label=label,
    )


def _guards(splits: dict[str, ReferenceData]):
    common = {
        "dataset_id": "fluka_test",
        "dataset_fingerprint": "fingerprint",
        "split_id": "split",
        "iqr_multiplier": 3.0,
        "lower_quantile": 0.01,
        "upper_quantile": 0.99,
    }
    return (
        fit_reference_guard(splits, fit_scope="train", **common),
        fit_reference_guard(splits, fit_scope="all_clean_splits", **common),
    )


def test_weighted_quantile_responds_to_fluka_weights() -> None:
    result = weighted_quantile(
        np.array([0.0, 10.0]),
        [0.5],
        np.array([1.0, 9.0]),
    )
    assert result[0] > 8.0


def test_train_and_all_reference_guards_have_separate_roles() -> None:
    splits = {
        "train": _data("train", np.linspace(0.0, 9.0, 20)),
        "validation": _data("validation", np.linspace(0.0, 30.0, 20)),
        "test": _data("test", np.linspace(0.0, 40.0, 20)),
    }
    train_guard, all_guard = _guards(splits)
    assert train_guard["selection_allowed"] is True
    assert train_guard["usage_role"] == "validation_and_selection"
    assert train_guard["source_splits"] == ["train"]
    assert all_guard["selection_allowed"] is False
    assert all_guard["usage_role"] == "production_only"
    assert all_guard["source_splits"] == ["train", "validation", "test"]
    assert train_guard["settings"]["raw_minmax_used"] is True
    assert (
        train_guard["settings"]["raw_minmax_usage"]
        == "finite_sample_observed_envelope"
    )
    assert train_guard["settings"]["robust_bound_action"] == "diagnostic_only"
    assert set(train_guard["empirical_support"]["feature_bounds"]) == set(
        EMPIRICAL_SUPPORT_FEATURES
    )
    assert train_guard["empirical_support"]["action"] == "operational_reject"
    assert train_guard["empirical_support"]["action_source"] == "caller_policy"
    assert all_guard["empirical_support"]["action"] == "operational_reject"
    assert all_guard["empirical_support"]["action_source"] == "production_only_scope"
    assert (
        train_guard["physical_contract"]["contract_id"]
        == "model4_drop_ze_physics_v1"
    )
    assert (
        all_guard["feature_bounds"]["x"]["physical_upper"]
        > train_guard["feature_bounds"]["x"]["physical_upper"]
    )
    assert train_guard["artifact_sha256"] != all_guard["artifact_sha256"]


def test_same_proposal_comparison_reports_overlap_and_weighted_coverage() -> None:
    splits = {
        "train": _data("train", np.linspace(0.0, 9.0, 20)),
        "validation": _data("validation", np.linspace(0.0, 30.0, 20)),
        "test": _data("test", np.linspace(0.0, 40.0, 20)),
    }
    train_guard, all_guard = _guards(splits)
    generated = _data("same_proposals", np.array([1.0, 5.0, 25.0, 100.0]))
    report = compare_reference_guards(
        splits,
        train_guard=train_guard,
        all_guard=all_guard,
        generated=generated,
    )
    generated_report = report["generated_same_proposals"]
    assert generated_report["train_ref"]["rows"] == 4
    assert generated_report["all_ref"]["rows"] == 4
    assert 0.0 <= generated_report["rejection_jaccard"] <= 1.0
    for split in ("train", "validation", "test"):
        for guard in ("train_ref", "all_ref"):
            result = report["reference"][split][guard]
            assert 0.0 <= result["rejected_row_fraction"] <= 1.0
            assert 0.0 <= result["rejected_weight_fraction"] <= 1.0
            assert set(result["reasons"]) == {
                f"diagnostic_robust_{feature}_outside"
                for feature in PHYSICAL_FEATURES
            } | {
                (
                    f"diagnostic_empirical_{feature}_outside"
                    if guard == "train_ref"
                    else f"production_empirical_{feature}_outside"
                )
                for feature in EMPIRICAL_SUPPORT_FEATURES
            } | {"physical_nonfinite", "physical_E_le_10", "physical_pz_le_0"}


def test_empirical_envelope_uses_raw_minmax_for_each_reference_scope() -> None:
    splits = {
        "train": _data("train", np.linspace(0.0, 9.0, 20)),
        "validation": _data("validation", np.linspace(-3.0, 30.0, 20)),
        "test": _data("test", np.linspace(-4.0, 40.0, 20)),
    }
    train_guard, all_guard = _guards(splits)
    train_x = train_guard["empirical_support"]["feature_bounds"]["x"]
    all_x = all_guard["empirical_support"]["feature_bounds"]["x"]
    assert train_x == {"physical_lower": 0.0, "physical_upper": 9.0}
    assert all_x == {"physical_lower": -4.0, "physical_upper": 40.0}


def test_train_envelope_rejects_outside_observed_fluka_support() -> None:
    """The default train envelope bounds generation to what FLUKA actually produced.

    This is leakage-free: the bounds come from the train split only, so no validation
    or test information reaches the guard. It is what keeps an unbounded inverse
    transform, such as the `exp` used for log-space energies, from placing rows far
    beyond any observed FLUKA event.
    """

    splits = {
        "train": _data("train", np.linspace(0.0, 9.0, 20)),
        "validation": _data("validation", np.linspace(0.0, 30.0, 20)),
        "test": _data("test", np.linspace(0.0, 40.0, 20)),
    }
    train_guard, all_guard = _guards(splits)
    proposal = _data("proposal", np.array([5.0, 25.0, 100.0]))
    train_result = evaluate_guard(proposal, train_guard)
    production_result = evaluate_guard(proposal, all_guard)
    assert train_result["rejected_rows"] == 2
    assert train_result["reasons"]["production_empirical_x_outside"]["rows"] == 2
    assert production_result["rejected_rows"] == 1
    assert production_result["reasons"]["production_empirical_x_outside"]["rows"] == 1


def test_train_envelope_can_be_declared_diagnostic_only() -> None:
    """The monitoring-only policy stays available and is recorded in the artifact."""

    splits = {"train": _data("train", np.linspace(0.0, 9.0, 20))}
    guard = fit_reference_guard(
        splits,
        dataset_id="d",
        dataset_fingerprint="f",
        split_id="s",
        fit_scope="train",
        empirical_support_action="diagnostic_only",
    )
    assert guard["empirical_support"]["action"] == "diagnostic_only"
    proposal = _data("proposal", np.array([5.0, 25.0, 100.0]))
    result = evaluate_guard(proposal, guard)
    assert result["rejected_rows"] == 0
    assert result["reasons"]["diagnostic_empirical_x_outside"]["rows"] == 2


def test_empirical_support_action_is_validated() -> None:
    splits = {"train": _data("train", np.linspace(0.0, 9.0, 20))}
    with pytest.raises(ValueError, match="empirical_support_action"):
        fit_reference_guard(
            splits,
            dataset_id="d",
            dataset_fingerprint="f",
            split_id="s",
            fit_scope="train",
            empirical_support_action="monitor",
        )


def test_all_reference_guard_requires_all_clean_splits() -> None:
    splits = {"train": _data("train", np.linspace(0.0, 9.0, 20))}
    with pytest.raises(ValueError, match="Missing requested guard-fit splits"):
        _guards(splits)
