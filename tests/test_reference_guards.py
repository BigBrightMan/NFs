from __future__ import annotations

import numpy as np
import pytest

from flashsim_nf.guards import (
    PHYSICAL_FEATURES,
    ReferenceData,
    compare_reference_guards,
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
    assert train_guard["settings"]["raw_minmax_used"] is False
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
                f"robust_{feature}_outside" for feature in PHYSICAL_FEATURES
            }


def test_all_reference_guard_requires_all_clean_splits() -> None:
    splits = {"train": _data("train", np.linspace(0.0, 9.0, 20))}
    with pytest.raises(ValueError, match="Missing requested guard-fit splits"):
        _guards(splits)
