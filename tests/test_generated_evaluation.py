from __future__ import annotations

import numpy as np
import pytest
import uproot

from flashsim_nf.evaluation.generated import (
    EvaluationSettings,
    evaluate_generated_arrays,
    evaluate_generated_root_files,
    weighted_midrank,
    weighted_spearman_correlation,
)


def _settings() -> EvaluationSettings:
    return EvaluationSettings(
        tail_quantiles=(0.9, 0.99),
        ccdf_quantiles=(0.8, 0.9),
        energy_sample_size=30,
        energy_repeats=2,
        sliced_wasserstein_projections=8,
        minimum_tail_ess=2.0,
        random_seed=7,
    )


def test_identical_generated_arrays_have_zero_deterministic_distances() -> None:
    rng = np.random.default_rng(4)
    train = rng.normal(size=(300, 8))
    reference = rng.normal(size=(120, 8))
    train_weights = rng.uniform(0.1, 1.0, size=len(train))
    reference_weights = rng.uniform(0.1, 1.0, size=len(reference))
    result = evaluate_generated_arrays(
        train=train,
        reference=reference,
        generated=reference.copy(),
        train_weights=train_weights,
        reference_weights=reference_weights,
        generated_weights=reference_weights.copy(),
        settings=_settings(),
    )
    assert result["global_multivariate"]["frechet_gaussian_distance"] < 1e-10
    assert result["global_multivariate"]["sliced_wasserstein"]["mean"] < 1e-10
    assert all(value["weighted_ks"] == 0 for value in result["marginal"].values())
    assert all(
        value["conditional_weighted_ks"] == 0
        for value in result["bulk"].values()
    )
    assert "q0.9900" in result["tails"]["E"]["levels"]


def test_shifted_generated_arrays_increase_fgd() -> None:
    rng = np.random.default_rng(8)
    train = rng.normal(size=(250, 8))
    reference = rng.normal(size=(100, 8))
    ones_train = np.ones(len(train))
    ones_reference = np.ones(len(reference))
    baseline = evaluate_generated_arrays(
        train=train,
        reference=reference,
        generated=reference.copy(),
        train_weights=ones_train,
        reference_weights=ones_reference,
        generated_weights=ones_reference,
        settings=_settings(),
    )
    shifted = evaluate_generated_arrays(
        train=train,
        reference=reference,
        generated=reference + 0.5,
        train_weights=ones_train,
        reference_weights=ones_reference,
        generated_weights=ones_reference,
        settings=_settings(),
    )
    assert shifted["global_multivariate"]["frechet_gaussian_distance"] > baseline[
        "global_multivariate"
    ]["frechet_gaussian_distance"]


def test_weighted_midrank_matches_expanded_integer_weight_population() -> None:
    values = np.array([1.0, 2.0, 2.0, 4.0])
    weights = np.array([1.0, 2.0, 3.0, 2.0])
    expanded = np.repeat(values, weights.astype(int))
    expanded_midrank = weighted_midrank(expanded, np.ones(len(expanded)))
    expected = np.array(
        [expanded_midrank[expanded == value].mean() for value in values]
    )
    assert np.allclose(weighted_midrank(values, weights), expected)


def test_weighted_spearman_matches_expanded_integer_weight_population() -> None:
    matrix = np.array(
        [
            [1.0, 4.0],
            [2.0, 1.0],
            [2.0, 3.0],
            [4.0, 2.0],
        ]
    )
    weights = np.array([1.0, 2.0, 3.0, 2.0])
    expanded = np.repeat(matrix, weights.astype(int), axis=0)
    expanded_ranks = np.column_stack(
        [
            weighted_midrank(expanded[:, index], np.ones(len(expanded)))
            for index in range(expanded.shape[1])
        ]
    )
    expected = np.corrcoef(expanded_ranks, rowvar=False)
    assert np.allclose(weighted_spearman_correlation(matrix, weights), expected)


def _write_root(path, *, rows: int, seed: int, include_weights: bool) -> None:
    rng = np.random.default_rng(seed)
    branches = {
        name: rng.normal(size=rows)
        for name in ("x", "y", "z", "E", "pz", "px", "py", "t")
    }
    branches["E"] = rng.lognormal(mean=4.0, sigma=0.5, size=rows)
    branches["pz"] = rng.lognormal(mean=3.8, sigma=0.5, size=rows)
    if include_weights:
        branches["w"] = rng.uniform(0.1, 1.0, size=rows)
    with uproot.recreate(path) as destination:
        destination["nt"] = branches


def test_root_evaluation_requires_fluka_weights_and_writes_immutable_report(
    tmp_path,
) -> None:
    train = tmp_path / "train.root"
    reference = tmp_path / "validation.root"
    generated = tmp_path / "generated.root"
    _write_root(train, rows=80, seed=1, include_weights=True)
    _write_root(reference, rows=50, seed=2, include_weights=True)
    _write_root(generated, rows=50, seed=3, include_weights=False)
    output = tmp_path / "evaluation"
    report = evaluate_generated_root_files(
        dataset_id="fluka_test",
        train_reference_root=train,
        reference_root=reference,
        generated_root=generated,
        output_directory=output,
        generated_weight_mode="uniform",
        settings=_settings(),
    )
    assert report["format"] == "flashsim_nf.generated_evaluation"
    assert report["format_version"] == 3
    assert report["inputs"]["weight_sources"]["reference"] == "ROOT branch w"
    assert report["inputs"]["weight_sources"]["generated"] == "uniform"
    assert (output / "generated_evaluation.json").is_file()
    assert (output / "bulk_tail_metrics.csv").is_file()
    assert {
        "all_features_bulk_q001_q999.png",
        "all_features_tail_full_range_logy.png",
        "energy_log10_bulk_tail.png",
        "pearson_correlation.png",
        "spearman_correlation.png",
        "tail_ccdf.png",
    } == {path.name for path in (output / "plots").glob("*.png")}
    assert (output / "_SUCCESS.json").is_file()
    with pytest.raises(FileExistsError):
        evaluate_generated_root_files(
            dataset_id="fluka_test",
            train_reference_root=train,
            reference_root=reference,
            generated_root=generated,
            output_directory=output,
            generated_weight_mode="uniform",
            settings=_settings(),
        )

    missing_weight = tmp_path / "reference_without_w.root"
    _write_root(missing_weight, rows=50, seed=4, include_weights=False)
    with pytest.raises(ValueError, match="FLUKA w branch"):
        evaluate_generated_root_files(
            dataset_id="fluka_test",
            train_reference_root=train,
            reference_root=missing_weight,
            generated_root=generated,
            output_directory=tmp_path / "bad_evaluation",
            generated_weight_mode="uniform",
            settings=_settings(),
        )
