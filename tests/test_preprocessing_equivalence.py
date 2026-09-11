from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from flashsim_nf.preprocessing import build_preprocessor

FS_SOURCE = Path(__file__).resolve().parents[2] / "FS" / "src"
if str(FS_SOURCE) not in sys.path:
    sys.path.insert(0, str(FS_SOURCE))

FEATURES = ["x", "y", "z", "E", "pz", "px", "py", "t"]


def _sample() -> np.ndarray:
    rng = np.random.default_rng(29)
    rows = 512
    return np.column_stack(
        [
            rng.uniform(-180.0, 20.0, rows),
            rng.uniform(-100.0, 120.0, rows),
            rng.uniform(44_860.0, 44_875.0, rows),
            rng.lognormal(6.1, 0.7, rows) + 20.0,
            rng.lognormal(5.8, 0.8, rows) + 1.0,
            rng.normal(-1.0, 3.0, rows),
            rng.normal(0.0, 2.0, rows),
            rng.normal(1496.7, 0.2, rows),
        ]
    )


@pytest.mark.parametrize("pipeline", ["A", "B", "C"])
def test_composed_pipeline_matches_fs_reference(pipeline: str) -> None:
    from flashsim.preprocessing import build_preprocessor as build_fs

    values = _sample()
    reference = build_fs(pipeline, FEATURES).fit(values)
    composed = build_preprocessor(pipeline, FEATURES).fit(values)

    np.testing.assert_allclose(
        composed.transform(values), reference.transform(values), rtol=2e-10, atol=2e-10
    )
    np.testing.assert_allclose(
        composed.inverse_transform(composed.transform(values)),
        reference.inverse_transform(reference.transform(values), as_frame=False),
        rtol=2e-10,
        atol=2e-10,
    )


def test_weight_is_not_required_as_model4_feature() -> None:
    values = _sample()[:, [0, 1, 2, 4, 5, 6, 7]]
    features = ["x", "y", "z", "pz", "px", "py", "t"]
    transformed = build_preprocessor("B", features).fit(values).transform(values)
    assert transformed.shape == values.shape


def test_preprocessor_must_be_fitted_before_transform() -> None:
    with pytest.raises(RuntimeError, match="not been fitted"):
        build_preprocessor("A", FEATURES).transform(_sample())


def test_composed_preprocessor_serialization_round_trip(tmp_path) -> None:
    values = _sample()
    original = build_preprocessor("C", FEATURES).fit(values)
    path = tmp_path / "preprocessor.json"
    original.save(path)
    restored = type(original).load(path)
    np.testing.assert_allclose(restored.transform(values), original.transform(values))


def test_pipeline_d_changes_only_the_energy_pair_relative_to_pipeline_a() -> None:
    values = _sample()
    pipeline_a = build_preprocessor("A", FEATURES).fit(values)
    pipeline_d = build_preprocessor("D", FEATURES).fit(values)
    transformed_a = pipeline_a.transform(values)
    transformed_d = pipeline_d.transform(values)
    for feature in set(FEATURES) - {"E", "pz"}:
        index = FEATURES.index(feature)
        np.testing.assert_allclose(transformed_d[:, index], transformed_a[:, index])
    for feature in ("E", "pz"):
        index = FEATURES.index(feature)
        assert not np.allclose(transformed_d[:, index], transformed_a[:, index])


@pytest.mark.parametrize(
    ("ablation", "dropped", "trained_energy_feature"),
    [("drop_ze", {"z", "E"}, "pz"), ("drop_z_pz", {"z", "pz"}, "E")],
)
def test_pipeline_d_is_a_single_variable_change_within_each_ablation(
    ablation: str, dropped: set[str], trained_energy_feature: str
) -> None:
    """D must differ from A in exactly one *trained* feature per ablation.

    Model 4 never trains on `E` and `pz` together, so changing both in D still
    yields an isolated comparison against A inside `drop_ze` and `drop_z_pz`.
    """

    trained = [name for name in FEATURES if name not in dropped]
    columns = [FEATURES.index(name) for name in trained]
    values = _sample()
    transformed_a = build_preprocessor("A", FEATURES).fit(values).transform(values)
    transformed_d = build_preprocessor("D", FEATURES).fit(values).transform(values)
    differing = [
        name
        for name, index in zip(trained, columns)
        if not np.allclose(transformed_d[:, index], transformed_a[:, index])
    ]
    assert differing == [trained_energy_feature], (
        f"{ablation}: expected only {trained_energy_feature} to differ, got {differing}"
    )


def test_pipeline_d_matches_pipeline_e_on_the_drop_z_pz_feature_set() -> None:
    """D subsumes E: with `pz` dropped, the two produce identical model space."""

    trained = [name for name in FEATURES if name not in {"z", "pz"}]
    values = _sample()
    transformed_d = build_preprocessor("D", FEATURES).fit(values).transform(values)
    transformed_e = build_preprocessor("E", FEATURES).fit(values).transform(values)
    for name in trained:
        index = FEATURES.index(name)
        np.testing.assert_allclose(transformed_d[:, index], transformed_e[:, index])


def test_pipeline_d_inverse_survives_model_space_far_below_the_energy_edge() -> None:
    """Regression guard for the Box-Cox inverse domain limit that D removes.

    Pipeline A's `inv_boxcox` is undefined below `-1/lambda`, which sits only
    0.075-0.203 sigma beneath the physical 10 GeV edge across the four campaigns,
    so a sampler that strays there yields NaN. D's inverse is `exp`, defined on all
    of R and strictly positive. `inverse_transform` raises on non-finite output, so
    an exception here is itself the regression signal.
    """

    sigmas = np.array([-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 4.0])
    probe = np.repeat(sigmas[:, None], len(FEATURES), axis=1)
    recovered = build_preprocessor("D", FEATURES).fit(_sample()).inverse_transform(probe)
    for feature in ("E", "pz"):
        column = recovered[:, FEATURES.index(feature)]
        assert np.all(column > 0.0), f"{feature} inverse produced non-positive values"


def test_pipeline_e_changes_only_energy_relative_to_pipeline_a() -> None:
    values = _sample()
    pipeline_a = build_preprocessor("A", FEATURES).fit(values)
    pipeline_e = build_preprocessor("E", FEATURES).fit(values)
    transformed_a = pipeline_a.transform(values)
    transformed_e = pipeline_e.transform(values)
    for feature in set(FEATURES) - {"E"}:
        index = FEATURES.index(feature)
        np.testing.assert_allclose(transformed_e[:, index], transformed_a[:, index])
    assert not np.allclose(
        transformed_e[:, FEATURES.index("E")],
        transformed_a[:, FEATURES.index("E")],
    )
