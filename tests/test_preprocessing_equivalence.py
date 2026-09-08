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
