from __future__ import annotations

from pathlib import Path

import pytest

from flashsim_nf.artifacts import ArtifactRoots, ArtifactStore, RunKey
from flashsim_nf.naming import build_run_id, validate_slug


def test_short_run_id() -> None:
    assert (
        build_run_id(
            year=2025,
            model="m4",
            preprocessing="A",
            trial_id="lr_hi",
            training_seed=42,
            config_hash="a31f48c2",
        )
        == "fs25-m4-A-lr_hi-s42-a31f48c2"
    )


def test_unsafe_slug_is_rejected() -> None:
    with pytest.raises(ValueError):
        validate_slug("../../bad", field="trial_id")


def test_tuning_path_contains_identity_once() -> None:
    store = ArtifactStore(ArtifactRoots(Path("/data"), Path("/output")))
    key = RunKey(
        dataset_id="fluka2025_muons_horizontal",
        preprocessing="A",
        ablation="drop_ze",
        trial_id="lr_hi",
        training_seed=42,
        config_hash="a31f48c2",
    )

    assert store.tuning_run(key) == Path(
        "/output/campaigns/fluka2025_muons_horizontal/model4/"
        "preprocessing_A/drop_ze/tuning/lr_hi/config_a31f48c2/train_seed_42"
    )
