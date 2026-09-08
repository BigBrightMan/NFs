from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import uproot

from flashsim_nf.data import Model4RootAdapter, RootSplitSpec

FEATURES = ("x", "y", "z", "pz", "px", "py", "t")


def _write_root(path: Path, *, rows: int, misalign: bool = False) -> None:
    index = np.arange(rows, dtype=np.int64)
    payload = {
        "run": np.ones(rows, dtype=np.int32),
        "event": index + (1 if misalign else 0),
        "id": np.full(rows, 13, dtype=np.int32),
        "generation": np.ones(rows, dtype=np.int32),
        "x": index.astype(np.float64),
        "y": index.astype(np.float64) + 1,
        "z": index.astype(np.float64) + 2,
        "pz": index.astype(np.float64) + 100,
        "px": index.astype(np.float64) - 2,
        "py": index.astype(np.float64) - 1,
        "t": index.astype(np.float64) + 10,
        "w": np.linspace(0.1, 0.2, rows, dtype=np.float64),
    }
    with uproot.recreate(path) as destination:
        destination["nt"] = payload


def _spec(tmp_path: Path, split: str = "train", *, raw_name: str = "raw.root"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = 6
    model = tmp_path / f"{split}_model.root"
    raw = tmp_path / raw_name
    manifest = tmp_path / "split_manifest.json"
    _write_root(model, rows=rows)
    if not raw.exists():
        _write_root(raw, rows=rows)
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "fluka_test",
                "split_id": "frozen_split",
                "dataset_fingerprint": "fingerprint",
                "counts": {"train": rows, "validation": rows, "test": rows},
            }
        )
    )
    return RootSplitSpec(
        dataset_id="fluka_test",
        split=split,
        model_space_path=model,
        raw_weight_path=raw,
        feature_order=FEATURES,
        expected_rows=rows,
        split_manifest_path=manifest,
        expected_split_id="frozen_split",
        expected_dataset_fingerprint="fingerprint",
    )


def test_root_adapter_loads_aligned_features_and_separate_weights(
    tmp_path: Path,
) -> None:
    loaded = Model4RootAdapter().load(_spec(tmp_path))
    assert loaded.features.shape == (6, len(FEATURES))
    assert loaded.weights.shape == (6,)
    assert loaded.features.dtype == np.float32
    assert loaded.weights.dtype == np.float32
    assert loaded.provenance["weight_is_input_feature"] is False


def test_root_adapter_rejects_metadata_misalignment(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write_root(spec.raw_weight_path, rows=6, misalign=True)
    with pytest.raises(ValueError, match="misaligned"):
        Model4RootAdapter().load(spec)


def test_training_pair_rejects_test_split(tmp_path: Path) -> None:
    train = _spec(tmp_path / "train", "train")
    test = _spec(tmp_path / "test", "test")
    with pytest.raises(ValueError, match="only train and validation"):
        Model4RootAdapter().load_training_pair(train, test)


def test_weight_cannot_be_an_input_feature(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    with pytest.raises(ValueError, match="must not be an input"):
        RootSplitSpec(**{**spec.__dict__, "feature_order": (*FEATURES, "w")})


def test_root_adapter_supports_float64_for_reference_statistics(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    loaded = Model4RootAdapter().load(
        RootSplitSpec(
            **{
                **spec.__dict__,
                "feature_dtype": "float64",
                "weight_dtype": "float64",
            }
        )
    )
    assert loaded.features.dtype == np.float64
    assert loaded.weights.dtype == np.float64
