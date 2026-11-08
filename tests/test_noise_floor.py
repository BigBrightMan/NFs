from __future__ import annotations

from pathlib import Path

import numpy as np
import uproot

from flashsim_nf.evaluation import noise_floor


def _write_reference(path: Path, rows: int) -> None:
    index = np.arange(rows)
    arrays = {
        "run": np.ones(rows, dtype=np.int32),
        "event": index.astype(np.int64),
        "id": np.full(rows, 13, dtype=np.int32),
        "generation": np.ones(rows, dtype=np.int32),
        "x": index - 10.0,
        "y": index + 1.0,
        "z": 44000.0 + index,
        "E": 30.0 + index,
        "pz": 20.0 + index,
        "px": 0.1 * index,
        "py": -0.1 * index,
        "t": 1000.0 + index,
        "w": 0.01 + 0.001 * index,
    }
    with uproot.recreate(path) as output:
        output["nt"] = arrays


def test_noise_floor_is_validation_only_and_row_matched(
    tmp_path: Path, monkeypatch
) -> None:
    train = tmp_path / "train.root"
    validation = tmp_path / "validation.root"
    output = tmp_path / "metrics.json"
    _write_reference(train, 30)
    _write_reference(validation, 20)

    c2st_calls = []

    def fake_evaluation(**values):
        c2st_calls.append(values.get("include_c2st", True))
        return {
            "rows": len(values["reference"]),
            "global_multivariate": {
                "c2st": {
                    "roc_auc": 0.51,
                    "roc_auc_sd": 0.02,
                    "repeats": 5,
                    "rows_per_class": 9,
                }
            },
        }

    monkeypatch.setattr(
        noise_floor, "evaluate_generated_arrays", fake_evaluation
    )
    monkeypatch.setattr(
        noise_floor,
        "_noise_metrics",
        lambda evaluation, **_: {
            "synthetic_distance": float(evaluation["rows"])
        },
    )
    report = noise_floor.compute_reference_noise_floor(
        dataset_id="fluka2025_test",
        train_root=train,
        validation_root=validation,
        output=output,
        repeats=3,
    )
    assert report["test_data_used"] is False
    assert report["format_version"] == 2
    assert report["method"]["rows_per_sample"] == 20
    assert len(report["bootstrap"]["raw_metrics"]) == 3
    assert all(
        "c2st_distance_from_half" not in record
        for record in report["bootstrap"]["raw_metrics"]
    )
    assert report["c2st_floor"]["method"] == "disjoint_stable_halves_by_source_point"
    assert report["c2st_floor"]["roc_auc"] == 0.51
    assert c2st_calls == [True, False, False, False]
    assert output.is_file()
