from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from flashsim_nf.experiments import create_model4_baseline_configs

SOURCE_PROJECT = Path(__file__).resolve().parents[1]


def _project_fixture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "NFs"
    for relative in (
        "configs/models/model4/baseline.yaml",
        "configs/models/model4/tuning_space.yaml",
        "configs/preprocessing/A.yaml",
        "configs/preprocessing/B.yaml",
        "configs/preprocessing/C.yaml",
    ):
        source = SOURCE_PROJECT / relative
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    prepared = tmp_path / "prepared"
    for pipeline in ("A", "B", "C"):
        directory = prepared / f"preprocessing_{pipeline}"
        (directory / "8d").mkdir(parents=True)
        (directory / "preprocessor.joblib").write_bytes(pipeline.encode())
        (directory / "preprocessing_parameters.json").write_text(
            json.dumps({"pipeline": pipeline})
        )
        (directory / "8d/feature_order.json").write_text(
            json.dumps(["x", "y", "z", "E", "pz", "px", "py", "t"])
        )
    (prepared / "split").mkdir()
    (prepared / "split/split_manifest.json").write_text("{}")

    dataset = project / "configs/datasets/fluka2025_test.yaml"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "fluka2025_test",
                "year": 2025,
                "selected_rows": 10,
                "dataset_fingerprint": "fingerprint",
                "split": {
                    "id": "split",
                    "counts": {"train": 6, "validation": 2, "test": 2},
                },
                "legacy_prepared_paths": {
                    "local": str(prepared),
                    "cern": str(prepared),
                },
            }
        )
    )
    return project, prepared


def test_create_all_three_baseline_pipeline_configs(tmp_path: Path) -> None:
    project, _ = _project_fixture(tmp_path)
    records = create_model4_baseline_configs(
        project_root=project,
        output_root=tmp_path / "outputs",
        config_output_directory=tmp_path / "resolved",
        environment="cern",
        stage="smoke",
        dataset_ids=["fluka2025_test"],
    )
    assert len(records) == 3
    assert {record.preprocessing for record in records} == {"A", "B", "C"}
    assert {record.status for record in records} == {"created"}
    assert len({record.run_id for record in records}) == 3
    assert all(Path(record.config_path).is_file() for record in records)


def test_baseline_matrix_can_verify_and_skip_existing_configs(
    tmp_path: Path,
) -> None:
    project, _ = _project_fixture(tmp_path)
    arguments = {
        "project_root": project,
        "output_root": tmp_path / "outputs",
        "config_output_directory": tmp_path / "resolved",
        "environment": "cern",
        "stage": "smoke",
        "dataset_ids": ["fluka2025_test"],
    }
    create_model4_baseline_configs(**arguments)
    repeated = create_model4_baseline_configs(**arguments, skip_existing=True)
    assert len(repeated) == 3
    assert {record.status for record in repeated} == {"existing_verified"}
