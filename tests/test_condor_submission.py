from __future__ import annotations

from pathlib import Path

import pytest

from flashsim_nf.execution import CondorResources, build_condor_submission


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "NFs"
    (project / "condor").mkdir(parents=True)
    submit = project / "condor/submit_model4_training.sub"
    launcher = project / "condor/run_model4_training.sh"
    config = tmp_path / "resolved.yaml"
    activate = tmp_path / "activate"
    for path in (submit, launcher, config, activate):
        path.write_text("placeholder")
    launcher.chmod(0o755)
    return project, config, activate


def _config(output_root: Path) -> dict:
    return {
        "dataset": {"dataset_id": "fluka2025_muons_horizontal"},
        "preprocessing": {"id": "B"},
        "experiment": {"stage": "smoke"},
        "output": {"run_directory": str(output_root / "campaigns/run")},
        "resolution": {
            "output_root": str(output_root),
            "run_id": "fs25-m4-B-base-s42-deadbeef",
        },
    }


def test_condor_submission_is_pure_and_points_to_eos_style_outputs(
    tmp_path: Path,
) -> None:
    project, config_path, activate = _files(tmp_path)
    output_root = tmp_path / "outputs"
    submission = build_condor_submission(
        config_path=config_path,
        config=_config(output_root),
        project_root=project,
        activate_path=activate,
        resources=CondorResources(),
    )
    assert submission.command[0] == "condor_submit"
    assert f"output_root={output_root}" in submission.command
    assert "gpus=1" in submission.command
    assert submission.run_name == "fs25-m4-B-base-s42-deadbeef"
    assert not submission.log_directory.exists()
    assert not submission.run_directory.exists()


def test_condor_submission_refuses_existing_run(tmp_path: Path) -> None:
    project, config_path, activate = _files(tmp_path)
    output_root = tmp_path / "outputs"
    config = _config(output_root)
    Path(config["output"]["run_directory"]).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        build_condor_submission(
            config_path=config_path,
            config=config,
            project_root=project,
            activate_path=activate,
            resources=CondorResources(),
        )


def test_model4_condor_resources_require_a_gpu() -> None:
    with pytest.raises(ValueError, match="at least one GPU"):
        CondorResources(gpus=0)
