from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flashsim_nf.experiments import create_tuning_plan

PROJECT = Path(__file__).resolve().parents[1]


def _plan(tmp_path: Path, **overrides):
    values = {
        "project_root": PROJECT,
        "campaign_config": PROJECT / "configs/campaigns/2025.yaml",
        "tuning_space": PROJECT / "configs/models/model4/tuning_space.yaml",
        "output_root": tmp_path / "output",
        "data_root": tmp_path / "data",
    }
    values.update(overrides)
    return create_tuning_plan(**values)


def test_default_plan_is_three_pipelines_by_nine_controlled_trials(
    tmp_path: Path,
) -> None:
    runs = _plan(tmp_path)
    assert len(runs) == 27
    assert len({run.run_id for run in runs}) == 27
    assert not (tmp_path / "output").exists()
    for run in runs:
        training = run.resolved_config["training"]
        assert training["maximum_epochs"] == 500
        assert training["checkpoint_interval"] == 50
        assert training["early_stopping"]["enabled"] is False
        assert run.resolved_config["experiment"]["data_roles"]["test"] == (
            "reporting_only"
        )


def test_single_trial_requires_explicit_existing_baseline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit baseline"):
        _plan(tmp_path, trial_ids=("lr_hi",), pipelines=("A",))
    runs = _plan(
        tmp_path,
        trial_ids=("lr_hi",),
        pipelines=("A",),
        require_baseline=False,
    )
    assert len(runs) == 1
    assert runs[0].changed_parameters == ("training.learning_rate",)


def test_multiple_overrides_require_combined_tag(tmp_path: Path) -> None:
    source = yaml.safe_load(
        (PROJECT / "configs/models/model4/tuning_space.yaml").read_text()
    )
    source["trials"] = [
        source["trials"][0],
        {
            "trial_id": "invalid",
            "display_name": "Invalid silent combination",
            "enabled": True,
            "tags": ["capacity"],
            "overrides": {
                "model": {"hidden_features": 96},
                "training": {"learning_rate": 0.001},
            },
        },
    ]
    path = tmp_path / "space.yaml"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="changes multiple parameters"):
        _plan(tmp_path, tuning_space=path, pipelines=("A",))
