from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from flashsim_nf.training import (
    BranchedTrainingSession,
    EarlyStoppingState,
    TrainingPolicy,
    TrainingSession,
    full_checkpoint_payload,
    inspect_checkpoint,
    prepare_resume_directory,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _objects():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return model, optimizer, scheduler


def _legacy_checkpoint(path: Path) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model, optimizer, scheduler = _objects()
    loss = model(torch.ones(4, 2)).square().mean()
    loss.backward()
    optimizer.step()
    torch.save(
        {
            "format_version": 4,
            "epoch": 450,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_validation_loss": 1.25,
            "epochs_without_improvement": 7,
        },
        path,
    )
    return model, optimizer


def test_inspection_reports_missing_legacy_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch_0450_model.pt"
    _legacy_checkpoint(checkpoint)
    report = inspect_checkpoint(checkpoint)
    assert report["epoch"] == 450
    assert report["next_epoch"] == 451
    assert report["available_state"]["optimizer"] is True
    assert report["available_state"]["scheduler"] is True
    assert report["available_state"]["scaler"] is False
    assert report["available_state"]["rng"] is False
    assert report["available_state"]["data_loader_rng"] is False


def test_branch_resumes_at_451_and_never_mutates_source(tmp_path: Path) -> None:
    source_run = tmp_path / "original"
    source_run.mkdir()
    checkpoint = source_run / "epoch_0450_model.pt"
    _legacy_checkpoint(checkpoint)
    before = _digest(checkpoint)

    model, optimizer, scheduler = _objects()
    destination = tmp_path / "resume_to_500"
    policy = TrainingPolicy(maximum_epochs=452, checkpoint_interval=50)
    session = BranchedTrainingSession.branch(
        source_checkpoint=checkpoint,
        destination_run=destination,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        data_loader_generator=torch.Generator().manual_seed(42),
        policy=policy,
        config={"training": {"maximum_epochs": 452}},
    )

    def train_epoch(model, optimizer, scaler, epoch):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(4, 2)).square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def validate_epoch(model, epoch):
        return 1.0 / epoch

    history = session.run(train_epoch=train_epoch, validate_epoch=validate_epoch)
    assert [row["epoch"] for row in history] == [451, 452]
    assert _digest(checkpoint) == before
    assert (destination / "checkpoints/last_model.pt").is_file()
    assert (destination / "config_resolved.json").is_file()
    assert (destination / "training_summary.json").is_file()
    assert (destination / "_SUCCESS.json").is_file()
    assert not (source_run / "training").exists()
    restore = json.loads((destination / "lineage/restore_report.json").read_text())
    missing = {item["name"] for item in restore["components"] if not item["restored"]}
    assert {"rng", "data_loader_rng", "best_model_weights"} <= missing


def test_resume_destination_must_not_exist(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch_0450_model.pt"
    _legacy_checkpoint(checkpoint)
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_resume_directory(checkpoint, destination)


def test_new_checkpoint_contains_all_restart_state(tmp_path: Path) -> None:
    model, optimizer, scheduler = _objects()
    early = EarlyStoppingState(best_validation_loss=0.75)
    payload = full_checkpoint_payload(
        epoch=450,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        early_stopping=early,
        config={},
        best_model_state_dict=model.state_dict(),
        data_loader_generator=torch.Generator().manual_seed(42),
    )
    assert payload["rng_state"]
    assert payload["optimizer_state_dict"]
    assert payload["scheduler_state_dict"]
    assert payload["early_stopping_state"]["best_validation_loss"] == 0.75
    assert "best_epoch" in payload
    assert payload["data_loader_generator_state"] is not None


def test_fresh_run_saves_best_last_and_full_periodic_checkpoints(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _objects()
    destination = tmp_path / "fresh"
    session = TrainingSession.fresh(
        destination_run=destination,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        policy=TrainingPolicy(maximum_epochs=51, checkpoint_interval=50),
        config={"training": {"maximum_epochs": 51}},
    )

    def train_epoch(model, optimizer, scaler, epoch):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 2)).square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    history = session.run(
        train_epoch=train_epoch,
        validate_epoch=lambda model, epoch: float(epoch),
    )
    assert len(history) == 51
    best = torch.load(
        destination / "checkpoints/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    last = torch.load(
        destination / "checkpoints/last_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    periodic = torch.load(
        destination / "checkpoints/epoch_0050_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert best["epoch"] == 1
    assert best["best_epoch"] == 1
    assert last["epoch"] == 51
    assert last["best_epoch"] == 1
    assert periodic["epoch"] == 50
    assert "optimizer_state_dict" in periodic
    assert "scheduler_state_dict" in periodic
    assert not (destination / "checkpoints/epoch_0051_model.pt").exists()


def test_training_policy_defaults_and_optional_early_stopping() -> None:
    default = TrainingPolicy.from_config({})
    assert default.maximum_epochs == 500
    assert default.checkpoint_interval == 50
    assert default.early_stopping_enabled is False

    enabled = TrainingPolicy.from_config(
        {"early_stopping": {"enabled": True, "patience": 12}}
    )
    assert enabled.early_stopping_enabled is True
    assert enabled.early_stopping_patience == 12


def test_training_prints_flushed_epoch_progress(tmp_path: Path, monkeypatch) -> None:
    model, optimizer, scheduler = _objects()
    destination = tmp_path / "live_progress"
    session = TrainingSession.fresh(
        destination_run=destination,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        policy=TrainingPolicy(maximum_epochs=1, checkpoint_interval=50),
        config={"training": {"maximum_epochs": 1}},
    )
    calls = []

    def record_print(*values, **options):
        calls.append((" ".join(str(value) for value in values), options))

    monkeypatch.setattr("builtins.print", record_print)

    def train_epoch(model, optimizer, scaler, epoch):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 2)).square().mean()
        loss.backward()
        optimizer.step()
        return {
            "loss": float(loss.detach()),
            "gradient_norm": 1.25,
            "batch_weight_mean_cv": 0.04,
        }

    session.run(
        train_epoch=train_epoch,
        validate_epoch=lambda model, epoch: 0.75,
    )

    assert len(calls) == 1
    message, options = calls[0]
    assert "Epoch 1/1" in message
    assert "train NLL=" in message
    assert "validation NLL=0.750000" in message
    assert "best=0.750000@1" in message
    assert "gradient norm=1.2500" in message
    assert "weight CV=0.0400" in message
    assert "new best" in message
    assert options["flush"] is True
