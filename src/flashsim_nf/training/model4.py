"""End-to-end Model 4 training from validated ROOT inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_yaml
from ..data import Model4RootAdapter, RootSplitSpec
from ..models.flows import FlowConfig, build_flow
from ..models.model4.objective import sample_weight_summary, weighted_nll
from .session import TrainingSession
from .state import TrainingPolicy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split_spec(config: dict[str, Any], split: str) -> RootSplitSpec:
    dataset = config["dataset"]
    data = config["data"]
    values = data[split]
    return RootSplitSpec(
        dataset_id=str(dataset["dataset_id"]),
        split=split,
        model_space_path=Path(values["model_space_path"]),
        raw_weight_path=Path(values["raw_weight_path"]),
        feature_order=tuple(data["feature_order"]),
        expected_rows=int(values["expected_rows"]),
        split_manifest_path=Path(data["split_manifest_path"]),
        expected_split_id=str(dataset["split_id"]),
        expected_dataset_fingerprint=str(dataset["dataset_fingerprint"]),
        tree_name=str(data.get("tree_name", "nt")),
    )


def _validate_preprocessor(config: dict[str, Any]) -> None:
    section = config["preprocessing"]
    if section.get("fit_split") != "train":
        raise ValueError("Preprocessor must be fitted on train only")
    artifact = Path(section["artifact"])
    if not artifact.is_file():
        raise FileNotFoundError(f"Preprocessing artifact is missing: {artifact}")
    expected = str(section["artifact_sha256"])
    if _sha256(artifact) != expected:
        raise ValueError("Preprocessing artifact SHA-256 mismatch")


def train_model4_from_config(config_path: str | Path) -> dict[str, Any]:
    """Train one fresh or branch-resumed weighted-density RQ-spline NF."""

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Model 4 training requires the train dependencies"
        ) from error

    config = load_yaml(config_path)
    required = {"dataset", "preprocessing", "data", "model", "training", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Training config is missing sections: {missing}")
    if config["model"].get("objective") != "weighted_nll":
        raise ValueError("Model 4 training objective must be weighted_nll")
    if config["model"].get("weight_is_input_feature") is not False:
        raise ValueError("Model 4 weight must not be an input feature")
    _validate_preprocessor(config)

    train, validation = Model4RootAdapter().load_training_pair(
        _split_spec(config, "train"), _split_spec(config, "validation")
    )
    feature_order = tuple(config["data"]["feature_order"])
    flow_config = FlowConfig.from_mapping(config["model"], input_dim=len(feature_order))
    policy = TrainingPolicy.from_config(config["training"])
    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(torch, str(config["training"].get("device", "auto")))
    model = build_flow(flow_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 5.0e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1.0e-6)),
    )
    scheduler_values = config["training"].get("scheduler", {})
    if scheduler_values.get("type", "ReduceLROnPlateau") != "ReduceLROnPlateau":
        raise ValueError("Only ReduceLROnPlateau is supported")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_values.get("factor", 0.5)),
        patience=int(scheduler_values.get("patience", 5)),
    )
    use_amp = bool(config["training"].get("mixed_precision", False))
    if use_amp and device.type != "cuda":
        raise ValueError("mixed_precision requires a CUDA device")
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

    loader_generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["training"].get("batch_size", 2048))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train.features), torch.from_numpy(train.weights)
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation.features),
            torch.from_numpy(validation.weights),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
        pin_memory=pin_memory,
    )
    gradient_clip = float(config["training"].get("gradient_clip_norm", 5.0))

    def train_epoch(model, optimizer, scaler, epoch):
        model.train()
        numerator = 0.0
        denominator = 0.0
        gradient_norms: list[float] = []
        batch_weight_means: list[float] = []
        for values, weights in train_loader:
            values = values.to(device, non_blocking=pin_memory)
            weights = weights.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            batch_weight_means.append(float(weights.mean().item()))
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                loss = weighted_nll(model.log_prob(values), weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            if scaler is None:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )
                scaler.step(optimizer)
                scaler.update()
            gradient_norms.append(float(gradient_norm))
            contribution = float(weights.sum().item())
            numerator += float(loss.detach().item()) * contribution
            denominator += contribution
        mean_weight = float(np.mean(batch_weight_means))
        return {
            "loss": numerator / denominator,
            "gradient_norm": float(np.mean(gradient_norms)),
            "batch_weight_mean": mean_weight,
            "batch_weight_mean_cv": float(
                np.std(batch_weight_means, ddof=0) / mean_weight
            ),
        }

    def validate_epoch(model, epoch):
        model.eval()
        numerator = 0.0
        denominator = 0.0
        with torch.no_grad():
            for values, weights in validation_loader:
                values = values.to(device, non_blocking=pin_memory)
                weights = weights.to(device, non_blocking=pin_memory)
                loss = weighted_nll(model.log_prob(values), weights)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss at epoch {epoch}"
                    )
                contribution = float(weights.sum().item())
                numerator += float(loss.item()) * contribution
                denominator += contribution
        return numerator / denominator

    resolved = json.loads(json.dumps(config))
    resolved["model"] = {
        "family": "model4",
        "ablation": config["model"]["ablation"],
        "objective": "weighted_nll",
        "weight_is_input_feature": False,
        **flow_config.to_dict(),
    }
    resolved["training_provenance"] = {
        "train": {
            **train.provenance,
            "weight_summary": sample_weight_summary(train.weights),
        },
        "validation": {
            **validation.provenance,
            "weight_summary": sample_weight_summary(validation.weights),
        },
    }
    destination = Path(config["output"]["run_directory"])
    resume = config.get("resume")
    if resume and resume.get("mode") == "branch":
        session = TrainingSession.branch(
            source_checkpoint=resume["source_checkpoint"],
            destination_run=destination,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_loader_generator=loader_generator,
            policy=policy,
            config=resolved,
            expected_epoch=int(resume.get("expected_epoch", 450)),
        )
    elif resume:
        raise ValueError("resume.mode must be 'branch'")
    else:
        session = TrainingSession.fresh(
            destination_run=destination,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_loader_generator=loader_generator,
            policy=policy,
            config=resolved,
        )
    history = session.run(train_epoch=train_epoch, validate_epoch=validate_epoch)
    return {
        "run_directory": str(destination.resolve()),
        "first_epoch": history[0]["epoch"],
        "last_epoch": history[-1]["epoch"],
        "best_epoch": session.early_stopping.best_epoch,
        "best_validation_loss": session.early_stopping.best_validation_loss,
        "mode": session.mode,
    }
