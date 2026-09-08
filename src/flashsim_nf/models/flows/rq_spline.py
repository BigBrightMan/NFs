"""Validated RQ-spline normalizing-flow backend used by Model 4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FlowConfig:
    input_dim: int
    architecture: str = "rq_spline"
    num_transforms: int = 8
    hidden_features: int = 64
    num_blocks: int = 2
    use_residual_blocks: bool = True
    num_bins: int = 8
    tails: str = "linear"
    tail_bound: float = 8.0
    dropout_probability: float = 0.0
    use_batch_norm: bool = False
    base_distribution: str = "standard_normal"
    permutation: str = "reverse"

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any], *, input_dim: int | None = None
    ) -> FlowConfig:
        parsed = dict(values)
        for metadata in ("family", "ablation", "objective", "weight_is_input_feature"):
            parsed.pop(metadata, None)
        aliases = {
            "num_flow_steps": "num_transforms",
            "num_transform_blocks": "num_blocks",
        }
        for legacy, current in aliases.items():
            if legacy in parsed:
                if current in parsed and parsed[current] != parsed[legacy]:
                    raise ValueError(f"Conflicting model keys: {legacy} and {current}")
                parsed[current] = parsed.pop(legacy)
        spline = parsed.pop("spline", {})
        if not isinstance(spline, Mapping):
            raise TypeError("model.spline must be a mapping")
        for key in ("num_bins", "tails", "tail_bound"):
            if key in spline:
                if key in parsed and parsed[key] != spline[key]:
                    raise ValueError(f"Conflicting spline key: {key}")
                parsed[key] = spline[key]
        if input_dim is not None:
            if "input_dim" in parsed and int(parsed["input_dim"]) != input_dim:
                raise ValueError("Configured input_dim does not match feature order")
            parsed["input_dim"] = input_dim
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(parsed) - known)
        if unknown:
            raise ValueError(f"Unknown flow configuration keys: {unknown}")
        if "input_dim" not in parsed:
            raise ValueError("input_dim is required")
        return cls(**parsed)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        spline = {key: values.pop(key) for key in ("num_bins", "tails", "tail_bound")}
        values["spline"] = spline
        return values


def build_flow(config: FlowConfig):
    """Build the exact nflows graph used by the frozen FS Model 4 backend."""

    if config.architecture != "rq_spline":
        raise ValueError("Model 4 supports architecture='rq_spline' only")
    if (
        min(
            config.input_dim,
            config.num_transforms,
            config.hidden_features,
            config.num_blocks,
            config.num_bins,
        )
        <= 0
    ):
        raise ValueError("Flow dimensions and capacities must be positive")
    if config.base_distribution != "standard_normal":
        raise ValueError("Only a standard_normal base distribution is supported")
    if config.permutation != "reverse":
        raise ValueError("Only reverse permutations are supported")
    if config.tails != "linear":
        raise ValueError("RQ-spline tails must be linear")

    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.transforms.autoregressive import (
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    )
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.permutations import ReversePermutation

    transforms = []
    for _ in range(config.num_transforms):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=config.input_dim,
                hidden_features=config.hidden_features,
                num_bins=config.num_bins,
                tails=config.tails,
                tail_bound=config.tail_bound,
                num_blocks=config.num_blocks,
                use_residual_blocks=config.use_residual_blocks,
                random_mask=False,
                dropout_probability=config.dropout_probability,
                use_batch_norm=config.use_batch_norm,
            )
        )
        transforms.append(ReversePermutation(features=config.input_dim))
    return Flow(
        transform=CompositeTransform(transforms),
        distribution=StandardNormal([config.input_dim]),
    )


class NormalizingFlowBackend:
    """Composition wrapper keeping training independent of nflows internals."""

    def __init__(self, config: FlowConfig) -> None:
        self.config = config
        self.model = build_flow(config)

    def log_prob(self, values):
        return self.model.log_prob(values)

    def sample(self, number_samples: int):
        return self.model.sample(number_samples)

    def sample_from_latent(self, latent):
        return self.model._transform.inverse(latent)[0]

    def to(self, device):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def __getattr__(self, name: str):
        return getattr(self.model, name)
