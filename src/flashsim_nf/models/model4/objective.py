"""Model 4 weighted-density objective ported from the validated FS baseline."""

from __future__ import annotations

import numpy as np


def weighted_nll(log_prob, sample_weights):
    """Return the negative FLUKA-weighted mean log probability."""
    import torch

    values = log_prob.reshape(-1)
    weights = sample_weights.reshape(-1).to(device=values.device, dtype=values.dtype)
    if values.shape != weights.shape:
        raise ValueError("log_prob and sample_weights shapes differ")
    if not torch.isfinite(values).all():
        raise FloatingPointError("Model 4 log probabilities contain non-finite values")
    if not torch.isfinite(weights).all() or bool(torch.any(weights <= 0)):
        raise ValueError("Model 4 requires finite positive FLUKA weights")
    return -(weights * values).sum() / weights.sum()


def sample_weight_summary(weights: np.ndarray) -> dict[str, float | int]:
    """Return positive-weight statistics used by Model 4 diagnostics."""
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("Model 4 weight array is empty")
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Model 4 requires finite positive FLUKA weights")
    total = float(values.sum(dtype=np.float64))
    effective = float(total * total / np.square(values).sum(dtype=np.float64))
    return {
        "rows": len(values),
        "sum_w": total,
        "mean_w": float(values.mean()),
        "minimum_w": float(values.min()),
        "maximum_w": float(values.max()),
        "effective_sample_size": effective,
        "effective_sample_fraction": effective / len(values),
    }
