"""Safe, short, stable identifiers for runs and artifacts."""

from __future__ import annotations

import re

_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def validate_slug(value: str, *, field: str) -> str:
    """Validate a path-safe identifier used by configs and manifests."""
    if not _SLUG.fullmatch(value):
        raise ValueError(f"{field} must match {_SLUG.pattern!r}; received {value!r}")
    return value


def build_run_id(
    *,
    year: int,
    model: str,
    preprocessing: str,
    trial_id: str,
    training_seed: int,
    config_hash: str,
) -> str:
    """Build a compact stable run identifier."""
    if year < 2000 or year > 2099:
        raise ValueError(f"Unsupported year: {year}")
    model_slug = validate_slug(model, field="model")
    trial_slug = validate_slug(trial_id, field="trial_id")
    hash_slug = validate_slug(config_hash, field="config_hash")
    preprocessing_slug = preprocessing.lower()
    if preprocessing_slug not in {"a", "b", "c", "d", "e"}:
        raise ValueError("preprocessing must be A, B, C, D, or E")
    if training_seed < 0:
        raise ValueError("training_seed must be non-negative")
    return (
        f"fs{year % 100:02d}-{model_slug}-{preprocessing.upper()}-"
        f"{trial_slug}-s{training_seed}-{hash_slug}"
    )
