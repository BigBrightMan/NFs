"""Deterministic configuration composition for NFs runs."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when maintained configuration cannot be resolved safely."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping and reject other top-level types."""
    source = Path(path)
    payload = yaml.safe_load(source.read_text())
    if not isinstance(payload, dict):
        raise ConfigError(f"Expected a YAML mapping: {source}")
    return payload


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive mapping merge without mutating either input.

    Mappings merge recursively. Lists and scalar values are replaced as complete
    values so that precedence remains explicit and deterministic.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def resolve_config(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compose ordered config layers; later layers take precedence."""
    if not layers:
        raise ConfigError("At least one configuration layer is required")
    resolved: dict[str, Any] = {}
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise ConfigError("Every configuration layer must be a mapping")
        resolved = deep_merge(resolved, layer)
    return resolved


def canonical_config_json(config: Mapping[str, Any]) -> str:
    """Return the stable JSON representation used for config identity."""
    return json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(config: Mapping[str, Any], *, length: int = 8) -> str:
    """Return a short stable SHA-256 identity for a resolved configuration."""
    if length < 8 or length > 64:
        raise ConfigError("Config hash length must be between 8 and 64")
    digest = hashlib.sha256(canonical_config_json(config).encode()).hexdigest()
    return digest[:length]
