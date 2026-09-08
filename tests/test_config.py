from __future__ import annotations

import pytest

from flashsim_nf.config import ConfigError, config_hash, deep_merge, resolve_config


def test_deep_merge_is_recursive_and_does_not_mutate_inputs() -> None:
    base = {"model": {"hidden": 64, "blocks": 2}, "pipelines": ["A"]}
    override = {"model": {"hidden": 96}, "pipelines": ["A", "B"]}

    merged = deep_merge(base, override)

    assert merged == {
        "model": {"hidden": 96, "blocks": 2},
        "pipelines": ["A", "B"],
    }
    assert base["model"]["hidden"] == 64


def test_hash_is_order_independent_and_changes_with_values() -> None:
    first = {"model": {"hidden": 64, "blocks": 2}}
    reordered = {"model": {"blocks": 2, "hidden": 64}}
    changed = {"model": {"hidden": 96, "blocks": 2}}

    assert config_hash(first) == config_hash(reordered)
    assert config_hash(first) != config_hash(changed)


def test_resolve_config_requires_a_layer() -> None:
    with pytest.raises(ConfigError):
        resolve_config([])
