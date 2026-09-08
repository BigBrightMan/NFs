from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from flashsim_nf.models.model4 import sample_weight_summary, weighted_nll

FS_SOURCE = Path(__file__).resolve().parents[2] / "FS" / "src"
sys.path.insert(0, str(FS_SOURCE))

from flashsim.model4 import (  # noqa: E402
    sample_weight_summary as fs_sample_weight_summary,
)
from flashsim.model4 import weighted_nll_from_log_prob as fs_weighted_nll  # noqa: E402


def test_weighted_nll_matches_fs_reference() -> None:
    log_prob = torch.tensor([-2.0, -0.5, -4.0], dtype=torch.float64)
    weights = torch.tensor([0.1, 2.0, 0.4], dtype=torch.float64)

    assert torch.equal(
        weighted_nll(log_prob, weights), fs_weighted_nll(log_prob, weights)
    )


def test_weight_summary_matches_fs_reference() -> None:
    weights = np.array([0.0013, 0.0625, 0.5], dtype=np.float64)
    assert sample_weight_summary(weights) == fs_sample_weight_summary(weights)


@pytest.mark.parametrize(
    "weights",
    [
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, float("nan")]),
    ],
)
def test_invalid_weights_match_fs_failure(weights: torch.Tensor) -> None:
    log_prob = torch.tensor([-1.0, -2.0])
    with pytest.raises(ValueError):
        weighted_nll(log_prob, weights)
    with pytest.raises(ValueError):
        fs_weighted_nll(log_prob, weights)
