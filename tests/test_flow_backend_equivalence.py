from __future__ import annotations

import sys
from pathlib import Path

import torch

from flashsim_nf.models.flows import FlowConfig, build_flow

FS_SOURCE = Path(__file__).resolve().parents[2] / "FS" / "src"
if str(FS_SOURCE) not in sys.path:
    sys.path.insert(0, str(FS_SOURCE))


def test_rq_spline_graph_and_initial_state_match_fs() -> None:
    from flashsim.nf import FlowConfig as FsFlowConfig
    from flashsim.nf import build_flow as build_fs_flow

    values = {
        "input_dim": 7,
        "architecture": "rq_spline",
        "num_transforms": 2,
        "hidden_features": 16,
        "num_blocks": 1,
        "num_bins": 8,
        "tail_bound": 8.0,
    }
    torch.manual_seed(123)
    reference = build_fs_flow(FsFlowConfig.from_dict(values))
    torch.manual_seed(123)
    ported = build_flow(FlowConfig.from_mapping(values))

    reference_state = reference.state_dict()
    ported_state = ported.state_dict()
    assert reference_state.keys() == ported_state.keys()
    for key in reference_state:
        torch.testing.assert_close(reference_state[key], ported_state[key])

    batch = torch.randn(8, 7)
    torch.testing.assert_close(reference.log_prob(batch), ported.log_prob(batch))


def test_model4_metadata_is_not_part_of_backend_configuration() -> None:
    config = FlowConfig.from_mapping(
        {
            "family": "model4",
            "architecture": "rq_spline",
            "ablation": "drop_ze",
            "objective": "weighted_nll",
            "weight_is_input_feature": False,
            "num_transforms": 8,
            "hidden_features": 64,
            "num_blocks": 2,
        },
        input_dim=7,
    )
    assert config.input_dim == 7
    assert config.num_transforms == 8
