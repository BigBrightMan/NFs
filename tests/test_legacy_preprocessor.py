import json

import numpy as np

from flashsim_nf.preprocessing import load_legacy_preprocessor


def test_load_legacy_pipeline_b_selected_inverse(tmp_path):
    parameters = {
        "version": 3,
        "pipeline": "B",
        "feature_order": ["x", "y", "z", "E", "pz", "px", "py", "t", "w"],
        "fitted": True,
        "parameters": {
            "x": {"kind": "identity", "final_mean": 10.0, "final_std": 2.0},
            "pz": {"kind": "log", "final_mean": 2.0, "final_std": 0.5},
        },
    }
    path = tmp_path / "preprocessing_parameters.json"
    path.write_text(json.dumps(parameters))
    preprocessor = load_legacy_preprocessor(
        path, feature_order=("x", "pz"), expected_pipeline="B"
    )
    physical = np.array([[12.0, np.exp(2.5)], [8.0, np.exp(1.5)]])
    transformed = preprocessor.transform(physical)
    assert np.allclose(transformed, [[1.0, 1.0], [-1.0, -1.0]])
    assert np.allclose(preprocessor.inverse_transform(transformed), physical)
