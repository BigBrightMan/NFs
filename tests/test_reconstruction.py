import numpy as np

from flashsim_nf.reconstruction import (
    ScoringPlane,
    reconstruct_drop_z_e,
    reconstruct_drop_z_pz,
)


def test_drop_z_e_reconstruction_obeys_physics_contract():
    plane = ScoringPlane(4.0, 2.0, -3.0, 20, "/train.root")
    values = np.array([[1.0, 2.0, 10.0, 3.0, 4.0, 8.0]])
    output = reconstruct_drop_z_e(
        values,
        feature_order=("x", "y", "pz", "px", "py", "t"),
        scoring_plane=plane,
    )
    assert np.allclose(output["z"], 0.0)
    assert np.allclose(
        output["E"] ** 2 - output["px"] ** 2 - output["py"] ** 2 - output["pz"] ** 2,
        0.1056583755**2,
    )


def test_drop_z_pz_reconstruction_obeys_physics_contract():
    plane = ScoringPlane(4.0, 2.0, -3.0, 20, "/train.root")
    mass = 0.1056583755
    pz = 10.0
    energy = np.sqrt(3.0**2 + 4.0**2 + pz**2 + mass**2)
    values = np.array([[1.0, 2.0, energy, 3.0, 4.0, 8.0]])
    output = reconstruct_drop_z_pz(
        values,
        feature_order=("x", "y", "E", "px", "py", "t"),
        scoring_plane=plane,
    )
    assert np.allclose(output["z"], 0.0)
    assert np.allclose(output["pz"], pz)
