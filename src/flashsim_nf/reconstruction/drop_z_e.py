"""Deterministically reconstruct the omitted z and E physical features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

MUON_MASS_GEV = 0.1056583755
PHYSICAL_ORDER = ("x", "y", "z", "E", "pz", "px", "py", "t")
DROP_Z_E_ORDER = ("x", "y", "pz", "px", "py", "t")
DROP_Z_PZ_ORDER = ("x", "y", "E", "px", "py", "t")


@dataclass(frozen=True)
class ScoringPlane:
    intercept: float
    x: float
    y: float
    fit_rows: int
    fit_source: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def fit_scoring_plane(path: str | Path, *, tree_name: str = "nt") -> ScoringPlane:
    """Fit z = intercept + bx*x + by*y on the frozen train split only."""

    try:
        import uproot
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("Scoring-plane fitting requires uproot") from error
    source_path = Path(path).resolve()
    with uproot.open(source_path) as source:
        tree = source[tree_name]
        arrays = tree.arrays(["x", "y", "z"], library="np")
    columns = {
        name: np.asarray(arrays[name], dtype=np.float64) for name in ("x", "y", "z")
    }
    matrix = np.column_stack([np.ones(len(columns["x"])), columns["x"], columns["y"]])
    finite = np.isfinite(matrix).all(axis=1) & np.isfinite(columns["z"])
    if not finite.any():
        raise ValueError("Train reference has no finite scoring-plane rows")
    coefficients, *_ = np.linalg.lstsq(matrix[finite], columns["z"][finite], rcond=None)
    return ScoringPlane(
        intercept=float(coefficients[0]),
        x=float(coefficients[1]),
        y=float(coefficients[2]),
        fit_rows=int(finite.sum()),
        fit_source=str(source_path),
    )


def reconstruct_drop_z_e(
    values: np.ndarray,
    *,
    feature_order: tuple[str, ...] | list[str],
    scoring_plane: ScoringPlane,
    muon_mass_gev: float = MUON_MASS_GEV,
) -> dict[str, np.ndarray]:
    """Return canonical physical features from six generated NF features."""

    order = tuple(feature_order)
    if order != DROP_Z_E_ORDER:
        raise ValueError(f"drop_ze feature order must be {DROP_Z_E_ORDER}, got {order}")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(order):
        raise ValueError(
            f"Expected generated shape (N, {len(order)}), got {matrix.shape}"
        )
    columns = {name: matrix[:, index] for index, name in enumerate(order)}
    columns["E"] = np.sqrt(
        np.square(columns["px"])
        + np.square(columns["py"])
        + np.square(columns["pz"])
        + float(muon_mass_gev) ** 2
    )
    columns["z"] = (
        scoring_plane.intercept
        + scoring_plane.x * columns["x"]
        + scoring_plane.y * columns["y"]
    )
    return {name: np.asarray(columns[name]) for name in PHYSICAL_ORDER}


def reconstruct_drop_z_pz(
    values: np.ndarray,
    *,
    feature_order: tuple[str, ...] | list[str],
    scoring_plane: ScoringPlane,
    muon_mass_gev: float = MUON_MASS_GEV,
) -> dict[str, np.ndarray]:
    """Reconstruct z and positive pz from generated E, px, and py."""

    order = tuple(feature_order)
    if order != DROP_Z_PZ_ORDER:
        raise ValueError(
            f"drop_z_pz feature order must be {DROP_Z_PZ_ORDER}, got {order}"
        )
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(order):
        raise ValueError(
            f"Expected generated shape (N, {len(order)}), got {matrix.shape}"
        )
    columns = {name: matrix[:, index] for index, name in enumerate(order)}
    radicand = (
        np.square(columns["E"])
        - np.square(columns["px"])
        - np.square(columns["py"])
        - float(muon_mass_gev) ** 2
    )
    # Invalid proposals become pz=0 and are rejected by the physical contract.
    columns["pz"] = np.sqrt(np.clip(radicand, 0.0, None))
    columns["z"] = (
        scoring_plane.intercept
        + scoring_plane.x * columns["x"]
        + scoring_plane.y * columns["y"]
    )
    return {name: np.asarray(columns[name]) for name in PHYSICAL_ORDER}


def reconstruct_full_8d(
    values: np.ndarray,
    *,
    feature_order: tuple[str, ...] | list[str],
) -> dict[str, np.ndarray]:
    """Return an already complete canonical 8D generated sample."""

    order = tuple(feature_order)
    if order != PHYSICAL_ORDER:
        raise ValueError(f"8D feature order must be {PHYSICAL_ORDER}, got {order}")
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(order):
        raise ValueError(f"Expected generated shape (N, 8), got {matrix.shape}")
    return {name: matrix[:, index] for index, name in enumerate(order)}
