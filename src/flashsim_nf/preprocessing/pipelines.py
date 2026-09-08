"""Composition-based definitions of FlashSim preprocessing A, B, and C."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .transforms import (
    ArcSinh,
    BoxCox,
    Identity,
    LogitClip,
    MinMax,
    PositiveLog,
    RobustScale,
    Shift,
    SignedLog1p,
    Standardize,
    TransformChain,
)

KNOWN_FEATURES = ("x", "y", "z", "E", "pz", "px", "py", "t", "w")


@dataclass
class FeaturePreprocessor:
    """A named mapping of features to composed, fitted transform chains."""

    name: str
    feature_order: tuple[str, ...]
    chains: dict[str, TransformChain]
    fitted: bool = False

    def fit(self, values: np.ndarray) -> FeaturePreprocessor:
        matrix = self._matrix(values)
        if len(matrix) == 0:
            raise ValueError("Cannot fit preprocessing on an empty training split")
        for index, feature in enumerate(self.feature_order):
            self.chains[feature].fit(matrix[:, index])
        self.fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        matrix = self._matrix(values)
        result = np.empty_like(matrix)
        for index, feature in enumerate(self.feature_order):
            result[:, index] = self.chains[feature].forward(matrix[:, index])
        if not np.isfinite(result).all():
            raise ValueError("Preprocessing produced non-finite values")
        return result

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        matrix = self._matrix(values)
        result = np.empty_like(matrix)
        for index, feature in enumerate(self.feature_order):
            result[:, index] = self.chains[feature].inverse(matrix[:, index])
        if not np.isfinite(result).all():
            raise ValueError("Inverse preprocessing produced non-finite values")
        return result

    def to_dict(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "format": "flashsim_nf.composed_preprocessor",
            "version": 1,
            "pipeline": self.name,
            "feature_order": list(self.feature_order),
            "fitted_on": "train",
            "chains": {
                feature: self.chains[feature].state() for feature in self.feature_order
            },
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> FeaturePreprocessor:
        if state.get("format") != "flashsim_nf.composed_preprocessor":
            raise ValueError("Unsupported preprocessing artifact format")
        if state.get("version") != 1:
            raise ValueError("Unsupported preprocessing artifact version")
        features = _validate_features(state["feature_order"])
        serialized = state["chains"]
        return cls(
            name=str(state["pipeline"]),
            feature_order=features,
            chains={
                feature: TransformChain.from_state(serialized[feature])
                for feature in features
            },
            fitted=True,
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> FeaturePreprocessor:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def _matrix(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        expected = len(self.feature_order)
        if matrix.ndim != 2 or matrix.shape[1] != expected:
            raise ValueError(f"Expected shape (N, {expected}), got {matrix.shape}")
        if not np.isfinite(matrix).all():
            raise ValueError("Preprocessing input contains non-finite values")
        return matrix.copy()

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("Preprocessor has not been fitted")


def _validate_features(feature_order: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(feature_order)
    if not result or len(result) != len(set(result)):
        raise ValueError("feature_order must be non-empty and unique")
    unknown = sorted(set(result) - set(KNOWN_FEATURES))
    if unknown:
        raise ValueError(f"Unknown preprocessing features: {unknown}")
    return result


def _finish(*components: object) -> TransformChain:
    return TransformChain([*components, Standardize()])


def _pipeline_a(feature: str, *, logit_epsilon: float) -> TransformChain:
    if feature in {"x", "y", "z"}:
        return _finish(MinMax(), LogitClip(logit_epsilon))
    if feature in {"px", "py"}:
        return _finish(SignedLog1p())
    if feature in {"E", "pz"}:
        return _finish(BoxCox())
    if feature == "w":
        return _finish(PositiveLog())
    return _finish(Identity())


def _pipeline_b(feature: str) -> TransformChain:
    if feature in {"px", "py"}:
        return _finish(RobustScale(center="zero"), ArcSinh())
    if feature in {"E", "pz", "w"}:
        return _finish(PositiveLog())
    return _finish(Identity())


def _pipeline_c(feature: str, *, energy_shift_gev: float) -> TransformChain:
    if feature in {"x", "y", "z"}:
        return _finish(RobustScale(center="median"), ArcSinh())
    if feature in {"px", "py"}:
        return _finish(RobustScale(center="zero"), ArcSinh())
    if feature == "E":
        return _finish(
            Shift(-energy_shift_gev),
            PositiveLog(),
            RobustScale(center="median"),
            ArcSinh(),
        )
    if feature in {"pz", "w"}:
        return _finish(PositiveLog(), RobustScale(center="median"), ArcSinh())
    return _finish(Identity())


def build_preprocessor(
    name: str,
    feature_order: list[str] | tuple[str, ...],
    config: dict[str, Any] | None = None,
) -> FeaturePreprocessor:
    """Construct A/B/C by composing feature-level transform components."""

    pipeline = str(name).upper()
    if pipeline not in {"A", "B", "C"}:
        raise ValueError("preprocessing must be A, B, or C")
    features = _validate_features(feature_order)
    options = dict(config or {})
    epsilon = float(options.get("logit_epsilon", 1.0e-6))
    energy_shift = float(options.get("energy_shift_gev", 10.0))
    builders = {
        "A": lambda feature: _pipeline_a(feature, logit_epsilon=epsilon),
        "B": _pipeline_b,
        "C": lambda feature: _pipeline_c(feature, energy_shift_gev=energy_shift),
    }
    build = builders[pipeline]
    return FeaturePreprocessor(
        name=pipeline,
        feature_order=features,
        chains={feature: build(feature) for feature in features},
    )
