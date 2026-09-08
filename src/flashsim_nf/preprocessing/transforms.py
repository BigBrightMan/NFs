"""Small invertible transforms composed into preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.special import boxcox, expit, inv_boxcox
from scipy.stats import boxcox as fit_boxcox


class Transform(Protocol):
    """Stateful one-dimensional transform component."""

    def fit(self, values: np.ndarray) -> None: ...

    def forward(self, values: np.ndarray) -> np.ndarray: ...

    def inverse(self, values: np.ndarray) -> np.ndarray: ...

    def state(self) -> dict[str, Any]: ...


def _array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"Expected one-dimensional values, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Transform input contains non-finite values")
    return result


@dataclass
class Identity:
    def fit(self, values: np.ndarray) -> None:
        _array(values)

    def forward(self, values: np.ndarray) -> np.ndarray:
        return _array(values).copy()

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return _array(values).copy()

    def state(self) -> dict[str, Any]:
        return {"kind": "identity"}


@dataclass
class Shift:
    offset: float

    def fit(self, values: np.ndarray) -> None:
        _array(values)

    def forward(self, values: np.ndarray) -> np.ndarray:
        return _array(values) + self.offset

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return _array(values) - self.offset

    def state(self) -> dict[str, Any]:
        return {"kind": "shift", "offset": self.offset}


@dataclass
class PositiveLog:
    def _check(self, values: np.ndarray) -> np.ndarray:
        result = _array(values)
        if np.any(result <= 0):
            raise ValueError("Log transform requires strictly positive values")
        return result

    def fit(self, values: np.ndarray) -> None:
        self._check(values)

    def forward(self, values: np.ndarray) -> np.ndarray:
        return np.log(self._check(values))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.exp(_array(values))

    def state(self) -> dict[str, Any]:
        return {"kind": "positive_log"}


@dataclass
class SignedLog1p:
    def fit(self, values: np.ndarray) -> None:
        _array(values)

    def forward(self, values: np.ndarray) -> np.ndarray:
        result = _array(values)
        return np.sign(result) * np.log1p(np.abs(result))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        result = _array(values)
        return np.sign(result) * np.expm1(np.abs(result))

    def state(self) -> dict[str, Any]:
        return {"kind": "signed_log1p"}


@dataclass
class BoxCox:
    fitted_lambda: float | None = None

    def fit(self, values: np.ndarray) -> None:
        result = _array(values)
        if np.any(result <= 0):
            raise ValueError("Box-Cox requires strictly positive values")
        _, fitted = fit_boxcox(result)
        self.fitted_lambda = float(fitted)

    def _lambda(self) -> float:
        if self.fitted_lambda is None:
            raise RuntimeError("Box-Cox transform has not been fitted")
        return self.fitted_lambda

    def forward(self, values: np.ndarray) -> np.ndarray:
        result = _array(values)
        if np.any(result <= 0):
            raise ValueError("Box-Cox requires strictly positive values")
        return boxcox(result, self._lambda())

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return inv_boxcox(_array(values), self._lambda())

    def state(self) -> dict[str, Any]:
        return {"kind": "box_cox", "lambda": self._lambda()}


@dataclass
class MinMax:
    minimum_scale: float = 1.0e-12
    minimum: float | None = None
    maximum: float | None = None

    def fit(self, values: np.ndarray) -> None:
        result = _array(values)
        self.minimum = float(np.min(result))
        self.maximum = float(np.max(result))
        if self.maximum - self.minimum < self.minimum_scale:
            raise ValueError("Min-max training range is too small")

    def _range(self) -> tuple[float, float]:
        if self.minimum is None or self.maximum is None:
            raise RuntimeError("Min-max transform has not been fitted")
        return self.minimum, self.maximum

    def forward(self, values: np.ndarray) -> np.ndarray:
        minimum, maximum = self._range()
        return (_array(values) - minimum) / (maximum - minimum)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        minimum, maximum = self._range()
        return minimum + _array(values) * (maximum - minimum)

    def state(self) -> dict[str, Any]:
        minimum, maximum = self._range()
        return {"kind": "minmax", "minimum": minimum, "maximum": maximum}


@dataclass
class LogitClip:
    epsilon: float = 1.0e-6

    def __post_init__(self) -> None:
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must be between zero and 0.5")

    def fit(self, values: np.ndarray) -> None:
        _array(values)

    def forward(self, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(_array(values), self.epsilon, 1.0 - self.epsilon)
        return np.log(clipped / (1.0 - clipped))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return expit(_array(values))

    def state(self) -> dict[str, Any]:
        return {"kind": "logit_clip", "epsilon": self.epsilon}


@dataclass
class RobustScale:
    center: str = "median"
    minimum_scale: float = 1.0e-12
    fitted_center: float | None = None
    fitted_scale: float | None = None

    def fit(self, values: np.ndarray) -> None:
        result = _array(values)
        if self.center not in {"median", "zero"}:
            raise ValueError("center must be 'median' or 'zero'")
        self.fitted_center = (
            float(np.median(result)) if self.center == "median" else 0.0
        )
        q25, q75 = np.quantile(result, [0.25, 0.75])
        self.fitted_scale = float(q75 - q25)
        if self.fitted_scale < self.minimum_scale:
            raise ValueError("Training IQR is too small")

    def _parameters(self) -> tuple[float, float]:
        if self.fitted_center is None or self.fitted_scale is None:
            raise RuntimeError("Robust scale has not been fitted")
        return self.fitted_center, self.fitted_scale

    def forward(self, values: np.ndarray) -> np.ndarray:
        center, scale = self._parameters()
        return (_array(values) - center) / scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        center, scale = self._parameters()
        return center + scale * _array(values)

    def state(self) -> dict[str, Any]:
        center, scale = self._parameters()
        return {"kind": "robust_scale", "center": center, "scale": scale}


@dataclass
class ArcSinh:
    def fit(self, values: np.ndarray) -> None:
        _array(values)

    def forward(self, values: np.ndarray) -> np.ndarray:
        return np.arcsinh(_array(values))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.sinh(_array(values))

    def state(self) -> dict[str, Any]:
        return {"kind": "arcsinh"}


@dataclass
class Standardize:
    minimum_scale: float = 1.0e-12
    mean: float | None = None
    standard_deviation: float | None = None

    def fit(self, values: np.ndarray) -> None:
        result = _array(values)
        self.mean = float(np.mean(result))
        self.standard_deviation = float(np.std(result))
        if self.standard_deviation < self.minimum_scale:
            raise ValueError("Training standard deviation is too small")

    def _parameters(self) -> tuple[float, float]:
        if self.mean is None or self.standard_deviation is None:
            raise RuntimeError("Standardizer has not been fitted")
        return self.mean, self.standard_deviation

    def forward(self, values: np.ndarray) -> np.ndarray:
        mean, standard_deviation = self._parameters()
        return (_array(values) - mean) / standard_deviation

    def inverse(self, values: np.ndarray) -> np.ndarray:
        mean, standard_deviation = self._parameters()
        return mean + standard_deviation * _array(values)

    def state(self) -> dict[str, Any]:
        mean, standard_deviation = self._parameters()
        return {
            "kind": "standardize",
            "mean": mean,
            "standard_deviation": standard_deviation,
        }


@dataclass
class TransformChain:
    """Fit and apply independent components in sequence."""

    components: list[Transform] = field(default_factory=list)

    def fit(self, values: np.ndarray) -> None:
        current = _array(values)
        for component in self.components:
            component.fit(current)
            current = component.forward(current)

    def forward(self, values: np.ndarray) -> np.ndarray:
        current = _array(values)
        for component in self.components:
            current = component.forward(current)
        return current

    def inverse(self, values: np.ndarray) -> np.ndarray:
        current = _array(values)
        for component in reversed(self.components):
            current = component.inverse(current)
        return current

    def state(self) -> dict[str, Any]:
        return {
            "kind": "chain",
            "components": [component.state() for component in self.components],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TransformChain:
        if state.get("kind") != "chain":
            raise ValueError("Expected a serialized transform chain")
        return cls([transform_from_state(item) for item in state["components"]])


def transform_from_state(state: dict[str, Any]) -> Transform:
    """Reconstruct one fitted component from a serialized state."""

    kind = state.get("kind")
    if kind == "identity":
        return Identity()
    if kind == "shift":
        return Shift(float(state["offset"]))
    if kind == "positive_log":
        return PositiveLog()
    if kind == "signed_log1p":
        return SignedLog1p()
    if kind == "box_cox":
        return BoxCox(float(state["lambda"]))
    if kind == "minmax":
        return MinMax(minimum=float(state["minimum"]), maximum=float(state["maximum"]))
    if kind == "logit_clip":
        return LogitClip(float(state["epsilon"]))
    if kind == "robust_scale":
        return RobustScale(
            fitted_center=float(state["center"]), fitted_scale=float(state["scale"])
        )
    if kind == "arcsinh":
        return ArcSinh()
    if kind == "standardize":
        return Standardize(
            mean=float(state["mean"]),
            standard_deviation=float(state["standard_deviation"]),
        )
    raise ValueError(f"Unknown serialized transform kind: {kind!r}")
