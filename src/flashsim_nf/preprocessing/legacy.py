"""Read frozen FS preprocessing-v3 metadata without importing the FS package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipelines import FeaturePreprocessor, _validate_features
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


def _standardizer(values: dict[str, Any]) -> Standardize:
    return Standardize(
        mean=float(values["final_mean"]),
        standard_deviation=float(values["final_std"]),
    )


def _chain(feature: str, values: dict[str, Any]) -> TransformChain:
    kind = str(values["kind"])
    final = _standardizer(values)
    if kind == "identity":
        components = [Identity(), final]
    elif kind == "minmax_logit":
        components = [
            MinMax(
                minimum=float(values["train_min"]),
                maximum=float(values["train_max"]),
            ),
            LogitClip(float(values["logit_epsilon"])),
            final,
        ]
    elif kind == "signed_log1p":
        components = [SignedLog1p(), final]
    elif kind == "boxcox":
        components = [BoxCox(float(values["boxcox_lambda"])), final]
    elif kind == "log":
        components = [PositiveLog(), final]
    elif kind == "scaled_arcsinh":
        components = [
            RobustScale(
                fitted_center=0.0,
                fitted_scale=float(values["scale"]),
            ),
            ArcSinh(),
            final,
        ]
    elif kind == "robust_arcsinh":
        components = [
            RobustScale(
                fitted_center=float(values["robust_center"]),
                fitted_scale=float(values["robust_scale"]),
            ),
            ArcSinh(),
            final,
        ]
    elif kind == "log_robust_arcsinh":
        components = [
            PositiveLog(),
            RobustScale(
                fitted_center=float(values["robust_center"]),
                fitted_scale=float(values["robust_scale"]),
            ),
            ArcSinh(),
            final,
        ]
    elif kind == "shifted_log_robust_arcsinh":
        components = [
            Shift(-float(values["energy_shift_gev"])),
            PositiveLog(),
            RobustScale(
                fitted_center=float(values["robust_center"]),
                fitted_scale=float(values["robust_scale"]),
            ),
            ArcSinh(),
            final,
        ]
    else:
        raise ValueError(f"Unsupported legacy transform for {feature!r}: {kind!r}")
    return TransformChain(components)


def load_legacy_preprocessor(
    path: str | Path,
    *,
    feature_order: list[str] | tuple[str, ...],
    expected_pipeline: str | None = None,
) -> FeaturePreprocessor:
    """Adapt one immutable FS ``preprocessing_parameters.json`` artifact.

    Only the requested ordered subset is materialized. This is the generation
    bridge for checkpoints trained from legacy prepared ROOT files; no joblib
    unpickling or import from the old FS repository is required.
    """

    source = Path(path)
    state = json.loads(source.read_text())
    if int(state.get("version", -1)) != 3 or not state.get("fitted"):
        raise ValueError(f"Unsupported or unfitted legacy preprocessor: {source}")
    pipeline = str(state.get("pipeline", "")).upper()
    if pipeline not in {"A", "B", "C"}:
        raise ValueError(f"Unknown legacy preprocessing pipeline: {pipeline!r}")
    if expected_pipeline and pipeline != str(expected_pipeline).upper():
        raise ValueError(
            f"Preprocessing pipeline mismatch: {pipeline} != {expected_pipeline}"
        )
    canonical = tuple(state.get("feature_order", ()))
    selected = _validate_features(feature_order)
    if [name for name in canonical if name in selected] != list(selected):
        raise ValueError("Requested features are not an ordered legacy feature subset")
    parameters = state.get("parameters", {})
    missing = sorted(set(selected) - set(parameters))
    if missing:
        raise ValueError(f"Legacy preprocessing parameters are missing {missing}")
    return FeaturePreprocessor(
        name=pipeline,
        feature_order=selected,
        chains={feature: _chain(feature, parameters[feature]) for feature in selected},
        fitted=True,
    )


def load_preprocessor(
    path: str | Path,
    *,
    feature_order: list[str] | tuple[str, ...],
    expected_pipeline: str | None = None,
) -> FeaturePreprocessor:
    """Load either a native composed artifact or frozen legacy metadata."""

    source = Path(path)
    state = json.loads(source.read_text())
    if state.get("format") == "flashsim_nf.composed_preprocessor":
        complete = FeaturePreprocessor.from_dict(state)
        if expected_pipeline and complete.name != str(expected_pipeline).upper():
            raise ValueError("Preprocessing pipeline mismatch")
        selected = _validate_features(feature_order)
        if [name for name in complete.feature_order if name in selected] != list(
            selected
        ):
            raise ValueError(
                "Requested features are not an ordered preprocessing subset"
            )
        return FeaturePreprocessor(
            name=complete.name,
            feature_order=selected,
            chains={name: complete.chains[name] for name in selected},
            fitted=True,
        )
    return load_legacy_preprocessor(
        source,
        feature_order=feature_order,
        expected_pipeline=expected_pipeline,
    )
