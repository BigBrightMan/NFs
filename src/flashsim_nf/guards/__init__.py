"""Generation-guard fitting and comparison components."""

from .reference import (
    EMPIRICAL_SUPPORT_FEATURES,
    HARD_SUPPORT_FEATURES,
    PHYSICAL_FEATURES,
    ReferenceData,
    compare_reference_guards,
    evaluate_guard,
    fit_reference_guard,
    guard_rejection_masks,
    load_generated_root,
    load_reference_splits,
    weighted_quantile,
)

__all__ = [
    "EMPIRICAL_SUPPORT_FEATURES",
    "HARD_SUPPORT_FEATURES",
    "PHYSICAL_FEATURES",
    "ReferenceData",
    "compare_reference_guards",
    "evaluate_guard",
    "fit_reference_guard",
    "guard_rejection_masks",
    "load_generated_root",
    "load_reference_splits",
    "weighted_quantile",
]
