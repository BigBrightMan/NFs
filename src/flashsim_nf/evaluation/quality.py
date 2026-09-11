"""Shared scalar score contract for Model 4 validation comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np


def selection_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    """Extract the lower-is-better scalars used by validation selection."""

    global_metrics = evaluation["global_multivariate"]
    marginal = evaluation["marginal"]
    bulk = evaluation["bulk"]
    tail_mass_errors = []
    for feature in evaluation["feature_order"]:
        level = evaluation["tails"][feature]["levels"]["q0.9900"]
        for side in ("lower", "upper"):
            values = level[side]
            ratio = values["generated_over_reference_mass"]
            if (
                not values["insufficient_statistics"]
                and ratio is not None
                and ratio > 0
            ):
                tail_mass_errors.append(abs(float(np.log10(ratio))))
    if not tail_mass_errors:
        raise ValueError("No statistically sufficient q0.99 tail metric is available")
    return {
        "frechet_gaussian_distance": float(global_metrics["frechet_gaussian_distance"]),
        "energy_distance": float(
            global_metrics["energy_distance"]["mean_clipped_nonnegative"]
        ),
        "sliced_wasserstein": float(global_metrics["sliced_wasserstein"]["mean"]),
        "mean_weighted_ks": float(
            np.mean([values["weighted_ks"] for values in marginal.values()])
        ),
        "mean_normalized_weighted_wasserstein": float(
            np.mean(
                [
                    values["normalized_weighted_wasserstein"]
                    for values in marginal.values()
                ]
            )
        ),
        "mean_bulk_conditional_weighted_ks": float(
            np.mean([values["conditional_weighted_ks"] for values in bulk.values()])
        ),
        "mean_bulk_conditional_normalized_weighted_wasserstein": float(
            np.mean(
                [
                    values["conditional_normalized_weighted_wasserstein"]
                    for values in bulk.values()
                ]
            )
        ),
        "mean_q99_tail_absolute_log10_mass_ratio": float(np.mean(tail_mass_errors)),
        "c2st_distance_from_half": float(
            global_metrics["c2st"]["distance_from_ideal_half"]
        ),
        "pearson_mean_absolute_difference": float(
            evaluation["correlations"]["pearson"]["mean_absolute_difference"]
        ),
        "spearman_mean_absolute_difference": float(
            evaluation["correlations"]["spearman"]["mean_absolute_difference"]
        ),
    }
