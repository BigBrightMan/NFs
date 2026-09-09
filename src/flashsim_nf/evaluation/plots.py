"""Compact, publication-oriented generated-versus-FLUKA plot pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FEATURES = ("x", "y", "z", "E", "pz", "px", "py", "t")


def _display_values(matrix: np.ndarray, index: int) -> tuple[np.ndarray, str]:
    values = matrix[:, index]
    feature = FEATURES[index]
    if feature == "E":
        positive_floor = np.finfo(np.float64).tiny
        return (
            np.log10(np.maximum(values, positive_floor)),
            r"$\log_{10}(E/\mathrm{GeV})$",
        )
    return values, feature


def _header(figure, title: str) -> None:
    figure.suptitle(title, fontsize=15, y=0.995)
    figure.text(
        0.012,
        0.988,
        "FlashSim in progress",
        ha="left",
        va="top",
        fontsize=13,
        fontstyle="italic",
        color="#1f5da8",
    )


def _feature_page(
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    *,
    destination: Path,
    title: str,
    core: bool,
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 18), constrained_layout=True)
    outer = figure.add_gridspec(4, 2)
    for index, feature in enumerate(FEATURES):
        cell = outer[index // 2, index % 2].subgridspec(
            2, 1, height_ratios=(3.2, 1.0), hspace=0.05
        )
        axis = figure.add_subplot(cell[0])
        ratio_axis = figure.add_subplot(cell[1], sharex=axis)
        reference_values, xlabel = _display_values(reference, index)
        generated_values, _ = _display_values(generated, index)
        combined = np.concatenate([reference_values, generated_values])
        if core:
            lower, upper = np.quantile(reference_values, (0.001, 0.999))
        else:
            lower, upper = float(np.min(combined)), float(np.max(combined))
        edges = np.linspace(lower, upper, 81)
        reference_hist, _ = np.histogram(
            reference_values,
            bins=edges,
            weights=reference_weights,
            density=True,
        )
        generated_hist, _ = np.histogram(
            generated_values,
            bins=edges,
            weights=generated_weights,
            density=True,
        )
        axis.stairs(reference_hist, edges, color="black", label="FLUKA (simulation)")
        axis.stairs(
            generated_hist,
            edges,
            color="#e87517",
            linestyle="--",
            linewidth=1.7,
            label="FlashSim (generated)",
        )
        axis.set_ylabel("Density [a.u.]")
        axis.set_title(feature)
        if not core:
            axis.set_yscale("log")
        if index == 0:
            axis.legend(frameon=False)
        ratio = np.divide(
            generated_hist - reference_hist,
            reference_hist,
            out=np.full_like(reference_hist, np.nan),
            where=reference_hist > 0,
        )
        ratio_axis.stairs(ratio, edges, color="#e87517")
        ratio_axis.axhline(0.0, color="black", linewidth=0.8)
        ratio_axis.set_ylabel("(FlashSim - FLUKA)\n/ FLUKA", fontsize=8)
        ratio_axis.set_xlabel(xlabel)
        ratio_axis.set_ylim(-1.1, 1.1)
    _header(figure, title)
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def _correlation_page(
    report: dict[str, Any], *, kind: str, destination: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    values = report["evaluation"]["correlations"][kind]
    matrices = [values["reference"], values["generated"], values["difference"]]
    labels = ["FLUKA", "FlashSim", "FlashSim - FLUKA"]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, matrix, label in zip(axes, matrices, labels):
        limit = (
            1.0
            if label != "FlashSim - FLUKA"
            else max(0.05, float(np.max(np.abs(matrix))))
        )
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
        axis.set_yticks(range(len(FEATURES)), FEATURES)
        axis.set_title(label)
        figure.colorbar(image, ax=axis, fraction=0.046)
    _header(figure, f"{title} · {kind.capitalize()} correlation")
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def _ccdf_page(
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    *,
    destination: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
    for index, axis in enumerate(axes.flat):
        reference_values, xlabel = _display_values(reference, index)
        generated_values, _ = _display_values(generated, index)
        for values, weights, color, label, linestyle in (
            (reference_values, reference_weights, "black", "FLUKA (simulation)", "-"),
            (
                generated_values,
                generated_weights,
                "#e87517",
                "FlashSim (generated)",
                "--",
            ),
        ):
            order = np.argsort(values)
            ordered = values[order]
            survival = 1.0 - np.cumsum(weights[order]) / np.sum(weights)
            axis.plot(ordered, survival, color=color, linestyle=linestyle, label=label)
        axis.set_yscale("log")
        axis.set_ylim(1.0e-5, 1.0)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Weighted CCDF")
        axis.set_title(FEATURES[index])
        if index == 0:
            axis.legend(frameon=False)
    _header(figure, f"{title} · Tail CCDF")
    figure.savefig(destination, dpi=170)
    plt.close(figure)


def write_generated_evaluation_plots(
    *,
    reference: np.ndarray,
    generated: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    report: dict[str, Any],
    output_directory: str | Path,
    context: dict[str, Any],
) -> list[str]:
    """Write the default combined core/full, correlation, and tail pages."""

    destination = Path(output_directory) / "plots"
    destination.mkdir(parents=True, exist_ok=False)
    title = (
        f"{context['year']} data · {context['reference_split'].capitalize()} evaluation"
        f" · Model 4 · Preprocess {context['preprocessing']} · drop_z_E"
    )
    paths = {
        "all_features_core_q001_q999.png": lambda path: _feature_page(
            reference,
            generated,
            reference_weights,
            generated_weights,
            destination=path,
            title=f"{title} · Core q0.001-q0.999",
            core=True,
        ),
        "all_features_full_minmax_logy.png": lambda path: _feature_page(
            reference,
            generated,
            reference_weights,
            generated_weights,
            destination=path,
            title=f"{title} · Full min-max",
            core=False,
        ),
        "pearson_correlation.png": lambda path: _correlation_page(
            report, kind="pearson", destination=path, title=title
        ),
        "spearman_correlation.png": lambda path: _correlation_page(
            report, kind="spearman", destination=path, title=title
        ),
        "tail_ccdf.png": lambda path: _ccdf_page(
            reference,
            generated,
            reference_weights,
            generated_weights,
            destination=path,
            title=title,
        ),
    }
    written = []
    for name, render in paths.items():
        path = destination / name
        render(path)
        written.append(str(path.resolve()))
    return written
