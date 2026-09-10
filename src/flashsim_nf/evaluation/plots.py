"""Shared publication template for generated-versus-FLUKA evaluation plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FEATURES = ("x", "y", "z", "E", "pz", "px", "py", "t")
FLUKA_COLOR = "#1f77b4"
FLASHSIM_COLOR = "#ff7f0e"
RATIO_COLOR = "#d62728"


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: tuple[float, ...]
) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_weights = weights[order]
    positions = (np.cumsum(ordered_weights) - 0.5 * ordered_weights) / np.sum(
        ordered_weights
    )
    return np.interp(probabilities, positions, ordered_values)


def _display_values(
    matrix: np.ndarray, index: int, *, log_energy: bool = False
) -> tuple[np.ndarray, str, str]:
    values = matrix[:, index]
    feature = FEATURES[index]
    if feature == "E" and log_energy:
        if np.any(values <= 0.0):
            raise ValueError("The log-energy plot requires E > 0")
        return np.log10(values), r"$\log_{10}(E/\mathrm{GeV})$", r"$\log E$"
    labels = {
        "E": (r"$E$ [GeV]", r"$E$"),
        "pz": (r"$p_z$ [GeV]", r"$p_z$"),
        "px": (r"$p_x$ [GeV]", r"$p_x$"),
        "py": (r"$p_y$ [GeV]", r"$p_y$"),
    }
    xlabel, panel_title = labels.get(feature, (feature, feature))
    return values, xlabel, panel_title


def _header(figure, *, title: str, section: str, rows: tuple[int, int]) -> None:
    reference_rows, generated_rows = rows
    figure.text(
        0.012,
        0.992,
        "FlashSim in progress",
        ha="left",
        va="top",
        fontsize=14,
        fontstyle="italic",
        fontweight="bold",
        color="#1f5da8",
    )
    figure.text(
        0.5,
        0.968,
        title,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.944,
        (
            f"{section}  |  FLUKA N={reference_rows:,}  |  "
            f"FlashSim N={generated_rows:,}"
        ),
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )


def _normalized_histogram(
    values: np.ndarray, weights: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = (values >= edges[0]) & (values <= edges[-1])
    counts, _ = np.histogram(values[selected], bins=edges, weights=weights[selected])
    sumw2, _ = np.histogram(
        values[selected], bins=edges, weights=np.square(weights[selected])
    )
    total = float(np.sum(counts))
    widths = np.diff(edges)
    if total <= 0.0:
        empty = np.zeros(len(widths), dtype=np.float64)
        return empty, empty
    density = counts / total / widths
    uncertainty = np.sqrt(sumw2) / total / widths
    return density, uncertainty


def _draw_distribution(
    axis,
    ratio_axis,
    *,
    train_values: np.ndarray,
    train_weights: np.ndarray,
    reference_values: np.ndarray,
    generated_values: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    xlabel: str,
    panel_title: str,
    bulk: bool,
) -> None:
    if bulk:
        lower, upper = _weighted_quantile(
            train_values, train_weights, (0.001, 0.999)
        )
    else:
        combined = np.concatenate([reference_values, generated_values])
        lower, upper = float(np.min(combined)), float(np.max(combined))
    edges = np.linspace(float(lower), float(upper), 61)
    reference_hist, reference_error = _normalized_histogram(
        reference_values, reference_weights, edges
    )
    generated_hist, generated_error = _normalized_histogram(
        generated_values, generated_weights, edges
    )
    axis.stairs(
        reference_hist,
        edges,
        color=FLUKA_COLOR,
        linewidth=1.8,
        label="FLUKA (simulation)",
    )
    axis.stairs(
        generated_hist,
        edges,
        color=FLASHSIM_COLOR,
        linewidth=1.8,
        label="FlashSim (generated)",
    )
    axis.set_ylabel("Density [a.u.]", fontweight="bold")
    axis.set_title(panel_title, fontweight="bold")
    axis.tick_params(axis="x", labelbottom=False)
    if not bulk:
        positive = np.concatenate(
            [reference_hist[reference_hist > 0], generated_hist[generated_hist > 0]]
        )
        if len(positive):
            axis.set_yscale("log")
            axis.set_ylim(bottom=max(float(np.min(positive)) * 0.45, 1.0e-12))

    valid = reference_hist > 0.0
    ratio = np.full_like(reference_hist, np.nan)
    ratio_error = np.full_like(reference_hist, np.nan)
    ratio[valid] = generated_hist[valid] / reference_hist[valid] - 1.0
    ratio_error[valid] = np.sqrt(
        np.square(generated_error[valid] / reference_hist[valid])
        + np.square(
            generated_hist[valid]
            * reference_error[valid]
            / np.square(reference_hist[valid])
        )
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    ratio_axis.errorbar(
        centers[valid],
        ratio[valid],
        yerr=ratio_error[valid],
        color=RATIO_COLOR,
        marker="o",
        markersize=2.8,
        linestyle="none",
        elinewidth=0.7,
        capsize=1.3,
    )
    ratio_axis.axhline(0.0, color="black", linewidth=0.9)
    ratio_axis.set_ylabel(
        "(FlashSim − FLUKA)\n/ FLUKA", fontsize=9, fontweight="bold"
    )
    ratio_axis.set_xlabel(xlabel, fontweight="bold")
    finite = np.isfinite(ratio)
    if finite.any():
        limit = float(np.nanquantile(np.abs(ratio[finite]), 0.98))
        limit = min(max(1.25 * limit, 0.10), 2.0)
        ratio_axis.set_ylim(-limit, limit)


def _feature_page(
    train: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    train_weights: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    *,
    destination: Path,
    title: str,
    bulk: bool,
) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(20, 24))
    outer = figure.add_gridspec(4, 2)
    for index in range(len(FEATURES)):
        cell = outer[index // 2, index % 2].subgridspec(
            2, 1, height_ratios=(3.6, 1.25), hspace=0.05
        )
        axis = figure.add_subplot(cell[0])
        ratio_axis = figure.add_subplot(cell[1], sharex=axis)
        train_values, xlabel, panel_title = _display_values(train, index)
        reference_values, _, _ = _display_values(reference, index)
        generated_values, _, _ = _display_values(generated, index)
        _draw_distribution(
            axis,
            ratio_axis,
            train_values=train_values,
            train_weights=train_weights,
            reference_values=reference_values,
            generated_values=generated_values,
            reference_weights=reference_weights,
            generated_weights=generated_weights,
            xlabel=xlabel,
            panel_title=panel_title,
            bulk=bulk,
        )
        if index == 0:
            axis.legend(frameon=False, fontsize=11)
    _header(
        figure,
        title=title,
        section="Bulk: train-weighted q0.001–q0.999" if bulk else "Tail: full range",
        rows=(len(reference), len(generated)),
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.91), h_pad=1.4, w_pad=1.2)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _energy_log_page(
    train: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    train_weights: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    *,
    destination: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    energy_index = FEATURES.index("E")
    train_values, xlabel, _ = _display_values(train, energy_index, log_energy=True)
    reference_values, _, _ = _display_values(
        reference, energy_index, log_energy=True
    )
    generated_values, _, _ = _display_values(
        generated, energy_index, log_energy=True
    )
    figure = plt.figure(figsize=(18, 8))
    outer = figure.add_gridspec(1, 2)
    for column, bulk in enumerate((True, False)):
        cell = outer[column].subgridspec(
            2, 1, height_ratios=(3.6, 1.25), hspace=0.05
        )
        axis = figure.add_subplot(cell[0])
        ratio_axis = figure.add_subplot(cell[1], sharex=axis)
        _draw_distribution(
            axis,
            ratio_axis,
            train_values=train_values,
            train_weights=train_weights,
            reference_values=reference_values,
            generated_values=generated_values,
            reference_weights=reference_weights,
            generated_weights=generated_weights,
            xlabel=xlabel,
            panel_title="Bulk" if bulk else "Tail",
            bulk=bulk,
        )
        if column == 0:
            axis.legend(frameon=False, fontsize=11)
    _header(
        figure,
        title=title,
        section=r"Energy view: $\log_{10}(E/\mathrm{GeV})$",
        rows=(len(reference), len(generated)),
    )
    figure.tight_layout(rect=(0.02, 0.05, 0.99, 0.89), h_pad=1.0, w_pad=1.4)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _correlation_page(
    report: dict[str, Any], *, kind: str, destination: Path, title: str
) -> None:
    import matplotlib.pyplot as plt

    values = report["evaluation"]["correlations"][kind]
    matrices = [
        np.asarray(values["reference"], dtype=np.float64),
        np.asarray(values["generated"], dtype=np.float64),
        np.asarray(values["difference"], dtype=np.float64),
    ]
    labels = ["FLUKA", "FlashSim", "FlashSim − FLUKA"]
    figure, axes = plt.subplots(1, 3, figsize=(21, 7.5))
    for axis, matrix, label in zip(axes, matrices, labels):
        difference = label == "FlashSim − FLUKA"
        limit = max(0.05, float(np.max(np.abs(matrix)))) if difference else 1.0
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(FEATURES)), FEATURES, rotation=45, ha="right")
        axis.set_yticks(range(len(FEATURES)), FEATURES)
        axis.set_title(label, fontweight="bold", fontsize=14)
        for row in range(len(FEATURES)):
            for column in range(len(FEATURES)):
                value = matrix[row, column]
                text_color = "white" if abs(value) > 0.58 * limit else "black"
                axis.text(
                    column,
                    row,
                    f"{value:.3f}" if difference else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    color=text_color,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    context = report["plot_context"]
    _header(
        figure,
        title=title,
        section=f"Weighted {kind.capitalize()} correlation",
        rows=(context["reference_rows"], context["generated_rows"]),
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.99, 0.89), w_pad=1.4)
    figure.savefig(destination, dpi=180)
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

    figure, axes = plt.subplots(4, 2, figsize=(18, 22))
    for index, axis in enumerate(axes.flat):
        reference_values, xlabel, panel_title = _display_values(reference, index)
        generated_values, _, _ = _display_values(generated, index)
        series = (
            (reference_values, reference_weights, FLUKA_COLOR, "FLUKA (simulation)"),
            (
                generated_values,
                generated_weights,
                FLASHSIM_COLOR,
                "FlashSim (generated)",
            ),
        )
        for values, weights, color, label in series:
            order = np.argsort(values)
            ordered = values[order]
            survival = 1.0 - np.cumsum(weights[order]) / np.sum(weights)
            axis.plot(ordered, survival, color=color, linewidth=1.8, label=label)
        axis.set_yscale("log")
        axis.set_ylim(1.0e-5, 1.0)
        axis.set_xlabel(xlabel, fontweight="bold")
        axis.set_ylabel("Weighted CCDF", fontweight="bold")
        axis.set_title(panel_title, fontweight="bold")
        if index == 0:
            axis.legend(frameon=False, fontsize=11)
    _header(
        figure,
        title=title,
        section="Tail: weighted complementary CDF",
        rows=(len(reference), len(generated)),
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.91), h_pad=1.5, w_pad=1.2)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def write_generated_evaluation_plots(
    *,
    train: np.ndarray,
    reference: np.ndarray,
    generated: np.ndarray,
    train_weights: np.ndarray,
    reference_weights: np.ndarray,
    generated_weights: np.ndarray,
    report: dict[str, Any],
    output_directory: str | Path,
    context: dict[str, Any],
) -> list[str]:
    """Write the shared bulk, tail, correlation, and log-energy plot pages."""

    destination = Path(output_directory) / "plots"
    destination.mkdir(parents=True, exist_ok=False)
    title = (
        f"{context['year']} · {context['reference_split'].capitalize()} evaluation"
        f" · Model 4 · Preprocess {context['preprocessing']} · drop_z_E"
    )
    report["plot_context"] = {
        **context,
        "reference_rows": len(reference),
        "generated_rows": len(generated),
        "template": "model4_evaluation_v2",
    }
    paths = {
        "all_features_bulk_q001_q999.png": lambda path: _feature_page(
            train,
            reference,
            generated,
            train_weights,
            reference_weights,
            generated_weights,
            destination=path,
            title=title,
            bulk=True,
        ),
        "all_features_tail_full_range_logy.png": lambda path: _feature_page(
            train,
            reference,
            generated,
            train_weights,
            reference_weights,
            generated_weights,
            destination=path,
            title=title,
            bulk=False,
        ),
        "energy_log10_bulk_tail.png": lambda path: _energy_log_page(
            train,
            reference,
            generated,
            train_weights,
            reference_weights,
            generated_weights,
            destination=path,
            title=title,
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
