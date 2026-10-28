#!/usr/bin/env python
"""Compare the refinement variants of one pose-prior benchmark scenario side by side.

Each variant is one ``evaluate_run.py`` output directory (``metrics.json``, ``fsc_curves.csv``,
``fsc_by_iteration.csv`` and ``pose_errors.npz``) of the same
dataset. ``variants.csv`` has one row per variant: the prior that scored it, the iterations and final HEALPix order,
the resolutions and area under the FSC curve against the ground truth and between the half maps, the pose
direction error, and the final learned prior's accuracy. ``variants.png`` shows the ground-truth FSC curves, the
FSC area, the distribution of the particles' direction errors, the learned prior's divergence from the true prior
and, per iteration, the half maps' FSC area to the truth (so runs that stop at different iterations or orders
compare at equal iteration).

Run::

    python scripts/pose_prior_benchmark/compare_variants.py --runs learned=EVAL_DIR fixed_true=EVAL_DIR \\
        --out-dir OUT --label kent_cap
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pose_prior_benchmark.evaluate_run import HALVES, PALETTE  # noqa: E402

# One colour per variant in the order given (validated for colour-vision deficiency on the light surface).
VARIANT_COLORS = ("#2a78d6", "#eb6834", "#2f9e6e", "#8a5cd6")
GT_FSC_THRESHOLDS = (0.5, 0.143)
POSE_ERROR_RANGE_DEG = (0.1, 180.0)  # the error axis; a smaller error counts in its first bin
POSE_ERROR_BINS = 200
POSE_ERROR_SMOOTHING_DECADES = 0.05
GROSS_POSE_ERROR_DEG = 90.0
COLUMNS = (
    "variant",
    "direction_prior",
    "iterations",
    "final_healpix_order",
    "reported_final_A",
    "gt_fsc_auc",
    "gt_resolution_0.5_A",
    "gt_resolution_0.143_A",
    "half_map_resolution_0.143_A",
    "half_map_fsc_auc",
    "pose_error_median_deg",
    "pose_error_p90_deg",
    "pose_error_fraction_over_90",
    "learned_prior_js_bits",
    "learned_prior_js_skill",
)


def parse_runs(items: list[str]) -> dict[str, Path]:
    """``label=directory`` items as an ordered mapping."""
    runs = {}
    for item in items:
        label, separator, directory = item.partition("=")
        if not separator or not label or not directory:
            raise SystemExit(f"--runs wants LABEL=DIRECTORY items, got {item!r}")
        runs[label] = Path(directory)
    return runs


def half_mean(values: dict[str, dict[str, Any]], key: str) -> float | None:
    """The mean of ``key`` over the two half sets, None when a half has none."""
    parts = [values[f"half{half}"][key] for half in HALVES]
    return None if any(part is None for part in parts) else float(np.mean(parts))


def summary_row(variant: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """One variant's row of ``COLUMNS`` from its ``metrics.json``."""
    resolution = metrics["resolution"]
    pose = metrics["pose_direction_error_deg"]
    prior = metrics["prior_final"]
    used = metrics["run"]["direction_prior_used"]
    return {
        "variant": variant,
        "direction_prior": used["source"],
        "iterations": metrics["run"]["n_iterations"],
        "final_healpix_order": metrics["run"]["final_healpix_order"],
        "reported_final_A": resolution["reported"]["final_all_data_A"],
        "gt_fsc_auc": resolution["gt_merged"]["auc"],
        "gt_resolution_0.5_A": resolution["gt_merged"]["at_0.5"]["first_A"],
        "gt_resolution_0.143_A": resolution["gt_merged"]["at_0.143"]["first_A"],
        "half_map_resolution_0.143_A": resolution["half_maps"]["at_0.143"]["first_A"],
        "half_map_fsc_auc": resolution["half_maps"]["auc"],
        "pose_error_median_deg": half_mean(pose, "median"),
        "pose_error_p90_deg": half_mean(pose, "p90"),
        "pose_error_fraction_over_90": half_mean(pose, "fraction_over_90"),
        "learned_prior_js_bits": half_mean(prior, "js_bits"),
        "learned_prior_js_skill": half_mean(prior, "js_skill"),
    }


def read_gt_curve(evaluation_dir: Path, metrics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The final map's FSC to the ground truth: spatial frequency (1/A) and correlation per shell."""
    with open(evaluation_dir / "fsc_curves.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    shells = np.array([int(row["shell"]) for row in rows], dtype=float)
    fsc = np.array([float(row["gt_merged"]) for row in rows])
    return shells / (metrics["resolution"]["box"] * metrics["resolution"]["voxel_size_A"]), fsc


def read_iteration_curve(evaluation_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The run's ``fsc_by_iteration.csv``: iteration, HEALPix order and the half maps' mean FSC area to the truth."""
    with open(evaluation_dir / "fsc_by_iteration.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (
        np.array([int(row["iteration"]) for row in rows]),
        np.array([int(row["healpix_order"]) for row in rows]),
        np.array([float(row["gt_auc"]) for row in rows]),
    )


def read_pose_errors(evaluation_dir: Path) -> np.ndarray:
    """Every particle's final direction error (degrees) of the run, both half sets, from ``pose_errors.npz``."""
    with np.load(evaluation_dir / "pose_errors.npz", allow_pickle=False) as archive:
        return np.concatenate([archive[f"half{half}"] for half in HALVES]).astype(np.float64)


def error_density(errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The density of ``errors`` (degrees) per decade on a log-spaced grid over ``POSE_ERROR_RANGE_DEG``.

    Returns the grid (degrees) and the density, a Gaussian-smoothed histogram of log10(error) that integrates to 1
    over log10(degrees); the ends reflect, so the mass at 0 and 180 degrees stays on the axis.
    """
    low, high = np.log10(POSE_ERROR_RANGE_DEG)
    edges = np.linspace(low, high, POSE_ERROR_BINS + 1)
    step = edges[1] - edges[0]
    counts, _ = np.histogram(np.log10(np.clip(errors, POSE_ERROR_RANGE_DEG[0], POSE_ERROR_RANGE_DEG[1])), bins=edges)
    reach = int(np.ceil(4 * POSE_ERROR_SMOOTHING_DECADES / step))
    kernel = np.exp(-0.5 * (np.arange(-reach, reach + 1) * step / POSE_ERROR_SMOOTHING_DECADES) ** 2)
    smooth = np.convolve(
        np.pad(counts.astype(np.float64), reach, mode="symmetric"), kernel / kernel.sum(), mode="valid"
    )
    return 10 ** (0.5 * (edges[:-1] + edges[1:])), smooth / (errors.size * step)


def write_table(rows: list[dict[str, Any]], path: Path) -> None:
    """Write ``rows`` as a CSV with the columns of ``COLUMNS`` (a value that never happened is empty)."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def tick_label(variant: str) -> str:
    """A variant name on two lines where it has a ``fixed_`` prefix."""
    return variant.replace("fixed_", "fixed\n", 1)


def style_axes(ax) -> None:
    ax.grid(color=PALETTE["grid"], linewidth=0.6, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def plot_comparison(
    rows: list[dict[str, Any]],
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    iteration_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    pose_errors: dict[str, np.ndarray],
    label: str,
    path: Path,
) -> None:
    """The comparison figure of ``rows`` (one per variant), their ground-truth FSC ``curves``, their
    per-iteration half-map FSC areas (``iteration_curves``: iterations, HEALPix orders, areas) and their
    particles' direction errors (``pose_errors``, degrees)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.lines import Line2D

    if len(rows) > len(VARIANT_COLORS):
        raise ValueError(f"{len(rows)} variants; the palette is validated for {len(VARIANT_COLORS)}")
    variants = [row["variant"] for row in rows]
    colors = dict(zip(variants, VARIANT_COLORS))
    fig = plt.figure(figsize=(11, 11.6), facecolor=PALETTE["surface"])
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85])
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(2)])
    trace_ax = fig.add_subplot(grid[2, :])
    for ax in (*axes.flat, trace_ax):
        ax.set_facecolor(PALETTE["surface"])
        ax.tick_params(colors=PALETTE["ink2"])
    fig.suptitle(f"{label}: refinement variants", x=0.01, ha="left", color=PALETTE["ink"])
    x = np.arange(len(variants))
    labels = [tick_label(variant) for variant in variants]

    ax = axes[0, 0]
    style_axes(ax)
    for variant in variants:
        frequency, fsc = curves[variant]
        ax.plot(frequency, fsc, color=colors[variant], linewidth=2)
    for threshold in GT_FSC_THRESHOLDS:
        ax.axhline(threshold, color=PALETTE["reference"], linewidth=1, linestyle=":")
        ax.text(ax.get_xlim()[1], threshold, f"{threshold}", ha="right", va="bottom", color=PALETTE["ink2"], fontsize=8)
    ax.set_title("FSC of the final map to the ground truth", color=PALETTE["ink"])
    ax.set_xlabel("spatial frequency (1/A)", color=PALETTE["ink2"])
    ax.set_ylabel("FSC", color=PALETTE["ink2"])

    def bars(ax, key: str, title: str, ylabel: str, title_pad: float = 6.0) -> None:
        style_axes(ax)
        values = [row[key] for row in rows]
        ax.bar(x, [np.nan if v is None else v for v in values], color=[colors[v] for v in variants], width=0.6)
        for position, value in zip(x, values):
            if value is not None:
                ax.text(position, value, f"{value:.3g}", ha="center", va="bottom", color=PALETTE["ink2"], fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_title(title, color=PALETTE["ink"], pad=title_pad)
        ax.set_ylabel(ylabel, color=PALETTE["ink2"])
        ax.margins(y=0.15)

    bars(axes[0, 1], "gt_fsc_auc", "Area under the ground-truth FSC curve", "normalized area")

    ax = axes[1, 0]
    style_axes(ax)
    densities = {variant: error_density(pose_errors[variant]) for variant in variants}
    half_width = 0.42 / max(density.max() for _, density in densities.values())
    for position, variant in zip(x, variants):
        grid, density = densities[variant]
        errors = pose_errors[variant]
        inside = (grid >= errors.min()) & (grid <= errors.max())
        ax.fill_betweenx(
            grid[inside],
            position - half_width * density[inside],
            position + half_width * density[inside],
            facecolor=colors[variant],
            edgecolor=colors[variant],
            alpha=0.35,
            linewidth=1.5,
        )
        median, p90 = np.median(errors), np.percentile(errors, 90)
        ax.plot(position, median, "o", color=colors[variant], markeredgecolor=PALETTE["surface"], markersize=8)
        ax.plot(position, p90, "o", markerfacecolor=PALETTE["surface"], markeredgecolor=colors[variant], markersize=8)
        ax.text(
            position,
            1.01,
            f"{100 * np.mean(errors > GROSS_POSE_ERROR_DEG):.1f}% > {GROSS_POSE_ERROR_DEG:g}\N{DEGREE SIGN}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=PALETTE["ink2"],
            fontsize=8,
        )
    ax.axhline(GROSS_POSE_ERROR_DEG, color=PALETTE["reference"], linewidth=1, linestyle=":")
    ax.set_yscale("log")
    ax.set_ylim(POSE_ERROR_RANGE_DEG)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(variants) - 0.4)
    ax.set_title("Distribution of the direction error of the final poses", color=PALETTE["ink"], pad=18)
    ax.set_ylabel("degrees (log scale)", color=PALETTE["ink2"])

    bars(
        axes[1, 1],
        "learned_prior_js_bits",
        "Learned prior: divergence from the true prior",
        "JS (bits); fixed runs: estimated, not used",
        title_pad=18,
    )

    style_axes(trace_ax)
    for variant in variants:
        iterations, orders, area = iteration_curves[variant]
        trace_ax.plot(iterations, area, color=colors[variant], linewidth=2)
        raised = np.flatnonzero(np.diff(orders, prepend=orders[0]) > 0)
        trace_ax.plot(iterations[raised], area[raised], "o", color=colors[variant], markersize=6)
        trace_ax.text(
            iterations[-1] + 0.15, area[-1], f"order {orders[-1]}", va="center", color=PALETTE["ink2"], fontsize=8
        )
    trace_ax.set_title("Half maps' FSC area to the ground truth, per iteration", color=PALETTE["ink"])
    trace_ax.set_xlabel("iteration", color=PALETTE["ink2"])
    trace_ax.set_ylabel("normalized area", color=PALETTE["ink2"])
    trace_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    trace_ax.margins(x=0.08)

    legend = [Line2D([], [], color=colors[v], linewidth=2, label=v) for v in variants] + [
        Line2D([], [], marker="o", color=PALETTE["ink2"], linestyle="", label="HEALPix order raised"),
        Line2D([], [], marker="o", color=PALETTE["ink2"], linestyle="", label="median error"),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor=PALETTE["surface"],
            markeredgecolor=PALETTE["ink2"],
            linestyle="",
            label="90th percentile",
        ),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, labelcolor=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="LABEL=EVALUATION_DIR of each variant")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="scenario")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Write ``variants.csv`` and ``variants.png`` for the evaluated variants."""
    args = parse_args(argv)
    runs = parse_runs(args.runs)
    metrics = {variant: json.loads((directory / "metrics.json").read_text()) for variant, directory in runs.items()}
    rows = [summary_row(variant, metrics[variant]) for variant in runs]
    curves = {variant: read_gt_curve(directory, metrics[variant]) for variant, directory in runs.items()}
    iteration_curves = {variant: read_iteration_curve(directory) for variant, directory in runs.items()}
    pose_errors = {variant: read_pose_errors(directory) for variant, directory in runs.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(rows, args.out_dir / "variants.csv")
    plot_comparison(rows, curves, iteration_curves, pose_errors, args.label, args.out_dir / "variants.png")
    for row in rows:
        print(
            f"{row['variant']}: {row['direction_prior']} prior, {row['iterations']} iterations, "
            f"GT-FSC area {row['gt_fsc_auc']:.3f}, median pose error {row['pose_error_median_deg']:.2f} deg"
        )


if __name__ == "__main__":
    main()
