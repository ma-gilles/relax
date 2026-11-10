#!/usr/bin/env python
"""Compare the refinement variants of one pose-prior benchmark scenario side by side.

Each variant is one ``evaluate_run.py`` output directory (``metrics.json``, ``fsc_curves.csv``,
``fsc_by_iteration.csv``, ``pose_errors.npz`` and ``conical_fsc.npz``) of the same
dataset. ``variants.csv`` has one row per variant: the prior that scored it, the iterations and final HEALPix order,
the resolutions and area under the FSC curve against the ground truth and between the half maps, the pose
direction error (``pose_error_*``: the angle between the true and the estimated viewing directions, the out-of-plane
part of the pose error) and in-plane error (``inplane_error_*``: the angle about the viewing direction that remains
once the directions agree; ``evaluate_run.py``), the final learned prior's accuracy and the frame offsets. The
ground-truth FSC and the pose errors are those of the final map and poses aligned to the ground truth's frame (``_aligned``: the map rigidly aligned to the
truth, the poses turned back by their average offset), with the values as refined beside them (``_as_refined``), so
runs that start from maps in different frames compare. The conical FSC (cFSC, ``scripts/conical_fsc.py``) adds the cFAR of
the aligned map against the truth (``gt_cfar_aligned``; the smallest over the largest area under the cones' curves, 1 for
an isotropic map), the resolution of its worst cone and the half maps' cFAR. ``variants.png`` shows the
aligned ground-truth FSC curves, the FSC area, the aligned map's cFSC (mean +- 1 SD over the cones; dashed, the best and the worst cone: the largest and the smallest area
under the cone's curve, whose ratio is the cFAR) and cFAR (hollow:
the half maps'), the distributions of the particles' aligned direction (out-of-plane) and in-plane errors, the learned
prior's divergence from the true prior (a distribution over viewing directions, so out of plane only) and, per iteration, the half maps' FSC area to the truth (as refined) and to each other and the refinement's
Fourier radius (so runs that stop at different iterations or orders compare at equal iteration).

``--reconstructions`` (relion_reconstruct maps from each variant's final poses), ``--angle-reconstructions`` (from
their angles alone, with zero origins) and ``--true-reconstruction`` (from the true poses) take
``reconstruct_with_poses.py`` output directories. Given together they add a figure row with those maps' FSC to the
ground truth beside the refinement's own maps, and write ``reconstructions.csv`` with their resolutions and FSC areas.

Run::

    python scripts/pose_prior_benchmark/compare_variants.py --runs learned=EVAL_DIR fixed_true=EVAL_DIR \\
        --out-dir OUT --label kent_cap [--reconstructions learned=DIR ... --angle-reconstructions learned=DIR ...
        --true-reconstruction DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.conical_fsc import ConicalFsc, axes_rot_tilt_deg, cone_areas, measured_curves  # noqa: E402
from scripts.pose_prior_benchmark.conical_fsc_plot import curve_statistics  # noqa: E402
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
    "gt_fsc_auc_aligned",
    "gt_resolution_0.5_A_aligned",
    "gt_resolution_0.143_A_aligned",
    "gt_fsc_auc_as_refined",
    "half_map_resolution_0.143_A",
    "half_map_fsc_auc",
    "gt_cfar_aligned",
    "gt_worst_cone_resolution_0.143_A_aligned",
    "half_map_cfar",
    "pose_error_median_aligned_deg",
    "pose_error_p90_aligned_deg",
    "pose_error_fraction_over_90_aligned",
    "pose_error_median_as_refined_deg",
    "inplane_error_median_aligned_deg",
    "inplane_error_p90_aligned_deg",
    "pose_offset_deg",
    "map_offset_deg",
    "map_shift_A",
    "learned_prior_js_bits",
    "learned_prior_js_skill",
)
RECONSTRUCTION_COLUMNS = ("poses", "variant", "gt_fsc_auc", "gt_resolution_0.5_A", "gt_resolution_0.143_A")


class IterationCurve(NamedTuple):
    """A run's ``fsc_by_iteration.csv`` columns: the iteration, its HEALPix order, its Fourier radius (pixels), the
    half maps' mean FSC area to the truth and the area of the FSC between the two half maps."""

    iteration: np.ndarray
    order: np.ndarray
    radius: np.ndarray
    gt_area: np.ndarray
    half_area: np.ndarray


class ConicalCurve(NamedTuple):
    """The aligned final map's cFSC to the ground truth over the cones: spatial frequency (1/A), mean and SD per
    shell over the cones every cone measures, and the curves of the best and the worst cone (the largest and the
    smallest wAuC, whose ratio is the cFAR); NaN in the shells where a cone is not measured."""

    frequency: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    best: np.ndarray
    worst: np.ndarray


class Reconstruction(NamedTuple):
    """A ``reconstruct_with_poses.py`` map's FSC to the ground truth: its summary and curve (1/A, FSC)."""

    summary: dict[str, Any]
    frequency: np.ndarray
    fsc: np.ndarray


class Reconstructions(NamedTuple):
    """The relion_reconstruct maps of a scenario: from each variant's final poses (with their origins, and the
    angles alone with zero origins) and from the true poses."""

    estimated: dict[str, Reconstruction]
    angles: dict[str, Reconstruction]
    true: Reconstruction


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
    if "conical_fsc" not in metrics:
        raise ValueError(f"{variant}: metrics.json has no conical_fsc; rerun evaluate_run.py on the run")
    conical = metrics["conical_fsc"]
    pose = metrics["pose_direction_error_aligned_deg"]
    if "pose_inplane_error_aligned_deg" not in metrics:
        raise ValueError(f"{variant}: metrics.json has no in-plane pose error; rerun evaluate_run.py on the run")
    inplane = metrics["pose_inplane_error_aligned_deg"]
    prior = metrics["prior_final"]
    used = metrics["run"]["direction_prior_used"]
    return {
        "variant": variant,
        "direction_prior": used["source"],
        "iterations": metrics["run"]["n_iterations"],
        "final_healpix_order": metrics["run"]["final_healpix_order"],
        "reported_final_A": resolution["reported"]["final_all_data_A"],
        "gt_fsc_auc_aligned": resolution["gt_merged_aligned"]["auc"],
        "gt_resolution_0.5_A_aligned": resolution["gt_merged_aligned"]["at_0.5"]["first_A"],
        "gt_resolution_0.143_A_aligned": resolution["gt_merged_aligned"]["at_0.143"]["first_A"],
        "gt_fsc_auc_as_refined": resolution["gt_merged"]["auc"],
        "half_map_resolution_0.143_A": resolution["half_maps"]["at_0.143"]["first_A"],
        "half_map_fsc_auc": resolution["half_maps"]["auc"],
        "gt_cfar_aligned": conical["gt_merged_aligned"]["cfar"],
        "gt_worst_cone_resolution_0.143_A_aligned": conical["gt_merged_aligned"]["worst_resolution_A"],
        "half_map_cfar": conical["half_maps"]["cfar"],
        "pose_error_median_aligned_deg": half_mean(pose, "median"),
        "pose_error_p90_aligned_deg": half_mean(pose, "p90"),
        "pose_error_fraction_over_90_aligned": half_mean(pose, "fraction_over_90"),
        "pose_error_median_as_refined_deg": half_mean(metrics["pose_direction_error_deg"], "median"),
        "inplane_error_median_aligned_deg": half_mean(inplane, "median"),
        "inplane_error_p90_aligned_deg": half_mean(inplane, "p90"),
        "pose_offset_deg": metrics["pose_frame_offset"]["angle_deg"],
        "map_offset_deg": resolution["alignment"]["angle_deg"],
        "map_shift_A": resolution["alignment"]["shift_A"],
        "learned_prior_js_bits": half_mean(prior, "js_bits"),
        "learned_prior_js_skill": half_mean(prior, "js_skill"),
    }


def read_gt_curve(evaluation_dir: Path, metrics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The final map's FSC to the ground truth after the map is aligned to it: spatial frequency (1/A) and correlation
    per shell."""
    with open(evaluation_dir / "fsc_curves.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    shells = np.array([int(row["shell"]) for row in rows], dtype=float)
    fsc = np.array([float(row["gt_merged_aligned"]) for row in rows])
    return shells / (metrics["resolution"]["box"] * metrics["resolution"]["voxel_size_A"]), fsc


def read_conical_curve(evaluation_dir: Path, metrics: dict[str, Any]) -> ConicalCurve:
    """The aligned final map's cFSC to the ground truth from ``conical_fsc.npz``: its mean and SD over the cones
    in the shells every cone measures, and the curves of its best and worst cone (the axes ``metrics.json`` names)."""
    summary = metrics["conical_fsc"]["gt_merged_aligned"]
    with np.load(evaluation_dir / "conical_fsc.npz", allow_pickle=False) as archive:
        cfsc = ConicalFsc(
            axes=archive["axes"],
            half_angle_deg=summary["half_angle_deg"],
            curves=archive["gt_merged_aligned_curves"],
            voxels=archive["gt_merged_aligned_voxels"],
        )
    stats = curve_statistics(cfsc, summary["min_cone_voxels"], summary["first_shell_measured_by_all_cones"])
    wauc, _ = cone_areas(cfsc, summary["threshold"], summary["min_cone_voxels"])
    worst, best = int(np.argmin(wauc)), int(np.argmax(wauc))
    rot, tilt = axes_rot_tilt_deg(cfsc.axes)
    for index, axis in ((worst, "min_axis"), (best, "max_axis")):
        if not np.allclose([rot[index], tilt[index]], [summary[axis]["rot_deg"], summary[axis]["tilt_deg"]]):
            raise ValueError(f"{evaluation_dir}: conical_fsc.npz and metrics.json disagree on the cone of {axis}")
    curves = measured_curves(cfsc, summary["min_cone_voxels"])
    frequency = np.arange(cfsc.curves.shape[1]) / (metrics["resolution"]["box"] * metrics["resolution"]["voxel_size_A"])
    return ConicalCurve(frequency, stats["mean"], stats["std"], curves[best], curves[worst])


def read_iteration_curve(evaluation_dir: Path) -> IterationCurve:
    """The run's ``fsc_by_iteration.csv`` as an :class:`IterationCurve`."""
    with open(evaluation_dir / "fsc_by_iteration.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return IterationCurve(
        np.array([int(row["iteration"]) for row in rows]),
        np.array([int(row["healpix_order"]) for row in rows]),
        np.array([int(row["fourier_radius"]) for row in rows]),
        np.array([float(row["gt_auc"]) for row in rows]),
        np.array([float(row["half_map_auc"]) for row in rows]),
    )


def read_reconstruction(directory: Path) -> Reconstruction:
    """A ``reconstruct_with_poses.py`` output directory: ``metrics.json`` and ``fsc_curve.csv``."""
    metrics = json.loads((directory / "metrics.json").read_text())["resolution"]
    with open(directory / "fsc_curve.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    shells = np.array([int(row["shell"]) for row in rows], dtype=float)
    fsc = np.array([float(row["gt_reconstruction"]) for row in rows])
    return Reconstruction(metrics["gt_reconstruction"], shells / (metrics["box"] * metrics["voxel_size_A"]), fsc)


def reconstruction_rows(reconstructions: Reconstructions) -> list[dict[str, Any]]:
    """One row of ``RECONSTRUCTION_COLUMNS`` per map: each variant's poses with and without origins, then the truth."""
    labelled = [
        *(("estimated", variant, r) for variant, r in reconstructions.estimated.items()),
        *(("estimated_angles", variant, r) for variant, r in reconstructions.angles.items()),
        ("true", "", reconstructions.true),
    ]
    return [
        {
            "poses": poses,
            "variant": variant,
            "gt_fsc_auc": r.summary["auc"],
            "gt_resolution_0.5_A": r.summary["at_0.5"]["first_A"],
            "gt_resolution_0.143_A": r.summary["at_0.143"]["first_A"],
        }
        for poses, variant, r in labelled
    ]


def read_pose_errors(evaluation_dir: Path) -> np.ndarray:
    """Every particle's final direction error (degrees) of the run after its poses are turned back by their average
    offset, both half sets, from ``pose_errors.npz``."""
    with np.load(evaluation_dir / "pose_errors.npz", allow_pickle=False) as archive:
        return np.concatenate([archive[f"aligned_half{half}"] for half in HALVES]).astype(np.float64)


def read_inplane_errors(evaluation_dir: Path) -> np.ndarray:
    """Every particle's final in-plane error (degrees) of the run after its poses are turned back by their average
    offset, both half sets, from ``pose_errors.npz``."""
    with np.load(evaluation_dir / "pose_errors.npz", allow_pickle=False) as archive:
        if "aligned_inplane_half1" not in archive.files:
            raise ValueError(
                f"{evaluation_dir}: pose_errors.npz has no in-plane errors; rerun evaluate_run.py on the run"
            )
        return np.concatenate([archive[f"aligned_inplane_half{half}"] for half in HALVES]).astype(np.float64)


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


def write_table(rows: list[dict[str, Any]], path: Path, columns: tuple[str, ...] = COLUMNS) -> None:
    """Write ``rows`` as a CSV with ``columns`` (a value that never happened is empty)."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
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


def draw_reconstructions(
    curve_ax,
    bar_ax,
    rows: list[dict[str, Any]],
    colors: dict[str, str],
    labels: list[str],
    reconstructions: Reconstructions,
) -> None:
    """The relion_reconstruct maps' FSC to the ground truth, in the frame of the poses (not aligned): the curves of
    the variants' final poses (dotted: the true poses) and, per variant, the FSC area of the maps from the poses with
    and without origins, against the area of the refinement's own final map as refined (black line) and of the true
    poses (dotted line)."""
    variants = [row["variant"] for row in rows]
    x = np.arange(len(variants))
    style_axes(curve_ax)
    for variant in variants:
        curve = reconstructions.estimated[variant]
        curve_ax.plot(curve.frequency, curve.fsc, color=colors[variant], linewidth=2)
    true = reconstructions.true
    curve_ax.plot(true.frequency, true.fsc, color=PALETTE["ink2"], linewidth=1.5, linestyle=":")
    for threshold in GT_FSC_THRESHOLDS:
        curve_ax.axhline(threshold, color=PALETTE["reference"], linewidth=1, linestyle=":")
    curve_ax.set_title("relion_reconstruct maps: FSC to the ground truth", color=PALETTE["ink"])
    curve_ax.set_xlabel("spatial frequency (1/A)", color=PALETTE["ink2"])
    curve_ax.set_ylabel("FSC", color=PALETTE["ink2"])

    style_axes(bar_ax)
    width = 0.36
    for offset, source, alpha in (
        (-width / 2, reconstructions.estimated, 1.0),
        (width / 2, reconstructions.angles, 0.45),
    ):
        for position, variant in zip(x, variants):
            auc = source[variant].summary["auc"]
            bar_ax.bar(
                position + offset,
                auc,
                width=width,
                color=colors[variant],
                alpha=alpha,
                edgecolor=colors[variant],
                linewidth=1.2,
            )
            bar_ax.text(
                position + offset,
                0.01,
                f"{auc:.3g}",
                rotation=90,
                ha="center",
                va="bottom",
                color=PALETTE["ink"],
                fontsize=8,
            )
    for position, row in zip(x, rows):
        bar_ax.plot(
            [position - 0.4, position + 0.4], [row["gt_fsc_auc_as_refined"]] * 2, color=PALETTE["ink"], linewidth=2
        )
    bar_ax.axhline(true.summary["auc"], color=PALETTE["ink2"], linewidth=1.5, linestyle=":")
    bar_ax.text(
        bar_ax.get_xlim()[1],
        true.summary["auc"],
        f"true poses {true.summary['auc']:.3g}",
        ha="right",
        va="bottom",
        color=PALETTE["ink2"],
        fontsize=8,
    )
    bar_ax.set_xticks(x, labels)
    bar_ax.set_title("Area under that FSC curve", color=PALETTE["ink"])
    bar_ax.set_ylabel("normalized area", color=PALETTE["ink2"])
    bar_ax.margins(y=0.15)


def plot_comparison(
    rows: list[dict[str, Any]],
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    conical_curves: dict[str, ConicalCurve],
    iteration_curves: dict[str, IterationCurve],
    pose_errors: dict[str, np.ndarray],
    inplane_errors: dict[str, np.ndarray],
    reconstructions: Reconstructions | None,
    label: str,
    path: Path,
) -> None:
    """The comparison figure of ``rows`` (one per variant), their ground-truth FSC ``curves``, the mean and SD of
    their cFSC over the cones (``conical_curves``), their per-iteration half-map FSC areas and Fourier radii (``iteration_curves``), their particles' direction errors
    (``pose_errors``) and in-plane errors (``inplane_errors``), degrees, and, when given, the relion_reconstruct maps from their poses (``reconstructions``)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if len(rows) > len(VARIANT_COLORS):
        raise ValueError(f"{len(rows)} variants; the palette is validated for {len(VARIANT_COLORS)}")
    variants = [row["variant"] for row in rows]
    colors = dict(zip(variants, VARIANT_COLORS))
    n_rows = 3 + (reconstructions is not None)
    fig = plt.figure(figsize=(11, 4.07 * (n_rows + 1.45)), facecolor=PALETTE["surface"])
    grid = fig.add_gridspec(n_rows + 2, 6, height_ratios=[1] * n_rows + [0.85, 0.6])
    spans = {2: ((0, 2), (4, 6))}  # the third row also holds the in-plane error between its two panels
    axes = np.array(
        [[fig.add_subplot(grid[row, a:b]) for a, b in spans.get(row, ((0, 3), (3, 6)))] for row in range(n_rows)]
    )
    inplane_ax = fig.add_subplot(grid[2, 2:4])
    trace_ax = fig.add_subplot(grid[n_rows, :])
    radius_ax = fig.add_subplot(grid[n_rows + 1, :], sharex=trace_ax)
    for ax in (*axes.flat, inplane_ax, trace_ax, radius_ax):
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
    ax.set_title("FSC of the final map, aligned to the ground truth", color=PALETTE["ink"])
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

    bars(axes[0, 1], "gt_fsc_auc_aligned", "Area under the aligned ground-truth FSC curve", "normalized area")

    ax = axes[1, 0]
    style_axes(ax)
    for variant in variants:
        cone = conical_curves[variant]
        ax.fill_between(cone.frequency, cone.mean - cone.std, cone.mean + cone.std, color=colors[variant], alpha=0.18)
        ax.plot(cone.frequency, cone.mean, color=colors[variant], linewidth=2)
        for extreme in (cone.best, cone.worst):
            ax.plot(cone.frequency, extreme, color=colors[variant], linewidth=1.2, linestyle="--")
    for threshold in GT_FSC_THRESHOLDS:
        ax.axhline(threshold, color=PALETTE["reference"], linewidth=1, linestyle=":")
    ax.set_title(
        "cFSC to the ground truth, aligned: mean \N{PLUS-MINUS SIGN}1 SD over cones\n(dashed: the best and the worst cone)",
        color=PALETTE["ink"],
    )
    ax.set_xlabel("spatial frequency (1/A)", color=PALETTE["ink2"])
    ax.set_ylabel("FSC", color=PALETTE["ink2"])

    ax = axes[1, 1]
    style_axes(ax)
    gt_cfar = np.array([row["gt_cfar_aligned"] for row in rows])
    half_cfar = np.array([row["half_map_cfar"] for row in rows])
    ax.bar(x, gt_cfar, color=[colors[v] for v in variants], width=0.6)
    ax.plot(x, half_cfar, "o", markerfacecolor=PALETTE["surface"], markeredgecolor=PALETTE["ink2"], markersize=8)
    for position, value, half, top in zip(x, gt_cfar, half_cfar, np.maximum(gt_cfar, half_cfar)):
        ax.annotate(
            f"{value:.3g}\n({half:.3g})",
            (position, top),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=PALETTE["ink2"],
            fontsize=9,
        )
    ax.set_xticks(x, labels)
    ax.set_title("cFAR of the aligned map (hollow, in brackets: of the half maps)", color=PALETTE["ink"])
    ax.set_ylabel("cFAR", color=PALETTE["ink2"])
    ax.set_ylim(0, 1.45 * max(gt_cfar.max(), half_cfar.max()))  # cFAR is 1 only for an isotropic map

    def error_violins(ax, errors_by_variant: dict[str, np.ndarray], title: str) -> None:
        """The error distribution of every variant: a violin on a log axis, the median (filled dot) and the 90th
        percentile (hollow), and the share of particles above ``GROSS_POSE_ERROR_DEG``."""
        style_axes(ax)
        densities = {variant: error_density(errors_by_variant[variant]) for variant in variants}
        half_width = 0.42 / max(density.max() for _, density in densities.values())
        for position, variant in zip(x, variants):
            grid, density = densities[variant]
            errors = errors_by_variant[variant]
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
            ax.plot(
                position, p90, "o", markerfacecolor=PALETTE["surface"], markeredgecolor=colors[variant], markersize=8
            )
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
        ax.set_title(title, color=PALETTE["ink"], pad=18)
        ax.set_ylabel("degrees (log scale)", color=PALETTE["ink2"])

    error_violins(axes[2, 0], pose_errors, "Out-of-plane error (viewing direction)\nof the final poses, offset removed")
    error_violins(
        inplane_ax, inplane_errors, "In-plane error (about the viewing direction)\nof the final poses, offset removed"
    )

    bars(
        axes[2, 1],
        "learned_prior_js_bits",
        "Learned prior (viewing directions):\ndivergence from the true prior",
        "JS (bits); fixed runs: estimated, not used",
        title_pad=18,
    )

    if reconstructions is not None:
        draw_reconstructions(axes[3, 0], axes[3, 1], rows, colors, labels, reconstructions)

    style_axes(trace_ax)
    style_axes(radius_ax)
    for variant in variants:
        curve = iteration_curves[variant]
        raised = np.flatnonzero(np.diff(curve.order, prepend=curve.order[0]) > 0)
        trace_ax.plot(curve.iteration, curve.half_area, color=colors[variant], linewidth=2, linestyle="--")
        trace_ax.plot(curve.iteration, curve.gt_area, color=colors[variant], linewidth=2)
        trace_ax.plot(curve.iteration[raised], curve.gt_area[raised], "o", color=colors[variant], markersize=6)
        trace_ax.text(
            curve.iteration[-1] + 0.15,
            curve.gt_area[-1],
            f"order {curve.order[-1]}",
            va="center",
            color=PALETTE["ink2"],
            fontsize=8,
        )
        radius_ax.plot(curve.iteration, curve.radius, color=colors[variant], linewidth=2)
        radius_ax.plot(curve.iteration[raised], curve.radius[raised], "o", color=colors[variant], markersize=6)
    trace_ax.set_title("Half maps' FSC area per iteration", color=PALETTE["ink"])
    trace_ax.set_ylabel("normalized area", color=PALETTE["ink2"])
    trace_ax.tick_params(labelbottom=False)
    radius_ax.set_title("Fourier radius of the refinement per iteration", color=PALETTE["ink"])
    radius_ax.set_xlabel("iteration", color=PALETTE["ink2"])
    radius_ax.set_ylabel("pixels", color=PALETTE["ink2"])
    radius_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    radius_ax.margins(x=0.08)

    legend = [Line2D([], [], color=colors[v], linewidth=2, label=v) for v in variants] + [
        Line2D([], [], color=PALETTE["ink2"], linewidth=2, label="half maps vs ground truth, as refined"),
        Line2D([], [], color=PALETTE["ink2"], linewidth=2, linestyle="--", label="half map vs other half map"),
        Line2D([], [], color=PALETTE["ink2"], linewidth=1.2, linestyle="--", label="best and worst cone (cFSC panel)"),
        Line2D([], [], marker="o", color=PALETTE["ink2"], linestyle="", label="HEALPix order raised"),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor=PALETTE["surface"],
            markeredgecolor=PALETTE["ink2"],
            linestyle="",
            label="half-map cFAR",
        ),
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
    if reconstructions is not None:
        legend += [
            Patch(facecolor=PALETTE["ink2"], label="reconstruction from the final poses"),
            Patch(facecolor=PALETTE["ink2"], alpha=0.45, label="... with zero origins"),
            Line2D([], [], color=PALETTE["ink"], linewidth=2, label="the refinement's final map, as refined"),
            Line2D([], [], color=PALETTE["ink2"], linewidth=1.5, linestyle=":", label="true poses"),
        ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, labelcolor=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0.07 if reconstructions is None else 0.09, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="LABEL=EVALUATION_DIR of each variant")
    parser.add_argument("--reconstructions", nargs="+", help="LABEL=DIR of each variant's relion_reconstruct map")
    parser.add_argument(
        "--angle-reconstructions", nargs="+", help="LABEL=DIR of each variant's map from the angles with zero origins"
    )
    parser.add_argument("--true-reconstruction", type=Path, help="DIR of the map from the true poses")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label", default="scenario")
    args = parser.parse_args(argv)
    given = [args.reconstructions, args.angle_reconstructions, args.true_reconstruction]
    if any(item is not None for item in given) and any(item is None for item in given):
        parser.error("--reconstructions, --angle-reconstructions and --true-reconstruction go together")
    return args


def read_reconstructions(args: argparse.Namespace, variants: list[str]) -> Reconstructions | None:
    """The reconstructions named on the command line, None when there are none; each must cover ``variants``."""
    if args.reconstructions is None:
        return None
    sets = []
    for items in (args.reconstructions, args.angle_reconstructions):
        directories = parse_runs(items)
        if list(directories) != variants:
            raise SystemExit(f"reconstructions for {list(directories)}, the runs are {variants}")
        sets.append({variant: read_reconstruction(directory) for variant, directory in directories.items()})
    return Reconstructions(sets[0], sets[1], read_reconstruction(args.true_reconstruction))


def main(argv: list[str] | None = None) -> None:
    """Write ``variants.csv``, ``variants.png`` and, with reconstructions, ``reconstructions.csv``."""
    args = parse_args(argv)
    runs = parse_runs(args.runs)
    metrics = {variant: json.loads((directory / "metrics.json").read_text()) for variant, directory in runs.items()}
    rows = [summary_row(variant, metrics[variant]) for variant in runs]
    curves = {variant: read_gt_curve(directory, metrics[variant]) for variant, directory in runs.items()}
    conical_curves = {variant: read_conical_curve(directory, metrics[variant]) for variant, directory in runs.items()}
    iteration_curves = {variant: read_iteration_curve(directory) for variant, directory in runs.items()}
    pose_errors = {variant: read_pose_errors(directory) for variant, directory in runs.items()}
    inplane_errors = {variant: read_inplane_errors(directory) for variant, directory in runs.items()}
    reconstructions = read_reconstructions(args, list(runs))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_table(rows, args.out_dir / "variants.csv")
    if reconstructions is not None:
        write_table(reconstruction_rows(reconstructions), args.out_dir / "reconstructions.csv", RECONSTRUCTION_COLUMNS)
    plot_comparison(
        rows,
        curves,
        conical_curves,
        iteration_curves,
        pose_errors,
        inplane_errors,
        reconstructions,
        args.label,
        args.out_dir / "variants.png",
    )
    for row in rows:
        print(
            f"{row['variant']}: {row['direction_prior']} prior, {row['iterations']} iterations, "
            f"aligned GT-FSC area {row['gt_fsc_auc_aligned']:.3f}, aligned cFAR {row['gt_cfar_aligned']:.3f}, "
            f"aligned median out-of-plane error {row['pose_error_median_aligned_deg']:.2f} deg, "
            f"in-plane {row['inplane_error_median_aligned_deg']:.2f} deg"
        )


if __name__ == "__main__":
    main()
