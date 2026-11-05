"""Figures of a conical FSC (``scripts/conical_fsc.py``).

Per pair of maps, two panels:

* the cFSC curves against spatial frequency: the mean over the cones with its +-1 SD band and its minimum and
  maximum, the global FSC for reference (from shell 1), and under it the number of cones that first cross the threshold at each
  shell (cones that never cross are counted in the strip's label);
* the wAuC of every cone as a fraction of the largest, over the cone axis' ``rot`` and ``tilt``: a cell takes the
  value of its nearest axis. Its smallest value is the cFAR.

Used by ``evaluate_run.py`` (``plot_conical_fsc``) and ``compare_variants.py`` (``curve_statistics``). The palette is
passed in by the caller, so this module imports nothing from either.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from scripts.conical_fsc import ConicalFsc, cone_areas, measured_curves

# The single-hue sequential blue ramp, steps 100 to 700: light is little signal, dark is the most.
SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
HEATMAP_STEP_DEG = 1.0
MARKER_COLOR_KEY = "half2"  # the palette entry of the lowest-wAuC axis marker


def curve_statistics(cfsc: ConicalFsc, min_voxels: int, first_shell: int) -> dict[str, np.ndarray]:
    """Per shell from ``first_shell`` on, over every cone: ``mean``, ``std`` (population), ``min`` and ``max`` of the
    cFSC; NaN in the shells below it, where some cone is not measured (``min_voxels``) and a statistic would
    average a changing set of cones.
    """
    curves = measured_curves(cfsc, min_voxels)
    curves[:, :first_shell] = np.nan
    measured = np.ma.masked_invalid(curves)
    return {
        "mean": measured.mean(axis=0).filled(np.nan),
        "std": measured.std(axis=0).filled(np.nan),
        "min": measured.min(axis=0).filled(np.nan),
        "max": measured.max(axis=0).filled(np.nan),
    }


def relative_wauc_map(cfsc: ConicalFsc, wauc: np.ndarray, step_deg: float = HEATMAP_STEP_DEG):
    """The wAuC over ``max(wAuC)`` on a regular (rot, tilt) grid, each cell taking its nearest axis' value.

    Returns ``(rot_edges, tilt_edges, values)`` for ``pcolormesh``; ``values`` has shape ``(n_tilt, n_rot)``. Axes
    are compared by ``|cos|`` of their angle: the cone of an axis and of its opposite are one.
    """
    rot_edges = np.arange(-180.0, 180.0 + step_deg / 2, step_deg)
    tilt_edges = np.arange(0.0, 90.0 + step_deg / 2, step_deg)
    rot = np.radians((rot_edges[:-1] + rot_edges[1:]) / 2)[None, :]
    tilt = np.radians((tilt_edges[:-1] + tilt_edges[1:]) / 2)[:, None]
    cells = np.stack(
        [np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt) * np.ones_like(rot)], axis=-1
    )
    nearest = np.argmax(np.abs(cells @ cfsc.axes.T), axis=-1)
    return rot_edges, tilt_edges, (wauc / np.max(wauc))[nearest]


def draw_curves(
    ax,
    strip,
    cfsc: ConicalFsc,
    global_fsc: np.ndarray,
    summary: Mapping[str, Any],
    voxel: float,
    box: int,
    c,
    thresholds,
) -> None:
    """The cFSC mean +- SD, min and max on ``ax`` and the crossing histogram on ``strip`` (shared x axis)."""
    from matplotlib.ticker import MaxNLocator

    min_voxels = summary["min_cone_voxels"]
    first = summary["first_shell_measured_by_all_cones"]
    stats = curve_statistics(cfsc, min_voxels, first)
    n_shells = cfsc.curves.shape[1]
    frequency = np.arange(n_shells) / (box * voxel)
    blue = SEQUENTIAL_BLUE[3]
    ax.fill_between(
        frequency, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=blue, alpha=0.25, linewidth=0
    )
    ax.plot(frequency, stats["min"], color=SEQUENTIAL_BLUE[2], linewidth=1)
    ax.plot(frequency, stats["max"], color=SEQUENTIAL_BLUE[2], linewidth=1)
    ax.plot(frequency, stats["mean"], color=blue, linewidth=2)
    ax.plot(frequency[1 : len(global_fsc)], global_fsc[1:], color=c["ink2"], linewidth=1.5, linestyle="--")
    for level in thresholds:
        ax.axhline(level, color=c["reference"], linewidth=1, linestyle=":")
        ax.text(frequency[-1], level + 0.015, f"{level}", color=c["ink2"], ha="right", va="bottom", fontsize=8)
    ax.set(ylim=(-0.25, 1.02), ylabel="FSC")
    ax.grid(color=c["grid"], linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelbottom=False)

    _, crossing = cone_areas(cfsc, summary["threshold"], min_voxels)
    crossed = crossing[crossing < n_shells]
    counts = np.bincount(crossed, minlength=n_shells)
    strip.bar(frequency, counts, width=0.8 / (box * voxel), color=c["half2"], linewidth=0)
    strip.set(xlabel="spatial frequency (1/A)", ylabel="cones", ylim=(0, max(1, counts.max())))
    strip.set_title(
        f"first {summary['threshold']} crossing per cone; {summary['n_axes_without_crossing']} of "
        f"{summary['n_axes']} never cross",
        fontsize=8,
        color=c["ink2"],
        pad=3,
    )
    strip.grid(color=c["grid"], linewidth=0.6, axis="y")
    strip.spines[["top", "right"]].set_visible(False)
    strip.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))


def draw_heatmap(ax, cfsc: ConicalFsc, summary: Mapping[str, Any], c):
    """The relative wAuC over the cone axis' rot and tilt on ``ax``; returns the mesh for a colorbar."""
    from matplotlib.colors import LinearSegmentedColormap

    wauc, _ = cone_areas(cfsc, summary["threshold"], summary["min_cone_voxels"])
    rot_edges, tilt_edges, values = relative_wauc_map(cfsc, wauc)
    mesh = ax.pcolormesh(
        rot_edges,
        tilt_edges,
        values,
        cmap=LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE),
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        rasterized=True,
    )
    lowest = summary["min_axis"]
    ax.plot(
        lowest["rot_deg"],
        lowest["tilt_deg"],
        marker="X",
        markersize=10,
        color=c[MARKER_COLOR_KEY],
        markeredgecolor=c["surface"],
        markeredgewidth=1.5,
        linestyle="",
    )
    ax.set(xlim=(-180, 180), ylim=(0, 90), xlabel="cone axis rot (deg)", ylabel="cone axis tilt from z (deg)")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(0, 91, 30))
    return mesh


def plot_conical_fsc(
    pairs: Mapping[str, tuple[ConicalFsc, np.ndarray, Mapping[str, Any], tuple[float, ...]]],
    voxel: float,
    box: int,
    label: str,
    palette: Mapping[str, str],
    path,
) -> None:
    """One row per pair of maps, ``pairs[title] = (cfsc, global_fsc, summary, thresholds)``: the cFSC curves with
    their SD band and crossing histogram, and the wAuC heatmap with its cFAR."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c = palette
    summaries = [summary for _, _, summary, _ in pairs.values()]
    half_angle, min_voxels = summaries[0]["half_angle_deg"], summaries[0]["min_cone_voxels"]
    plt.rcParams.update(
        {
            "figure.facecolor": c["surface"],
            "axes.facecolor": c["surface"],
            "savefig.facecolor": c["surface"],
            "text.color": c["ink"],
            "axes.labelcolor": c["ink2"],
            "xtick.color": c["ink2"],
            "ytick.color": c["ink2"],
            "axes.edgecolor": c["grid"],
            "font.size": 9,
        }
    )
    fig = plt.figure(figsize=(13, 5.0 * len(pairs) + 0.8))
    outer = fig.add_gridspec(len(pairs), 2, width_ratios=[1.0, 1.05], hspace=0.38, wspace=0.18)
    for row, (title, (cfsc, global_fsc, summary, thresholds)) in enumerate(pairs.items()):
        inner = outer[row, 0].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.55)
        ax = fig.add_subplot(inner[0])
        strip = fig.add_subplot(inner[1], sharex=ax)
        draw_curves(ax, strip, cfsc, global_fsc, summary, voxel, box, c, thresholds)
        ax.set_title(f"{title}: cFSC over {summary['n_axes']} cones", loc="left", pad=8)
        heat = fig.add_subplot(outer[row, 1])
        mesh = draw_heatmap(heat, cfsc, summary, c)
        heat.set_title(f"cFAR = {summary['cfar']:.3f}", loc="left", pad=8)
        fig.colorbar(mesh, ax=heat, pad=0.02, fraction=0.04).set_label("wAuC / largest wAuC (smallest = cFAR)")
    handles = [
        Line2D([], [], color=SEQUENTIAL_BLUE[3], linewidth=2, label="mean over cones, band: +-1 SD"),
        Line2D([], [], color=SEQUENTIAL_BLUE[2], linewidth=1, label="smallest and largest over cones"),
        Line2D([], [], color=c["ink2"], linewidth=1.5, linestyle="--", label="global FSC"),
        Line2D(
            [], [], linestyle="", label=f"cone curves start where every cone holds >= {min_voxels} voxels per shell"
        ),
        Line2D(
            [],
            [],
            color=c[MARKER_COLOR_KEY],
            marker="X",
            markeredgecolor=c["surface"],
            linestyle="",
            label="smallest wAuC",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0), labelcolor=c["ink2"]
    )
    fig.suptitle(f"{label}: conical FSC, half-angle {half_angle:g} deg", x=0.02, ha="left", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
