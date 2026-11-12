#!/usr/bin/env python
"""Summarise the pose-prior sweep across its datasets: one table and one figure per metric.

Reads ``<scenario>/comparison/variants.csv`` (``compare_variants.py``) of every scenario of the sweep that
``config.yaml``'s ``sweep`` section (plus the override files the run used) describes, from the first of the output
roots that has it (a run that added cells later has its own root), and writes to ``--out-dir``:

* ``summary.csv``: one row per scenario and variant with the sweep coordinates (``kent_weight``, ``n_images``,
  ``noise_level``; a cell with offsets has ``offsets_sigma_A``) beside the comparison's columns and
  ``gt_fsc_auc_fraction_of_true_poses``, the aligned FSC area over the area of the map from the true poses;
* ``summary_<metric>.png``: a panel per noise level (columns) and particle count (rows), the metric against the
  Kent share with one line per variant;
* ``summary_offsets.png``: the cells with offsets against their centred counterparts (the mixture of the same Kent
  share, noise level and particle count), a column per noise level and particle count and a row per metric, the metric
  against the Kent share with one line per variant, solid centred and dashed with offsets.

A scenario without a ``variants.csv`` is a gap in the figures; the script stops unless ``--allow-missing`` is given,
and lists the gaps either way.

Run::

    python scripts/pose_prior_benchmark/summarize_sweep.py --output-root ROOT [ROOT ...] --out-dir ROOT/sweep_summary \\
        [--config OVERRIDE.yaml ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.pose_prior_benchmark import sweep as sweep_lib  # noqa: E402
from scripts.pose_prior_benchmark.compare_variants import VARIANT_COLORS  # noqa: E402
from scripts.pose_prior_benchmark.evaluate_run import PALETTE  # noqa: E402

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
VARIANT_ORDER = ("learned", "fixed_true", "fixed_uniform")
MARKERS = {"learned": "o", "fixed_true": "s", "fixed_uniform": "^"}
# (column, title, y label, log axis): the metrics drawn against the Kent share
METRICS = (
    ("gt_fsc_auc_aligned", "Area under the aligned ground-truth FSC curve", "normalized area", False),
    ("gt_fsc_auc_fraction_of_true_poses", "Aligned FSC area over that of the true poses", "fraction", False),
    ("gt_resolution_0.143_A_aligned", "Resolution of the aligned map (FSC 0.143)", "A", False),
    ("pose_error_median_aligned_deg", "Median out-of-plane error", "degrees (log scale)", True),
    ("pose_error_fraction_over_90_aligned", "Share of particles with an out-of-plane error over 90 degrees", "share", False),
    ("inplane_error_median_aligned_deg", "Median in-plane error", "degrees (log scale)", True),
    ("gt_cfar_aligned", "cFAR of the aligned map", "cFAR", False),
    ("final_healpix_order", "Final HEALPix order", "order", False),
    ("iterations", "Iterations", "iterations", False),
)
# the offsets figure: (column, title, y label, log axis)
OFFSET_METRICS = (
    ("gt_fsc_auc_aligned", "Aligned FSC area", "normalized area", False),
    ("pose_error_fraction_over_90_aligned", "Out-of-plane error over 90 degrees", "share", False),
    ("pose_error_median_aligned_deg", "Median out-of-plane error", "degrees (log scale)", True),
    ("origin_error_median_A", "Median origin error", "A (log scale)", True),
)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """``base`` with ``update`` merged in: nested dicts merge, every other value is replaced."""
    merged = dict(base)
    for key, value in update.items():
        merged[key] = deep_merge(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def load_sweep(config_files: list[Path]) -> dict[str, Any]:
    """The ``sweep`` section of ``config.yaml`` with the override files merged over it, in order."""
    config = yaml.safe_load(DEFAULT_CONFIG.read_text())
    for path in config_files:
        config = deep_merge(config, yaml.safe_load(path.read_text()) or {})
    return config["sweep"]


def read_variants(path: Path) -> list[dict[str, Any]]:
    """The rows of a ``variants.csv`` with numbers as floats and empty cells as None."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{key: _number(value) for key, value in row.items()} for row in rows]


def _number(value: str) -> Any:
    """A CSV cell as a float, None when empty, the string itself when it is not a number."""
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


def sweep_cells(sweep: dict[str, Any]) -> list[dict[str, Any]]:
    """Every scenario of the sweep with its coordinates: the mixtures, then the mixtures with offsets."""
    cells = [
        {"scenario": sweep_lib.mixture_name(share, n, noise), "kent_weight": share, "n_images": n, "noise_level": noise}
        for noise in sweep["noise_levels"]
        for n in sweep["n_images"]
        for share in sweep["kent_weights"]
    ]
    for share, n, noise in sweep_lib.offsets_cells(sweep):
        cells.append(
            {
                "scenario": sweep_lib.offsets_mixture_name(share, n, noise),
                "kent_weight": share,
                "n_images": n,
                "noise_level": noise,
                "offsets_sigma_A": sweep["offsets"]["sigma_angstrom"],
            }
        )
    return cells


def collect_rows(output_roots: list[Path], sweep: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The rows of every scenario's ``variants.csv`` (the first output root that has it) with the sweep coordinates
    added, and the scenarios without one."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in sweep_cells(sweep):
        paths = [root / cell["scenario"] / "comparison" / "variants.csv" for root in output_roots]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            missing.append(cell["scenario"])
            continue
        for row in read_variants(path):
            ceiling = row.get("gt_fsc_auc_true_poses")
            fraction = None if not ceiling or row["gt_fsc_auc_aligned"] is None else row["gt_fsc_auc_aligned"] / ceiling
            rows.append({**cell, **row, "gt_fsc_auc_fraction_of_true_poses": fraction})
    return rows, missing


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    """Write ``rows`` as a CSV (every key of any row is a column; a value that never happened is empty)."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row[key] for key in columns})


def _style(ax) -> None:
    ax.grid(color=PALETTE["grid"], linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_facecolor(PALETTE["surface"])
    ax.tick_params(colors=PALETTE["ink2"])


def plot_metric(
    rows: list[dict[str, Any]], sweep: dict[str, Any], column: str, title: str, ylabel: str, log: bool, path: Path
) -> None:
    """A panel per noise level (columns) and particle count (rows): ``column`` against the Kent share, a line per
    variant on one y axis shared by the panels; the true-pose FSC area is a dotted line in the area panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mixtures = [row for row in rows if "offsets_sigma_A" not in row]
    noises, counts = sweep["noise_levels"], sorted(sweep["n_images"], reverse=True)
    values = [row[column] for row in mixtures if row[column] is not None]
    if column == "gt_fsc_auc_aligned":  # the true-pose ceilings are drawn in these panels too
        values += [row["gt_fsc_auc_true_poses"] for row in mixtures if row.get("gt_fsc_auc_true_poses") is not None]
    fig, axes = plt.subplots(
        len(counts), len(noises), figsize=(3.6 * len(noises) + 0.8, 2.9 * len(counts) + 1.2), squeeze=False,
        sharex=True, sharey=True, facecolor=PALETTE["surface"],
    )
    colors = dict(zip(VARIANT_ORDER, VARIANT_COLORS))
    for r, n in enumerate(counts):
        for c, noise in enumerate(noises):
            ax = axes[r, c]
            _style(ax)
            cell = [row for row in mixtures if row["n_images"] == n and row["noise_level"] == noise]
            for variant in VARIANT_ORDER:
                points = sorted(
                    (row["kent_weight"], row[column]) for row in cell if row["variant"] == variant and row[column] is not None
                )
                if points:
                    x, y = zip(*points)
                    ax.plot(x, y, color=colors[variant], marker=MARKERS[variant], markersize=7, linewidth=2,
                            markeredgecolor=PALETTE["surface"], markeredgewidth=1.5)
            ceilings = sorted(
                {(row["kent_weight"], row["gt_fsc_auc_true_poses"]) for row in cell if row.get("gt_fsc_auc_true_poses") is not None}
            )
            if column == "gt_fsc_auc_aligned" and ceilings:
                ax.plot(*zip(*ceilings), color=PALETTE["ink2"], linewidth=1.5, linestyle=":", marker="_", markersize=9)
            if log:
                ax.set_yscale("log")
            ax.set_title(f"{n // 1000}k particles, noise {noise:g}", color=PALETTE["ink"], fontsize=10)
            ax.set_xticks(sweep["kent_weights"])
            ax.set_xticklabels([f"{100 * w:g}" for w in sweep["kent_weights"]])
            if c == 0:
                ax.set_ylabel(ylabel, color=PALETTE["ink2"])
            if r == len(counts) - 1:
                ax.set_xlabel("Kent share of the particles (%)", color=PALETTE["ink2"])
    if values and not log:
        low, high = min(values), max(values)
        pad = 0.08 * (high - low if high > low else 1.0)
        axes[0, 0].set_ylim(low - pad, high + pad)
    legend = [
        Line2D([], [], color=colors[v], marker=MARKERS[v], markersize=7, linewidth=2, markeredgecolor=PALETTE["surface"],
               label=v)
        for v in VARIANT_ORDER
    ]
    if column == "gt_fsc_auc_aligned":
        legend.append(Line2D([], [], color=PALETTE["ink2"], linewidth=1.5, linestyle=":", label="true poses"))
    fig.suptitle(title, x=0.01, ha="left", color=PALETTE["ink"])
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, labelcolor=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)


def plot_offsets(rows: list[dict[str, Any]], path: Path) -> bool:
    """The cells with offsets against their centred counterparts: a column per (particle count, noise level) and a row
    per metric, the metric against the Kent share with a line per variant (solid centred, dashed and hollow with
    offsets); False when no cell with offsets has a centred counterpart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    shifted = [row for row in rows if "offsets_sigma_A" in row]
    centred = [row for row in rows if "offsets_sigma_A" not in row]
    cells = sorted(
        {(row["n_images"], row["noise_level"]) for row in shifted
         if any(c["n_images"] == row["n_images"] and c["noise_level"] == row["noise_level"] for c in centred)},
        key=lambda cell: (cell[1], -cell[0]),
    )
    if not cells:
        return False
    sigma = shifted[0]["offsets_sigma_A"]
    colors = dict(zip(VARIANT_ORDER, VARIANT_COLORS))
    fig, axes = plt.subplots(
        len(OFFSET_METRICS), len(cells), figsize=(3.3 * len(cells) + 0.8, 2.6 * len(OFFSET_METRICS) + 1.2),
        squeeze=False, sharex=True, sharey="row", facecolor=PALETTE["surface"],
    )
    for c, (n, noise) in enumerate(cells):
        in_cell = [row for row in rows if row["n_images"] == n and row["noise_level"] == noise]
        shares = {row["kent_weight"] for row in in_cell if "offsets_sigma_A" in row}
        for r, (column, title, ylabel, log) in enumerate(OFFSET_METRICS):
            ax = axes[r, c]
            _style(ax)
            for variant in VARIANT_ORDER:
                for group, style, hollow in (("centred", "-", False), ("offsets", "--", True)):
                    points = sorted(
                        (row["kent_weight"], row[column]) for row in in_cell
                        if row["variant"] == variant and row[column] is not None and row["kent_weight"] in shares
                        and (("offsets_sigma_A" in row) == (group == "offsets"))
                    )
                    if points:
                        x, y = zip(*points)
                        ax.plot(x, y, color=colors[variant], linestyle=style, marker=MARKERS[variant], markersize=7,
                                linewidth=2, markeredgecolor=colors[variant] if hollow else PALETTE["surface"],
                                markerfacecolor=PALETTE["surface"] if hollow else colors[variant], markeredgewidth=1.5)
            if log:
                ax.set_yscale("log")
            if r == 0:
                ax.set_title(f"{n // 1000}k particles, noise {noise:g}", color=PALETTE["ink"], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{title}\n{ylabel}", color=PALETTE["ink2"], fontsize=9)
            ax.set_xticks(sorted(shares))
            ax.set_xticklabels([f"{100 * w:g}" for w in sorted(shares)])
            if r == len(OFFSET_METRICS) - 1:
                ax.set_xlabel("Kent share of the particles (%)", color=PALETTE["ink2"])
    legend = [
        Line2D([], [], color=colors[v], marker=MARKERS[v], markersize=7, linewidth=2, markeredgecolor=PALETTE["surface"], label=v)
        for v in VARIANT_ORDER
    ] + [
        Line2D([], [], color=PALETTE["ink2"], linewidth=2, linestyle="-", label="centred"),
        Line2D([], [], color=PALETTE["ink2"], linewidth=2, linestyle="--", label="with offsets"),
    ]
    fig.suptitle(f"Offsets (sigma {sigma:g} A per axis) against centred particles", x=0.01, ha="left", color=PALETTE["ink"])
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, labelcolor=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, nargs="+", required=True, help="the run's output_root(s), searched in order")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, nargs="*", default=[], help="override files merged over config.yaml, in order")
    parser.add_argument("--allow-missing", action="store_true", help="draw the scenarios that finished, list the others")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    sweep = load_sweep(args.config)
    rows, missing = collect_rows(args.output_root, sweep)
    if missing:
        print(f"no variants.csv for {len(missing)} scenario(s): {', '.join(missing)}")
        if not args.allow_missing:
            raise SystemExit("stopping: pass --allow-missing to summarise the scenarios that finished")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary(rows, args.out_dir / "summary.csv")
    for column, title, ylabel, log in METRICS:
        plot_metric(rows, sweep, column, title, ylabel, log, args.out_dir / f"summary_{column}.png")
    if plot_offsets(rows, args.out_dir / "summary_offsets.png"):
        print("wrote summary_offsets.png")
    print(f"summarised {len(rows)} runs of {len({row['scenario'] for row in rows})} scenarios into {args.out_dir}")


if __name__ == "__main__":
    main()
