#!/usr/bin/env python
"""Summarise the pose-prior sweep across its datasets: one table and one figure per metric.

Reads ``<scenario>/comparison/variants.csv`` (``compare_variants.py``) of every scenario of the sweep that
``config.yaml``'s ``sweep`` section (plus the override files the run used) describes, and writes to ``--out-dir``:

* ``summary.csv``: one row per scenario and variant with the sweep coordinates (``kent_weight``, ``n_images``,
  ``noise_level``; the offsets scenario has ``offsets_sigma_A``) beside the comparison's columns and
  ``gt_fsc_auc_fraction_of_true_poses``, the aligned FSC area over the area of the map from the true poses;
* ``summary_<metric>.png``: a panel per noise level (columns) and particle count (rows), the metric against the
  Kent share with one line per variant;
* ``summary_offsets.png``: the offsets scenario against its centred counterpart (the mixture of Kent share 1 with the
  same noise level and particle count), per variant.

A scenario without a ``variants.csv`` is a gap in the figures; the script stops unless ``--allow-missing`` is given,
and lists the gaps either way.

Run::

    python scripts/pose_prior_benchmark/summarize_sweep.py --output-root ROOT --out-dir ROOT/sweep_summary \\
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
# the offsets figure: (column, title, y label)
OFFSET_METRICS = (
    ("gt_fsc_auc_aligned", "Aligned FSC area", "normalized area"),
    ("pose_error_fraction_over_90_aligned", "Out-of-plane error over 90 degrees", "share"),
    ("pose_error_median_aligned_deg", "Median out-of-plane error", "degrees"),
    ("origin_error_median_A", "Median origin error", "A"),
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
    """Every scenario of the sweep with its coordinates: the mixtures, then the offsets scenario."""
    cells = [
        {"scenario": sweep_lib.mixture_name(share, n, noise), "kent_weight": share, "n_images": n, "noise_level": noise}
        for noise in sweep["noise_levels"]
        for n in sweep["n_images"]
        for share in sweep["kent_weights"]
    ]
    offsets = sweep.get("offsets")
    if offsets:
        cells.append(
            {
                "scenario": sweep_lib.offsets_name(offsets["noise_level"]),
                "kent_weight": 1.0,
                "n_images": max(sweep["n_images"]),
                "noise_level": offsets["noise_level"],
                "offsets_sigma_A": offsets["sigma_angstrom"],
            }
        )
    return cells


def collect_rows(output_root: Path, sweep: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The rows of every scenario's ``variants.csv`` with the sweep coordinates added, and the scenarios without one."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for cell in sweep_cells(sweep):
        path = output_root / cell["scenario"] / "comparison" / "variants.csv"
        if not path.is_file():
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
            ceilings = {row["gt_fsc_auc_true_poses"] for row in cell if row.get("gt_fsc_auc_true_poses") is not None}
            if column == "gt_fsc_auc_aligned" and ceilings:
                for ceiling in ceilings:
                    ax.axhline(ceiling, color=PALETTE["ink2"], linewidth=1.2, linestyle=":")
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
        legend.append(Line2D([], [], color=PALETTE["ink2"], linewidth=1.2, linestyle=":", label="true poses"))
    fig.suptitle(title, x=0.01, ha="left", color=PALETTE["ink"])
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), frameon=False, labelcolor=PALETTE["ink2"])
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)


def plot_offsets(rows: list[dict[str, Any]], path: Path) -> bool:
    """The offsets scenario against its centred counterpart, per variant, one bar panel per metric; False when the
    sweep has no offsets scenario or its counterpart ran."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shifted = [row for row in rows if "offsets_sigma_A" in row]
    if not shifted:
        return False
    first = shifted[0]
    centred = [
        row for row in rows
        if "offsets_sigma_A" not in row and row["kent_weight"] == 1.0 and row["n_images"] == first["n_images"]
        and row["noise_level"] == first["noise_level"]
    ]
    if not centred:
        return False
    variants = [v for v in VARIANT_ORDER if any(row["variant"] == v for row in shifted)]
    colors = dict(zip(VARIANT_ORDER, VARIANT_COLORS))
    fig, axes = plt.subplots(1, len(OFFSET_METRICS), figsize=(3.4 * len(OFFSET_METRICS), 3.8), facecolor=PALETTE["surface"])
    width = 0.38
    for ax, (column, title, ylabel) in zip(axes, OFFSET_METRICS):
        _style(ax)
        for k, (group, alpha, label) in enumerate(((centred, 1.0, "centred"), (shifted, 0.45, "with offsets"))):
            for i, variant in enumerate(variants):
                value = next((row[column] for row in group if row["variant"] == variant), None)
                if value is None:
                    continue
                ax.bar(i + (k - 0.5) * width, value, width=width, color=colors[variant], alpha=alpha,
                       edgecolor=colors[variant], linewidth=1.2)
                ax.text(i + (k - 0.5) * width, value, f"{value:.3g}", ha="center", va="bottom", color=PALETTE["ink2"], fontsize=8)
        ax.set_xticks(range(len(variants)), [v.replace("fixed_", "fixed\n") for v in variants])
        ax.set_title(title, color=PALETTE["ink"], fontsize=10)
        ax.set_ylabel(ylabel, color=PALETTE["ink2"])
        ax.margins(y=0.15)
    from matplotlib.patches import Patch

    fig.suptitle(
        f"Offsets (sigma {first['offsets_sigma_A']:g} A per axis) against centred particles: {first['n_images'] // 1000}k "
        f"particles, noise {first['noise_level']:g}", x=0.01, ha="left", color=PALETTE["ink"],
    )
    fig.legend(
        handles=[Patch(facecolor=PALETTE["ink2"], label="centred"), Patch(facecolor=PALETTE["ink2"], alpha=0.45, label="with offsets")],
        loc="lower center", ncol=2, frameon=False, labelcolor=PALETTE["ink2"],
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["surface"])
    plt.close(fig)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True, help="the run's output_root")
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
