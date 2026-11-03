#!/usr/bin/env python
"""Tables and the motion plot of ``scripts.run_pose_ordering_comparison``.

    python -m scripts.plot_pose_orderings --dir <orderings dir>

Writes ``orderings_table.csv``, ``orderings_overlap.csv`` and ``figure_motion.png``/``.pdf``: for cases C and A, the
fitted 3D translation (density fit) against the rotation along the motion, per pose source, the mean of each along the
true level, and the rotation against the other body's twist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_pose_ordering_comparison import lines_point

SOURCES = {"fitted": "fitted", "true_body_pose_true_camera": "true body pose, true camera"}


def tables(directory: Path) -> None:
    rows, overlaps = [], []
    for path in sorted(directory.glob("orderings_case*.json")):
        report = json.loads(path.read_text())
        for body, sources in report["bodies"].items():
            for source, entry in sources.items():
                for state, target in entry["targets"].items():
                    for name in target["same_state"]:
                        for kept, one, mean in zip(target["kept"], target["same_state"][name], target["same_state_mean_first_20"][name]):
                            rows.append(f'{report["dataset"]},{body},"{source}",{state},{target["state_size"]},{name},{kept},{one:.3f},{mean:.3f}')
                    overlaps += [f'{report["dataset"]},{body},"{source}",{state},{pair},{value}' for pair, value in target["overlap_at_300"].items()]
    (directory / "orderings_table.csv").write_text("dataset,body,source,state,state_size,ordering,kept,same_state_declared_target,same_state_mean_first_20\n" + "\n".join(rows) + "\n")
    (directory / "orderings_overlap.csv").write_text("dataset,body,source,state,pair,jaccard_at_300\n" + "\n".join(overlaps) + "\n")


CURVE_ORDERINGS = {"bounded": "C0", "rotation only": "C1", "RELION PC1": "C2", "rank 1": "C3", "rank 2": "C4", "density": "C5",
                   "neighbour depth K30": "C6", "perfect": "k"}


def curve_figure(directory: Path, rows: list[tuple[str, list[str]]], stem: str) -> None:
    """Same-state fraction against kept particles, each row a case, averaged over its fixtures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 3 * len(rows)), squeeze=False, sharex=True, sharey=True)
    for row, (label, datasets) in zip(axes, rows):
        reports = [json.loads((directory / f"orderings_{d}.json").read_text())["bodies"]["2"] for d in datasets]
        states = list(reports[0]["fitted"]["targets"])
        for c, (source, state) in enumerate((s, t) for s in ("fitted", "true body pose, true camera") for t in states):
            for name, color in CURVE_ORDERINGS.items():
                curves = [r[source]["targets"][state]["curves_mean_first_20"].get(name) for r in reports]
                if curves[0] is None:
                    continue
                row[c].plot(reports[0][source]["targets"][state]["curve_sizes"], np.mean(curves, axis=0), c=color, lw=1.2, label=name)
            row[c].set_xscale("log")
            row[c].set_title(f"{label}, state {state}: {source}", fontsize=8)
        row[0].set_ylabel("same-state fraction")
    axes[0, 0].legend(fontsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("kept particles")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=130)


def along(values: np.ndarray) -> np.ndarray:
    """Coordinates along the first principal axis."""
    centred = values - values.mean(axis=0)
    axis = np.linalg.svd(centred[:: max(1, len(centred) // 2000)], full_matrices=False)[2][0]
    return centred @ axis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()
    tables(args.dir)
    curve_figure(args.dir, [(c, [f"case{c}_5k"]) for c in "ABCD"], "figure_curves_5k")
    curve_figure(args.dir, [(f"{c} noise {n}", [f"case{c}_10k_n{n}_s{s}" for s in (0, 1)]) for c in "AB" for n in ("0.1", "0.3")], "figure_curves_10k")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for row, case in zip(axes, ("caseC_5k", "caseA_5k")):
        arrays = np.load(args.dir / f"orderings_{case}_arrays.npz")
        level = arrays["level"]
        for ax, (key, label) in zip(row, SOURCES.items()):
            rotation = along(arrays[f"{key}_rotation_vectors_deg"])
            translation = along(arrays[f"{key}_density_positions"])
            image = ax.scatter(rotation, translation, c=level, s=2, cmap="coolwarm", rasterized=True)
            ax.set_title(f"{case}: {label}", fontsize=9)
            ax.set_xlabel("rotation along the motion (deg)")
            ax.set_ylabel("fitted 3D translation along its axis (A)")
        fig.colorbar(image, ax=row[1], label="true level")
        levels = np.unique(level)
        for key, label in SOURCES.items():
            points, normals = arrays[f"{key}_points"], arrays[f"{key}_normals"]
            means = np.stack([lines_point(points[level == v], normals[level == v]) for v in levels])
            turning = np.stack([arrays[f"{key}_rotation_vectors_deg"][level == v].mean(axis=0) for v in levels])
            row[2].plot(levels, np.linalg.norm(means - means[len(levels) // 2], axis=1), marker="o", ms=3, label=f"translation, {label} (A)")
            row[2].plot(levels, np.linalg.norm(turning - turning[len(levels) // 2], axis=1), marker="s", ms=3, ls="--", label=f"rotation, {label} (deg)")
        row[2].set_xlabel("true level")
        row[2].set_title("distance from the central state, level means", fontsize=9)
        row[2].legend(fontsize=6)
        rotation = along(arrays["fitted_rotation_vectors_deg"])
        twists = np.unique(arrays["body1_twist"])
        row[3].boxplot([rotation[arrays["body1_twist"] == t] for t in twists], labels=[f"{t:g}" for t in twists], showfliers=False)
        row[3].set_xlabel("body 1 twist (deg)")
        row[3].set_ylabel("body 2 fitted rotation along the motion (deg)")
        correlation = np.corrcoef(rotation, arrays["body1_twist"])[0, 1]
        row[3].set_title(f"correlation with the other body's twist {correlation:.3f}", fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.dir / f"figure_motion.{suffix}", dpi=130)


if __name__ == "__main__":
    main()
