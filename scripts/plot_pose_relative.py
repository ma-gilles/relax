#!/usr/bin/env python
"""Same-state curves of ``scripts.run_pose_relative_comparison``. ``python -m scripts.plot_pose_relative --dir <dir>``"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SETS = {"own pose": "-", "relative": "--", "body 1 rotation regressed out": ":", "relative, true cameras": "-."}
ORDERINGS = {"bounded": "C0", "rotation only": "C1", "rank 1": "C3", "rank 2": "C4"}


def figure(directory: Path, rows: list[tuple[str, list[str]]], stem: str) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 3 * len(rows)), squeeze=False, sharey=True)
    for row, (label, datasets) in zip(axes, rows):
        reports = [json.loads((directory / f"relative_{d}.json").read_text())["sets"] for d in datasets]
        states = list(reports[0]["own pose"]["targets"])
        for ax, state in zip(row, states):
            for name, style in SETS.items():
                entries = [r[name]["targets"][state] for r in reports]
                for ordering, color in ORDERINGS.items():
                    ax.plot(entries[0]["curve_sizes"], np.mean([e["curves_mean_first_20"][ordering] for e in entries], axis=0),
                            c=color, ls=style, lw=1, label=f"{ordering}, {name}")
            ax.plot(entries[0]["curve_sizes"], np.mean([e["curves_mean_first_20"]["perfect"] for e in entries], axis=0), c="k", lw=1.2, label="perfect")
            ax.set_xscale("log")
            ax.set_title(f"{label}, state {state}", fontsize=9)
        row[0].set_ylabel("same-state fraction")
    axes[0, 0].legend(fontsize=5, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel("kept particles")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=130)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path)
    args = parser.parse_args()
    figure(args.dir, [(c, [f"case{c}_5k"]) for c in "ABCD"], "figure_relative_5k")
    figure(args.dir, [(f"{c} noise {n}", [f"case{c}_10k_n{n}_s{s}" for s in (0, 1)]) for c in "AB" for n in ("0.1", "0.3")], "figure_relative_10k")


if __name__ == "__main__":
    main()
