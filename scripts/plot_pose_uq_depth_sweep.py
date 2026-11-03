#!/usr/bin/env python
"""Figure of ``run_pose_uq_depth_sweep``: depth RMSE and correlation, in-plane and rotation-feature RMSE against q.

    python -m scripts.plot_pose_uq_depth_sweep --results <pose_uq dir>/full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {("joint", "true"): ("#1baf7a", "--"), ("joint", "fitted"): ("#1baf7a", "-"), ("per-body", "true"): ("#2a78d6", "--"), ("per-body", "fitted"): ("#2a78d6", "-")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    panels = (("depth RMSE (A)", lambda x: x["depth"]["rmse"], lambda x: x["depth"]["true_sd"], "true-depth sd = zero-depth RMSE"),
              ("depth correlation", lambda x: x["depth"]["correlation"], None, None),
              ("in-plane RMSE per direction (A)", lambda x: x["in_plane_rmse_per_direction"], lambda x: x["in_plane_true_sd_per_direction"], "true in-plane sd"),
              ("rotation-feature RMSE (A)", lambda x: x["rotation_features"]["rmse"], lambda x: x["rotation_features"]["mean_only_rmse"], "mean-only RMSE"))
    fig, ax = plt.subplots(4, 4, figsize=(17, 14))
    for i, case in enumerate("ABCD"):
        report = json.loads((args.results / f"depth_sweep_case{case}_5k.json").read_text())
        for row, (label, value, baseline, base_label) in enumerate(panels):
            a = ax[row, i]
            for (arm, poses), (colour, ls) in STYLE.items():
                rows = [x for x in report["rows"] if x["arm"] == arm and x["poses"] == poses and x["body"] == 2]
                a.plot([x["rank"] for x in rows], [value(x) for x in rows], marker="o", color=colour, ls=ls, label=f"{arm}, {poses} poses, body 2")
            rows = [x for x in report["rows"] if x["arm"] == "joint" and x["poses"] == "fitted" and x["body"] == 1]
            a.plot([x["rank"] for x in rows], [value(x) for x in rows], marker="s", color="#eb6834", label="joint, fitted poses, body 1")
            if baseline is not None:
                b2 = [x for x in report["rows"] if x["body"] == 2 and x["poses"] == "fitted"][0]
                a.axhline(baseline(b2), color="#52514e", ls=":", lw=1.5, label=f"{base_label}, body 2 ({baseline(b2):.2f})")
            rel = report.get("relative_q10_fitted")
            if rel is not None and row == 0:
                a.plot([10], [rel["depth"]["rmse"]], "D", color="#6250d6", ms=7, label="relative, fitted, q 10")
            if rel is not None and row == 1:
                a.plot([10], [rel["depth"]["correlation"]], "D", color="#6250d6", ms=7, label="relative, fitted, q 10")
            a.set(xscale="log", xticks=[1, 2, 3, 4, 6, 10], xticklabels=["1", "2", "3", "4", "6", "10"], xlabel="number of PCs q", ylabel=label)
            if row == 0:
                a.set_title(f"case {case}")
            a.grid(color="#e4e3df", lw=0.6)
            a.legend(fontsize=6)
    fig.suptitle("Depth, in-plane and rotation-feature recovery against rank; joint vs per-body, true vs fitted poses. Truth for evaluation only; body 1's own baseline differs.", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(args.results / "fig7_depth_vs_rank.png", dpi=140)


if __name__ == "__main__":
    main()
