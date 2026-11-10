#!/usr/bin/env python3
"""Compare the contrast experiments of two code versions run by run: the same (dataset, arm, seed), old rounds against new.

Both sides are read with ``collect.py`` (the suite scorer's final-checkpoint scores and the in-run latent scores). For each
(dataset, arm, seed) of the new rounds the old value is the one of the latest old round that has the run. The first table is
per run (old, new, new - old); the second is per (dataset, arm): the mean over the shared seeds, the old seeds' range, and whether
the new mean lies inside that range; the third is the contrast arm minus its baseline (``s3_vdam_uniform`` for VDAM arms,
``s3_scale1_4_uniform`` for momentum SGD arms) on kmeans ARI, old and new, so a change of the baseline is not read as a change of
the contrast model.

Run:
    python scripts/ppca_contrast/compare_rounds.py --old ROUND [ROUND ...] --new ROUND [ROUND ...] \
        --datasets-root DIR [--checkpoint 3000] [-o table.md]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ppca_contrast.collect import collect  # noqa: E402

METRICS = ("kmeans ARI", "state FSC", "mean FSC", "pose med (deg)", "corr(c)", "sd prior")
BASELINES = {"vdam": "s3_vdam_uniform", "sgd": "s3_scale1_4_uniform"}


def latest_by_run(rows_per_round: list[list[dict]]) -> dict[tuple, dict]:
    """The row of each (dataset, arm, seed), the later round's winning over an earlier one's."""
    runs = {}
    for rows in rows_per_round:
        for row in rows:
            if row.get("kmeans ARI") is not None:
                runs[(row["dataset"], row["arm"], row["seed"])] = row
    return runs


def family(arm: str) -> str:
    """``vdam`` or ``sgd``, from the arm's name."""
    return "vdam" if "vdam" in arm else "sgd"


def fmt(value) -> str:
    """A value in a table cell."""
    return "" if value is None else f"{value:.3f}"


def per_run_table(old: dict, new: dict) -> list[str]:
    """Lines of the per-run table: each metric as old / new."""
    lines = ["| dataset | arm | seed | " + " | ".join(METRICS) + " |", "|" + " --- |" * (3 + len(METRICS))]
    for key in sorted(new):
        if key not in old:
            continue
        cells = []
        for metric in METRICS:
            a, b = old[key].get(metric), new[key].get(metric)
            cells.append("" if a is None and b is None else f"{fmt(a)} / {fmt(b)}")
        lines.append("| " + " | ".join(list(key) + cells) + " |")
    return lines


def per_arm_table(old: dict, new: dict) -> list[str]:
    """Lines of the per-arm table on kmeans ARI and state FSC: mean over shared seeds, old range, inside or not."""
    lines = ["| dataset | arm | seeds | metric | old mean (min..max) | new mean | new - old | new inside old range |",
             "|" + " --- |" * 8]
    groups = defaultdict(list)
    for key in new:
        if key in old:
            groups[key[:2]].append(key)
    for (dataset, arm), keys in sorted(groups.items()):
        for metric in ("kmeans ARI", "state FSC"):
            o = np.array([old[k][metric] for k in keys if old[k].get(metric) is not None])
            n = np.array([new[k][metric] for k in keys if new[k].get(metric) is not None])
            if not len(o) or not len(n):
                continue
            inside = o.min() <= n.mean() <= o.max() if len(o) > 1 else abs(n.mean() - o.mean()) <= 0.02
            lines.append(
                f"| {dataset} | {arm} | {','.join(k[2] for k in keys)} | {metric} | {o.mean():.3f} ({o.min():.3f}..{o.max():.3f}) "
                f"| {n.mean():.3f} | {n.mean() - o.mean():+.3f} | {'yes' if inside else 'NO'}{'' if len(o) > 1 else ' (one seed: within 0.02)'} |"
            )
    return lines


def gain_table(old: dict, new: dict) -> list[str]:
    """Lines of the table of kmeans ARI of each contrast arm minus its baseline's, per run, old and new."""
    lines = ["| dataset | arm | seed | old: arm - baseline | new: arm - baseline |", "|" + " --- |" * 5]
    for (dataset, arm, seed) in sorted(new):
        base = BASELINES[family(arm)]
        if arm == base or arm.startswith("s3_"):
            continue
        cells = []
        for runs in (old, new):
            a, b = runs.get((dataset, arm, seed)), runs.get((dataset, base, seed))
            cells.append(fmt(a["kmeans ARI"] - b["kmeans ARI"]) if a and b else "")
        if any(cells):
            lines.append(f"| {dataset} | {arm} | {seed} | {cells[0]} | {cells[1]} |")
    return lines


def main() -> None:
    """Print the three comparison tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", type=Path, nargs="+", required=True)
    parser.add_argument("--new", type=Path, nargs="+", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--checkpoint", default="3000")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    old = latest_by_run([collect([r], args.datasets_root, args.checkpoint) for r in args.old])
    new = latest_by_run([collect([r], args.datasets_root, args.checkpoint) for r in args.new])
    text = "\n".join(
        ["Per run (old / new):", ""] + per_run_table(old, new)
        + ["", "Per arm:", ""] + per_arm_table(old, new)
        + ["", "Contrast arm minus baseline on kmeans ARI:", ""] + gain_table(old, new)
    ) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
