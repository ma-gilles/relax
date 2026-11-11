#!/usr/bin/env python3
"""Compare arms run by run: the same (dataset, seed), one column per arm, one table per metric.

The runs are read with ``collect.py`` (the suite scorer's final-checkpoint scores, the in-run latent scores, the contrast estimate's
correlation with the truth and the sd it ended at, the wall time). Rounds are merged in order, a later round's run winning over an earlier
one's of the same (dataset, arm, seed). A row appears when every arm of ``--arms`` has the run, so the columns are paired.

Run:
    python scripts/ppca_contrast/compare_arms.py ROUND [ROUND ...] --arms s3_vdam_uniform co_vdam qo_vdam \
        --datasets-root <dir> [--metrics "kmeans ARI" "state FSC" ...] [-o table.md]
"""

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.ppca_contrast.collect import collect  # noqa: E402

DEFAULT_METRICS = ("kmeans ARI", "state FSC", "mean FSC", "pose med (deg)", "pose <10", "subspace (deg)", "energy", "corr(c)", "sd prior", "wall (min)")


def merged_runs(rounds: list[Path], datasets_root: Path, checkpoint: str) -> dict[tuple, dict]:
    """The row of each (dataset, arm, seed) over the rounds, later rounds winning."""
    runs = {}
    for root in rounds:
        for row in collect([root], datasets_root, checkpoint):
            if row.get("kmeans ARI") is not None:
                runs[(row["dataset"], row["arm"], row["seed"])] = row
    return runs


def tables(runs: dict[tuple, dict], arms: list[str], metrics: list[str]) -> str:
    """One Markdown table per metric: a row per (dataset, seed) that all arms have, a column per arm, then the mean over rows."""
    keys = sorted({(d, s) for d, a, s in runs if all((d, arm, s) in runs for arm in arms)})
    lines = []
    for metric in metrics:
        values = [[runs[(d, arm, s)].get(metric) for arm in arms] for d, s in keys]
        if all(v is None for row in values for v in row):
            continue
        lines += [f"\n{metric}\n", "| dataset | seed | " + " | ".join(arms) + " |", "|" + " --- |" * (2 + len(arms))]
        for (d, s), row in zip(keys, values):
            lines.append(f"| {d} | {s} | " + " | ".join("" if v is None else f"{v:.3f}" for v in row) + " |")
        complete = [row for row in values if all(v is not None for v in row)]
        if complete:
            means = [sum(row[k] for row in complete) / len(complete) for k in range(len(arms))]
            lines.append(f"| mean over {len(complete)} | | " + " | ".join(f"{m:.3f}" for m in means) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Print (or write) the per-metric tables."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rounds", type=Path, nargs="+")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--checkpoint", default="3000")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    text = tables(merged_runs(args.rounds, args.datasets_root, args.checkpoint), args.arms, args.metrics)
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
