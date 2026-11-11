#!/usr/bin/env python3
"""The prior sd of the contrast along the runs of one or more rounds: its peak, when it peaks, how long it stays above a level.

For every run with a contrast latent (``iterations.jsonl`` records with a ``contrast`` entry) the table gives the largest sd of
the run, the iteration of that peak, the number of iterations above ``--level`` (and the first and last of them), the sd at the end,
the number of iterations at the cap ``--cap`` and, from ``embeddings.npz``, the mean contrast estimate and the number of particles whose
estimate fell below the floor. A transient above the true sd is the contrast absorbing a mismatch of the model (docs/math/ppca_contrast.md,
section 7); a run that ends on the cap has not recovered from it.

Run:
    python scripts/ppca_contrast/contrast_sd_trajectory.py ROUND_DIR [ROUND_DIR ...] [--level 0.36] [--cap 0.5]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def run_row(run: Path, level: float, cap: float) -> dict | None:
    """The sd statistics of one run, or None when it has no contrast records."""
    log = run / "iterations.jsonl"
    if not log.exists():
        return None
    sd = [json.loads(line).get("contrast", {}).get("sd") for line in log.read_text().splitlines()]
    sd = np.asarray([value for value in sd if value is not None])
    if not len(sd):
        return None
    above = np.flatnonzero(sd > level)
    row = {
        "max sd": float(sd.max()), "at iteration": int(sd.argmax()), f"iterations > {level}": int(above.size),
        "first / last": f"{above[0]} / {above[-1]}" if above.size else "", "final sd": float(sd[-1]),
        "iterations at cap": int(np.sum(sd >= cap - 1e-4)),
    }
    embeddings = run / "embeddings.npz"
    if embeddings.exists():
        with np.load(embeddings, allow_pickle=False) as saved:
            if "contrast" in saved.files:
                row["mean estimate"] = float(saved["contrast"].mean())
                row["below floor"] = int(saved["n_contrast_below_floor"])
    return row


def main() -> None:
    """Print the table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rounds", type=Path, nargs="+")
    parser.add_argument("--level", type=float, default=0.36)
    parser.add_argument("--cap", type=float, default=0.5)
    args = parser.parse_args()
    rows = []
    for root in args.rounds:
        for run in sorted(root.glob("*/*/seed*")):
            row = run_row(run, args.level, args.cap)
            if row:
                rows.append({"round": root.name, "dataset": "/".join(run.parents[1].name.split("__")[1:3]), "arm": run.parent.name, "seed": run.name, **row})
    columns = list(dict.fromkeys(key for row in rows for key in row))
    print("| " + " | ".join(columns) + " |\n|" + " --- |" * len(columns))
    for row in rows:
        print("| " + " | ".join("" if row.get(c) is None else (f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c])) for c in columns) + " |")


if __name__ == "__main__":
    main()
