#!/usr/bin/env python3
"""Tables of the contrast experiments: the scores of every (dataset, arm, seed) run of one or more rounds.

A run directory is ``<round>/<dataset>/<arm>/seed<N>`` as the PPCA schedule workflow writes it. Per run the table has
the suite scorer's checkpoint scores (the final checkpoint) in the owner's order, ARI and the FSC metrics first, then
the pose fit, the loading metrics, the in-run latent scores (``latent_scores.json``) and, for a run with the contrast
latent, how well its contrast estimate follows the true contrast of the dataset (``evaluation/contrast.npy``) and the
sd it ended at.

Run:
    python scripts/ppca_contrast/collect.py ROUND_DIR [ROUND_DIR ...] --datasets-root <dir> [--checkpoint 3000] [-o table.md]
"""

import argparse
import json
from pathlib import Path

import numpy as np

# (column, source, key): source "suite" reads the checkpoint's scores, "latent" the in-run latent scores.
COLUMNS = (
    ("kmeans ARI", "suite", "kmeans_ari"),
    ("state FSC", "suite", "state_fsc_direct"),
    ("mean FSC", "suite", "mean_fsc_to_true_mean"),
    ("pose med (deg)", "suite", "pose_direct_median_deg"),
    ("pose <10", "suite", "pose_direct_within_10deg"),
    ("subspace (deg)", "suite", "subspace_max_angle_deg"),
    ("energy", "suite", "true_energy_captured"),
    ("PC/mean", "suite", "pc_over_mean_power"),
    ("gmm ARI", "latent", "ari_gmm"),
    ("nn acc", "latent", "nn_accuracy"),
    ("R2 states", "latent", "between_state_r2"),
)


def read_json(path: Path) -> dict | None:
    """The JSON at ``path``, or None when it does not exist."""
    return json.loads(path.read_text()) if path.exists() else None


def contrast_columns(run: Path, evaluation: Path) -> dict:
    """The contrast estimate's correlation with the true contrast, its spread and the sd the run ended at; empty for a
    run without the contrast latent."""
    embeddings = run / "embeddings.npz"
    if not embeddings.exists() or not (evaluation / "contrast.npy").exists():
        return {}
    with np.load(embeddings, allow_pickle=False) as saved:
        if "contrast" not in saved.files:
            return {}
        ids, estimate, sd = saved["particle_ids"], saved["contrast"], float(saved["contrast_sd"])
    truth = np.load(evaluation / "contrast.npy")[ids]
    return {"corr(c)": float(np.corrcoef(estimate, truth)[0, 1]), "sd(c est)": float(estimate.std()), "sd prior": sd}


def wall_minutes(run: Path) -> float | None:
    """Training wall time from the iteration log, in minutes."""
    log = run / "iterations.jsonl"
    if not log.exists():
        return None
    return sum(json.loads(line)["elapsed_seconds"] for line in log.read_text().splitlines()) / 60


def collect(rounds: list[Path], datasets_root: Path, checkpoint: str) -> list[dict]:
    """One row per run found under the round directories."""
    rows = []
    for root in rounds:
        for run in sorted(root.glob("*/*/seed*")):
            dataset, arm, seed = run.parents[1].name, run.parent.name, run.name.removeprefix("seed")
            suite = (read_json(run / "suite_scores.json") or {}).get("checkpoints", {}).get(checkpoint, {})
            latent = read_json(run / "latent_scores.json") or {}
            row = {"dataset": dataset.split("__")[1], "arm": arm, "seed": seed}
            for name, source, key in COLUMNS:
                row[name] = (suite if source == "suite" else latent).get(key)
            row["wall (min)"] = wall_minutes(run)
            row.update(contrast_columns(run, datasets_root / dataset / "evaluation"))
            rows.append(row)
    return rows


def markdown(rows: list[dict]) -> str:
    """The rows as a Markdown table, the columns in the order they first appear."""
    columns = list(dict.fromkeys(key for row in rows for key in row))

    def cell(value) -> str:
        return "" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    lines += ["| " + " | ".join(cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Collect and print (or write) the table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rounds", type=Path, nargs="+")
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--checkpoint", default="3000")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    table = markdown(collect(args.rounds, args.datasets_root, args.checkpoint))
    if args.output:
        args.output.write_text(table)
    print(table)


if __name__ == "__main__":
    main()
