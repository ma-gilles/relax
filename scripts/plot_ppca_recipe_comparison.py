"""The recipe schedule against the local radius and order schedule, under VDAM and momentum SGD (round 9).

Reads one output_root of scripts/ppca_schedule (arms vdam_recipe, vdam_local, sgd_recipe, sgd_local; seeds 11 to 13).
Prints for every run its scores (latent_scores.json), the iteration of its first pose entropy below 8, 3 and 2, the
iteration its Fourier radius first reaches 15 and first exceeds it, its final radius, HEALPix order and pose entropy, the
order changes, the pose accuracy of the last order-rule evaluation and the summed iteration time. Writes
  recipe_trajectories.png  pose entropy, Fourier radius and HEALPix order over the iterations, a column per optimizer;
  recipe_scores.png        R^2, ARI, first-PC rank correlation and nearest-neighbour accuracy, a dot per seed.

iterations.jsonl holds a per-direction prior of several MB on every line, so the lines are read one at a time with
that field cut out before parsing.

Run (CPU, minutes: the local arms' iterations.jsonl are 8 to 20 GB):
  python scripts/plot_ppca_recipe_comparison.py --root ROOT/round9 -o OUT_DIR
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ppca_schedule import last_trajectory  # noqa: E402
from plot_ppca_schedule_results import BLUE, GRAY, INK_2, SEED_MARKERS, SEEDS, style  # noqa: E402

PRIOR_KEY = '"direction_prior": ['
SCORES = (
    ("between_state_r2", "between-state $R^2$"),
    ("ari_gmm", "ARI of a Gaussian mixture"),
    ("spearman_first", "first PC, rank correlation"),
    ("nn_accuracy", "nearest-neighbour accuracy"),
)


@dataclass(frozen=True)
class Arm:
    """One arm of the comparison: its run directory name, its legend entry, its optimizer and its color."""

    name: str
    label: str
    optimizer: str
    color: str


ARMS = (
    Arm("vdam_recipe", "recipe schedule", "VDAM", GRAY),
    Arm("vdam_local", "local radius and order", "VDAM", BLUE),
    Arm("sgd_recipe", "recipe schedule", "momentum SGD", GRAY),
    Arm("sgd_local", "local radius and order", "momentum SGD", BLUE),
)
OPTIMIZERS = ("VDAM", "momentum SGD")


def read_trajectory(path: Path) -> dict[str, np.ndarray]:
    """The last trajectory of ``path`` as per-iteration arrays: iteration, entropy, radius, order, seconds, acc_rot."""
    rows = []
    with open(path) as stream:
        for line in stream:
            start = line.find(PRIOR_KEY)
            if start >= 0:
                line = line[:start] + '"direction_prior": null' + line[line.index("]", start) + 1 :]
            row = json.loads(line)
            rows.append(
                {
                    "iteration": row["iteration"],
                    "entropy": float(np.mean([p["pose_entropy_mean"] for p in row["posterior"]])),
                    "radius": row["radius"],
                    "order": row["healpix_order"],
                    "seconds": row["elapsed_seconds"],
                    "acc_rot": (row.get("pose_accuracy") or {}).get("acc_rot", np.nan),
                }
            )
    rows = last_trajectory(rows)
    return {key: np.array([r[key] for r in rows]) for key in rows[0]}


def first_below(run: dict[str, np.ndarray], threshold: float) -> int | None:
    """The first iteration with a mean pose entropy below ``threshold`` (None if there is none)."""
    below = run["entropy"] < threshold
    return int(run["iteration"][np.argmax(below)]) if below.any() else None


def order_changes(run: dict[str, np.ndarray]) -> str:
    """The orders of the run with the iteration each is first used at, like ``1:1 11:2``."""
    changed = np.flatnonzero(np.diff(run["order"]) != 0) + 1
    return " ".join(f"{run['iteration'][i]}:{run['order'][i]}" for i in [0, *changed])


def read_scores(directory: Path) -> dict[str, float]:
    """The latent scores of a run (NaN for each when the run has not been scored)."""
    path = directory / "latent_scores.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    return {key: saved.get(key, float("nan")) for key, _ in SCORES}


def run_directories(root: Path) -> dict[tuple[str, int], Path]:
    """The run directory of every arm and seed that has an iterations.jsonl."""
    found = {(a.name, s): root / a.name / f"seed{s}" for a in ARMS for s in SEEDS}
    return {key: d for key, d in found.items() if (d / "iterations.jsonl").exists()}


def print_table(runs: dict[tuple[str, int], dict[str, np.ndarray]], root: Path) -> None:
    """One line per run: the scores, the entropy crossings, the radius milestones, the orders and the time."""
    print(
        f"{'arm':12s} seed  iters   R2     ARI    rho1   nn     <8     <3     <2   r>=15  r>15  r_end  order_end  "
        f"H_end  acc_rot  hours  orders (iteration:order)"
    )
    for (name, seed), run in runs.items():
        score = read_scores(root / name / f"seed{seed}")
        wide = run["iteration"][run["radius"] >= 15]
        opened = run["iteration"][run["radius"] > 15]
        measured = run["acc_rot"][~np.isnan(run["acc_rot"])]
        print(
            f"{name:12s} {seed}  {int(run['iteration'][-1]):5d}  "
            f"{score['between_state_r2']:.3f}  {score['ari_gmm']:.3f}  {score['spearman_first']:.3f}  "
            f"{score['nn_accuracy']:.3f}  "
            + " ".join(f"{first_below(run, x)!s:>6s}" for x in (8, 3, 2))
            + f"  {int(wide[0]) if wide.size else None!s:>5s} {int(opened[0]) if opened.size else None!s:>5s}  "
            f"{int(run['radius'][-1]):5d}  {int(run['order'][-1]):9d}  "
            f"{run['entropy'][-1]:5.2f}  {measured[-1] if measured.size else float('nan'):7.2f}  {run['seconds'].sum() / 3600:5.2f}  "
            + order_changes(run)
        )


def trajectories(runs: dict[tuple[str, int], dict[str, np.ndarray]], out: Path) -> None:
    """Pose entropy, radius and order against the iteration: a column per optimizer, the seeds of an arm one color."""
    rows = (
        ("entropy", "pose entropy", (0, 13)),
        ("radius", "Fourier radius (shells)", (0, 33)),
        ("order", "HEALPix order", (0, 5)),
    )
    fig, axes = plt.subplots(len(rows), len(OPTIMIZERS), figsize=(10.5, 8.4), sharex=True, squeeze=False)
    for col, optimizer in enumerate(OPTIMIZERS):
        arms = [a for a in ARMS if a.optimizer == optimizer]
        for row, (key, label, limits) in enumerate(rows):
            ax = axes[row, col]
            for arm in arms:
                for seed in SEEDS:
                    run = runs.get((arm.name, seed))
                    if run is not None:
                        ax.plot(run["iteration"], run[key], color=arm.color, lw=1.3, alpha=0.9)
            ax.set_ylim(*limits)
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(optimizer, loc="left", color=INK_2, fontsize=10)
                ax.legend(
                    handles=[plt.Line2D([], [], color=a.color, lw=2, label=a.label) for a in arms],
                    loc="upper right",
                    fontsize=8,
                )
            if col == 0:
                ax.set_ylabel(label, color=INK_2)
            if row == len(rows) - 1:
                ax.set_xlabel("iteration")
    fig.suptitle("Recipe schedule against local radius and order, three seeds per arm", x=0.06, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def scores_figure(root: Path, out: Path) -> None:
    """The four latent scores of every arm, one dot per seed and a bar at the mean of the scored seeds."""
    fig, axes = plt.subplots(1, len(SCORES), figsize=(12.5, 3.8), squeeze=False)
    for ax, (key, label) in zip(axes[0], SCORES):
        for x, arm in enumerate(ARMS):
            values = [read_scores(root / arm.name / f"seed{s}")[key] for s in SEEDS]
            for s, v in zip(SEEDS, values):
                ax.plot([x], [v], SEED_MARKERS[s], color=arm.color, ms=6, mec="white", mew=0.8)
            if not np.all(np.isnan(values)):
                ax.hlines(np.nanmean(values), x - 0.3, x + 0.3, color=arm.color, lw=1.5)
        ax.set_xticks(
            range(len(ARMS)),
            [f"{a.optimizer.replace('momentum ', '')}\n{a.name.split('_')[1]}" for a in ARMS],
            fontsize=8,
        )
        ax.set_title(label, loc="left", color=INK_2, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
    fig.legend(
        handles=[plt.Line2D([], [], marker=SEED_MARKERS[s], ls="", color=GRAY, label=f"seed {s}") for s in SEEDS],
        loc="lower center",
        ncol=len(SEEDS),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Read the runs, print the table and draw the figures into the output directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the output_root of the round")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = {key: read_trajectory(d / "iterations.jsonl") for key, d in run_directories(args.root).items()}
    print_table(runs, args.root)
    style()
    trajectories(runs, args.out / "recipe_trajectories.png")
    scores_figure(args.root, args.out / "recipe_scores.png")
    print(f"\nwrote recipe_trajectories.png and recipe_scores.png to {args.out}")


if __name__ == "__main__":
    main()
