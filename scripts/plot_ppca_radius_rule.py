"""Figures of the data_vs_prior radius runs against the stages-table runs (docs/math/vdam_ppca_algorithm.md, 18.6).

Reads the output roots of scripts/ppca_schedule (rounds 1, 3 and 4), prints the scores, the iteration of the pose
entropy step and the radius milestones of every run, and writes:

  radius_scores.png   between-state R^2, mixture ARI and nearest-neighbour accuracy of every arm and seed;
  radius_runs.png     the Fourier radius and the pose entropy over the iterations, one column per seed.

Colors: the stages-table arms are neutral, the data_vs_prior arms orange. The scores are the runs' latent_scores.json.

Run (CPU, about a minute):
  python scripts/plot_ppca_radius_rule.py --round1 ROOT/round1 --round3 ROOT/round3 --round4 ROOT/round4 -o OUT_DIR
"""

import argparse
import functools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ppca_schedule import last_trajectory  # noqa: E402
from plot_ppca_schedule_results import (  # noqa: E402
    GRAY,
    INK,
    INK_2,
    ORANGE,
    SEED_MARKERS,
    SEEDS,
    Arm,
    draw_dots,
    run_dir,
    style,
    x_positions,
)

ARMS = (
    Arm("control", 1, "stages table", "600 iterations, q = 1", INK_2),
    Arm("dvp_600", 4, "data_vs_prior\nradius", "600 iterations, q = 1", ORANGE),
    Arm("long_q1_control", 3, "stages table", "1200 iterations, q = 1", INK_2),
    Arm("dvp_1200", 4, "data_vs_prior\nradius", "1200 iterations, q = 1", ORANGE),
    Arm("long_q3_control", 3, "stages table", "1200 iterations, q = 3", GRAY),
    Arm("dvp_q3_1200", 4, "data_vs_prior\nradius", "1200 iterations, q = 3", ORANGE),
)


@functools.cache
def read_run(directory: Path) -> dict[str, np.ndarray]:
    """Per iteration: the radius, the mean pose entropy and the planned radius (NaN under the stages table)."""
    rows = last_trajectory([json.loads(line) for line in open(directory / "iterations.jsonl")])
    return {
        "iteration": np.array([r["iteration"] for r in rows]),
        "radius": np.array([r["radius"] for r in rows]),
        "entropy": np.array([np.mean([p["pose_entropy_mean"] for p in r["posterior"]]) for r in rows]),
    }


def first_below(run: dict[str, np.ndarray], threshold: float) -> int | None:
    """The first iteration with a mean pose entropy below ``threshold`` (None if there is none)."""
    below = run["entropy"] < threshold
    return int(run["iteration"][np.argmax(below)]) if below.any() else None


def print_table(roots: dict[int, Path]) -> None:
    """Final scores, the first iteration below pose entropy 8 and 2, and the radius milestones of every run."""
    print(f"{'arm':18s} seed     R2    ARI  nn_acc  entropy<8  entropy<2  radius>=16 at  final radius")
    for arm in ARMS:
        for seed in SEEDS:
            directory = run_dir(roots, arm, seed)
            scores = json.loads((directory / "latent_scores.json").read_text())
            run = read_run(directory)
            wide = run["iteration"][run["radius"] >= 16]
            print(
                f"{arm.name:18s} {seed}  {scores['between_state_r2']:.3f}  {scores['ari_gmm']:.3f}  "
                f"{scores['nn_accuracy']:.3f}  {first_below(run, 8)!s:>9s}  {first_below(run, 2)!s:>9s}  "
                f"{int(wide[0]) if wide.size else None!s:>13s}  {int(run['radius'][-1]):>12d}"
            )


def figure_scores(roots: dict[int, Path], out: Path) -> None:
    """Three score panels: the same arms and seeds under R^2, mixture ARI and nearest-neighbour accuracy."""
    positions = x_positions(ARMS)
    scores = {
        (a.name, s): json.loads((run_dir(roots, a, s) / "latent_scores.json").read_text()) for a in ARMS for s in SEEDS
    }
    panels = (
        ("between_state_r2", "Share of the latent variance explained by the true state (between-state R²)"),
        ("ari_gmm", "Adjusted Rand index of an 11-component mixture fitted to the latents (0: chance)"),
        ("nn_accuracy", "Nearest-neighbour label accuracy in the latents (chance 0.091)"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.2), sharex=True, gridspec_kw={"hspace": 0.16})
    for ax, (key, title) in zip(axes, panels):
        values = {k: v[key] for k, v in scores.items()}
        draw_dots(ax, values, positions, ARMS)
        low, high = min(values.values()), max(values.values())
        pad = 0.12 * (high - low)
        ax.set_ylim(min(low, 0.0) - pad if key == "ari_gmm" else low - pad, high + pad)
        ax.set_title(title, loc="left", color=INK_2, fontsize=10)
        ax.grid(axis="x", visible=False)
    for group in dict.fromkeys(arm.group for arm in ARMS):
        xs = [x for arm, x in zip(ARMS, positions) if arm.group == group]
        axes[0].text(np.mean(xs), 1.14, group, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=10, color=INK, weight="bold")  # fmt: skip
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([arm.label for arm in ARMS], fontsize=9)
    handles = [
        plt.Line2D([], [], marker=m, ls="", color=GRAY, ms=8, label=f"seed {s}") for s, m in SEED_MARKERS.items()
    ]
    axes[0].legend(handles=handles, loc="upper left", ncol=3, fontsize=9, bbox_to_anchor=(0.0, 0.93))
    fig.suptitle("The data_vs_prior radius against the stages table (horizontal bars: mean of three seeds)",
                 x=0.07, ha="left", fontsize=12, y=1.02)  # fmt: skip
    fig.savefig(out / "radius_scores.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure_runs(roots: dict[int, Path], out: Path) -> None:
    """Fourier radius (top) and pose entropy (bottom) against the iteration of the 1200-iteration q = 1 arms."""
    arms = (ARMS[2], ARMS[3])
    fig, axes = plt.subplots(2, len(SEEDS), figsize=(13.5, 7.2), sharex=True)
    for col, seed in enumerate(SEEDS):
        for arm in arms:
            run = read_run(run_dir(roots, arm, seed))
            axes[0, col].plot(run["iteration"], run["radius"], color=arm.color, lw=1.8, drawstyle="steps-post")
            axes[1, col].plot(run["iteration"], run["entropy"], color=arm.color, lw=1.6)
        axes[0, col].set_title(f"seed {seed}", loc="left", color=INK_2)
        axes[1, col].set_xlabel("iteration")
        for ax in axes[:, col]:
            ax.grid(axis="x", visible=False)
        axes[1, col].set_ylim(0.8, 13)
    axes[0, 0].set_ylabel("Fourier radius (shells)", color=INK_2)
    axes[1, 0].set_ylabel("pose entropy (mean over images)", color=INK_2)
    handles = [plt.Line2D([], [], color=a.color, lw=2.2, label=a.label.replace("\n", " ")) for a in arms]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("1200 iterations, q = 1: the radius the two rules use, and the pose entropy it gives",
                 x=0.07, ha="left", fontsize=11, y=0.995)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out / "radius_runs.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Print the table and draw the figures into the output directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round1", type=Path, required=True, help="output_root of round 1 (the 600-iteration control)")
    parser.add_argument(
        "--round3", type=Path, required=True, help="output_root of round 3 (the 1200-iteration controls)"
    )
    parser.add_argument("--round4", type=Path, required=True, help="output_root of the data_vs_prior round")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    roots = {1: args.round1, 3: args.round3, 4: args.round4}
    print_table(roots)
    style()
    figure_scores(roots, args.out)
    figure_runs(roots, args.out)
    print(f"\nwrote radius_scores.png and radius_runs.png to {args.out}")


if __name__ == "__main__":
    main()
