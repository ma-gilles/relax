"""Figures of the longer PPCA schedule runs with several PCs (docs/math/vdam_ppca_algorithm.md, section 18.6).

Reads the output roots of scripts/ppca_schedule (rounds 1 and 3), prints the mean scores of every arm and writes:

  long_scores.png    between-state R^2, mixture ARI and nearest-neighbour accuracy of every arm and seed;
  long_axes.png      the R^2 split over the principal axes of the latent cloud, per arm and seed;
  long_pcs.png       loading singular value of every PC and latent posterior variance over the iterations;
  long_entropy.png   pose entropy over the iterations of every long run (dashed: final R^2 below 0.3).

Colors: arms that start with all their PCs are neutral, the rest take the first three slots of the reference
categorical palette. Ground truth is for evaluation only.

Run (CPU, about a minute):
  python scripts/plot_ppca_longer_runs.py --round1 ROOT/round1 --round3 ROOT/round3 \
      --truth FIXTURE/evaluation/truth.npz -o OUT_DIR
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ppca_latents import between_state_r2  # noqa: E402
from analyze_ppca_schedule import last_trajectory  # noqa: E402
from plot_ppca_schedule_results import (  # noqa: E402
    AQUA,
    BLUE,
    GRAY,
    GRID,
    INK,
    INK_2,
    ORANGE,
    SEED_MARKERS,
    SEEDS,
    SURFACE,
    Arm,
    draw_dots,
    load_latent,
    run_dir,
    style,
    x_positions,
)  # fmt: skip

ARMS = (
    Arm("control", 1, "all PCs\nfrom the start", "600 iterations, q = 1", INK_2),
    Arm("pc_at_480", 1, "PC joins\nat 480", "600 iterations, q = 1", BLUE),
    Arm("long_q1_control", 3, "all PCs\nfrom the start", "1200 iterations, q = 1", INK_2),
    Arm("long_q1_pc_at_480", 3, "PC joins\nat 480", "1200 iterations, q = 1", BLUE),
    Arm("long_q3_control", 3, "3 PCs\nfrom the start", "1200 iterations, q = 3", GRAY),
    Arm("long_q3_every_120", 3, "PCs join at\n480, 600, 720", "1200 iterations, q = 3", ORANGE),
    Arm("long_q3_every_240", 3, "PCs join at\n480, 720, 960", "1200 iterations, q = 3", AQUA),
)
Q3_ARMS = ARMS[4:]
JOINS = {"long_q3_control": (), "long_q3_every_120": (480, 600, 720), "long_q3_every_240": (480, 720, 960)}
STAGE_STARTS = ((200, "radius 8"), (400, "radius 16"))
PC_COLORS = (BLUE, ORANGE, AQUA)
AXIS_SHADES = ("#52514e", "#9a9993", "#d3d2cc")


def read_scores(roots: dict[int, Path]) -> dict[tuple[str, int], dict]:
    """The latent_scores.json of every arm and seed."""
    return {
        (arm.name, seed): json.loads((run_dir(roots, arm, seed) / "latent_scores.json").read_text())
        for arm in ARMS
        for seed in SEEDS
    }


def nn_chance(labels: np.ndarray) -> float:
    """The nearest-neighbour accuracy of latents that carry no information: the sum of squared state shares."""
    share = np.bincount(labels) / len(labels)
    return float(np.sum(share**2))


def print_means(scores: dict[tuple[str, int], dict]) -> None:
    """Mean over the seeds of every score, one row per arm."""
    keys = ("between_state_r2", "ari_gmm", "nn_accuracy")
    print(f"{'arm':22s}" + "".join(f"{k:>18s}" for k in keys) + "   per-seed R2")
    for arm in ARMS:
        means = [np.mean([scores[arm.name, s][k] for s in SEEDS]) for k in keys]
        per_seed = " / ".join(f"{scores[arm.name, s]['between_state_r2']:.3f}" for s in SEEDS)
        print(f"{arm.name:22s}" + "".join(f"{m:18.3f}" for m in means) + f"   {per_seed}")


def figure_scores(scores: dict[tuple[str, int], dict], labels: np.ndarray, out: Path) -> None:
    """Three score panels: the same arms and seeds under R^2, mixture ARI and nearest-neighbour accuracy."""
    positions = x_positions(ARMS)
    panels = (
        ("between_state_r2", "Share of the latent variance explained by the true state (between-state R²)", None),
        ("ari_gmm", "Adjusted Rand index of an 11-component mixture fitted to the latents (0: chance)", 0.0),
        ("nn_accuracy", "Nearest-neighbour label accuracy in the latents", nn_chance(labels)),
    )
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 10.2), sharex=True, gridspec_kw={"hspace": 0.16})
    for ax, (key, title, chance) in zip(axes, panels):
        values = {k: v[key] for k, v in scores.items()}
        draw_dots(ax, values, positions, ARMS)
        low, high = min(values.values()), max(values.values())
        if chance is not None:
            ax.axhline(chance, color=GRAY, lw=1.0, zorder=0)
            ax.text(positions[-1] + 0.9, chance, "chance", va="center", fontsize=8.5, color=INK_2)
            low = min(low, chance)
        pad = 0.12 * (high - low)
        ax.set_ylim(low - pad, high + pad)
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
    fig.suptitle("Longer runs and more PCs: separation of the eleven states (horizontal bars: mean of three seeds)",
                 x=0.07, ha="left", fontsize=12, y=1.02)  # fmt: skip
    fig.savefig(out / "long_scores.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def axis_contributions(z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The share of the total latent variance that the states explain along each principal axis of ``z``.

    The axes are those of the latent cloud, largest variance first; the shares sum to ``between_state_r2``.
    """
    centered = z - z.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ axes.T
    between = np.zeros(coords.shape[1])
    for state in np.unique(labels):
        between += np.sum(labels == state) * coords[labels == state].mean(axis=0) ** 2
    return between / np.sum(coords**2)


def figure_axes(roots: dict[int, Path], labels: np.ndarray, out: Path) -> None:
    """Stacked bars: the R^2 of every long run split over the principal axes of its latents."""
    arms = ARMS[2:]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x, ticks, tick_labels, centers = 0.0, [], [], []
    for arm in arms:
        first = x
        for seed in SEEDS:
            z = load_latent(run_dir(roots, arm, seed))
            parts = axis_contributions(z, labels)
            assert abs(parts.sum() - between_state_r2(z, labels)) < 1e-9
            bottom = 0.0
            for k, part in enumerate(parts):
                ax.bar(x, part, bottom=bottom, width=0.8, color=AXIS_SHADES[k], edgecolor=SURFACE, linewidth=1.5,
                       label=f"axis {k + 1}" if arm is arms[-1] and seed == SEEDS[0] else None)  # fmt: skip
                bottom += part
            ax.text(x, bottom + 0.004, f"{bottom:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
            ticks.append(x)
            tick_labels.append(f"{seed}")
            x += 1.0
        centers.append((arm, (first + x - 1.0) / 2))
        x += 0.9
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=9)
    for arm, center in centers:
        ax.text(center, -0.09, f"{arm.group.split(', ')[1]}\n{arm.label}", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=8.5, color=INK)  # fmt: skip
    ax.set_ylabel("between-state R², split by axis")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    fig.suptitle("Which latent axes carry the state information (axes: principal axes of the latent cloud; x: seed)",
                 x=0.07, ha="left", fontsize=11, y=0.99)  # fmt: skip
    fig.savefig(out / "long_axes.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def read_pcs(directory: Path) -> dict[str, np.ndarray]:
    """Per iteration: the loading singular value of each PC and the latent posterior variance per active PC."""
    rows = last_trajectory([json.loads(line) for line in open(directory / "iterations.jsonl")])
    traces = np.array([np.mean([p["latent_covariance_trace_mean"] for p in r["posterior"]]) for r in rows])
    active = np.array([r.get("active_pcs", len(r["loading_singular_values"])) for r in rows])
    return {
        "iteration": np.array([r["iteration"] for r in rows]),
        "singular": np.array([r["loading_singular_values"] for r in rows]),
        "variance": np.where(active > 0, traces / np.maximum(active, 1), np.nan),
    }


def print_convergence(roots: dict[int, Path]) -> None:
    """The final singular values of each PC and their change over the last 120 iterations."""
    print("\nfinal loading singular values (x1e3) and change over the last 120 iterations")
    for arm in Q3_ARMS:
        for seed in SEEDS:
            data = read_pcs(run_dir(roots, arm, seed))
            last, before = data["singular"][-1], data["singular"][data["iteration"] <= data["iteration"][-1] - 120][-1]
            change = (last / np.where(before > 0, before, np.nan) - 1) * 100
            print(
                f"  {arm.name:20s} seed {seed}: "
                + "  ".join(f"{v * 1e3:6.2f} ({c:+5.1f}%)" for v, c in zip(last, change))
            )


def figure_pcs(roots: dict[int, Path], out: Path) -> None:
    """Singular value of every PC and posterior variance per PC against the iteration, for seeds 11 and 13."""
    seeds = (11, 13)
    fig, axes = plt.subplots(2 * len(seeds), len(Q3_ARMS), figsize=(13.5, 10.5), sharex=True)
    for col, arm in enumerate(Q3_ARMS):
        for i, seed in enumerate(seeds):
            data = read_pcs(run_dir(roots, arm, seed))
            ax_sv, ax_var = axes[i, col], axes[len(seeds) + i, col]
            for k in range(data["singular"].shape[1]):
                y = np.where(data["singular"][:, k] > 0, data["singular"][:, k], np.nan)
                ax_sv.plot(data["iteration"], y, color=PC_COLORS[k], lw=1.8, label=f"PC {k + 1}")
            ax_var.plot(data["iteration"], data["variance"], color=INK_2, lw=1.8)
            ax_sv.set_yscale("log")
            for ax in (ax_sv, ax_var):
                for x, _ in STAGE_STARTS:
                    ax.axvline(x, color=GRID, lw=1.0, zorder=0)
                for x in JOINS[arm.name]:
                    ax.axvline(x, color=GRAY, lw=1.0, zorder=0)
                ax.grid(axis="x", visible=False)
            if i == 0:
                ax_sv.set_title(f"{arm.label.replace(chr(10), ' ')}", loc="left", color=INK)
            if col == 0:
                ax_sv.set_ylabel(f"seed {seed}\nloading singular value", color=INK_2)
                ax_var.set_ylabel(f"seed {seed}\nposterior variance per PC", color=INK_2)
    axes[0, 0].legend(loc="lower right", fontsize=9, ncol=3)
    for ax in axes[-1]:
        ax.set_xlabel("iteration")
    fig.suptitle("How the PCs train in the three-PC runs (gray lines: PCs join; light lines: radius 8 at 200, 16 at 400)\n"
                 "singular value of the loadings, and the mean latent posterior variance per active PC (prior: 1)",
                 x=0.07, ha="left", fontsize=11, y=0.995)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "long_pcs.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure_entropy(roots: dict[int, Path], scores: dict[tuple[str, int], dict], out: Path) -> None:
    """Pose entropy against the iteration for every long run, one panel per seed."""
    fig, axes = plt.subplots(1, len(SEEDS), figsize=(13.5, 4.6), sharey=True)
    for ax, seed in zip(axes, SEEDS):
        for arm in ARMS[2:]:
            rows = last_trajectory([json.loads(line) for line in open(run_dir(roots, arm, seed) / "iterations.jsonl")])
            entropy = [np.mean([p["pose_entropy_mean"] for p in r["posterior"]]) for r in rows]
            collapsed = scores[arm.name, seed]["between_state_r2"] < 0.3
            ax.plot(
                [r["iteration"] for r in rows], entropy, color=arm.color, lw=1.8, ls=(0, (4, 2)) if collapsed else "-"
            )
        ax.axvline(400, color=GRID, lw=1.0, zorder=0)
        ax.set_title(f"seed {seed}", loc="left", color=INK_2)
        ax.set_xlabel("iteration")
        ax.set_ylim(0.8, 6.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("pose entropy (mean over images)")
    axes[0].text(404, 6.3, "radius 16", fontsize=8.5, color=INK_2, va="top")
    handles = [
        plt.Line2D([], [], color=a.color, lw=2.2, label=f"q={a.group[-1]}, {a.label.replace(chr(10), ' ')}")
        for a in ARMS[2:]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Pose entropy of the 1200-iteration runs: the runs that end near 1.3 have separated the states\n"
                 "dashed: between-state R² below 0.3 at the end", x=0.07, ha="left", fontsize=11, y=1.04)  # fmt: skip
    fig.savefig(out / "long_entropy.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Print the score table and draw the figures into the output directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round1", type=Path, required=True, help="output_root of the first Snakemake round")
    parser.add_argument("--round3", type=Path, required=True, help="output_root of the longer-run round")
    parser.add_argument("--truth", type=Path, required=True, help="the fixture's evaluation/truth.npz (labels)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    roots = {1: args.round1, 3: args.round3}
    with np.load(args.truth, allow_pickle=False) as saved:
        labels = saved["labels"]
    scores = read_scores(roots)
    print_means(scores)
    print_convergence(roots)
    style()
    figure_scores(scores, labels, args.out)
    figure_axes(roots, labels, args.out)
    figure_pcs(roots, args.out)
    figure_entropy(roots, scores, args.out)
    print(f"\nwrote long_scores.png, long_axes.png, long_pcs.png, long_entropy.png to {args.out}")


if __name__ == "__main__":
    main()
