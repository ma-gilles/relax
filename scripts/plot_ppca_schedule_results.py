"""Figures of the PPCA schedule experiments (docs/math/vdam_ppca_algorithm.md, section 18.4).

Reads the output roots of scripts/ppca_schedule (one per round) and writes four PNG files:

  r2_by_arm.png          between-state R^2 of every arm and seed, and the same on the ranks of the latents;
  latents_by_state.png   the latent coordinate of every true state (median and interquartile range), per arm and seed;
  latent_distribution.png  the distribution of the latent coordinate (log density) against a normal one;
  training_dynamics.png  pose entropy, loading power and latent posterior variance over the iterations.

Colors: the control is neutral, the three variants the first three slots of the reference categorical palette
(validated all-pairs for color-vision deficiency); every other arm is gray. Ground truth is for evaluation only.

Run (CPU, about a minute):
  python scripts/plot_ppca_schedule_results.py --round1 ROOT/round1 --round2 ROOT/round2 \
      --truth FIXTURE/evaluation/truth.npz -o OUT_DIR
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
from scipy.stats import kurtosis, norm, rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_ppca_latents import between_state_r2  # noqa: E402
from analyze_ppca_schedule import last_trajectory  # noqa: E402

SEEDS = (11, 12, 13)
SURFACE, INK, INK_2, GRID, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#9a9993"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEED_MARKERS = {11: "o", 12: "s", 13: "^"}


@dataclass(frozen=True)
class Arm:
    """One experiment arm: where its runs are, what to call it and how to draw it."""

    name: str
    round: int
    label: str
    group: str
    color: str


ARMS = (
    Arm("control", 1, "all PCs\nfrom the start", "radius 16, order 3", INK_2),
    Arm("pc_at_200", 1, "PC joins\nat 200", "radius 16, order 3", GRAY),
    Arm("pc_at_400", 1, "PC joins\nat 400", "radius 16, order 3", GRAY),
    Arm("pc_at_480", 1, "PC joins\nat 480", "radius 16, order 3", BLUE),
    Arm("r8_control", 2, "all PCs\nfrom the start", "radius 8, order 3", GRAY),
    Arm("r8_pc_at_200", 2, "PC joins\nat 200", "radius 8, order 3", GRAY),
    Arm("hp4_r16", 2, "all PCs\nfrom the start", "radius 16, order 4", ORANGE),
    Arm("hp4_r16_pc_at_480", 2, "PC joins\nat 480", "radius 16, order 4", AQUA),
)
KEY_ARMS = (ARMS[0], ARMS[3], ARMS[6], ARMS[7])
KEY_NAMES = {
    "control": "control",
    "pc_at_480": "PC joins at 480",
    "hp4_r16": "order 4",
    "hp4_r16_pc_at_480": "order 4, PC joins at 480",
}


def style() -> None:
    """Recessive chrome: hairline solid grid, light surface, text in ink tokens."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2,
            "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
            "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
            "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "regular", "legend.frameon": False,
        }
    )  # fmt: skip


def run_dir(roots: dict[int, Path], arm: Arm, seed: int) -> Path:
    """The output directory of one arm and seed."""
    return roots[arm.round] / arm.name / f"seed{seed}"


def load_latent(directory: Path) -> np.ndarray:
    """The final posterior-mean latents of every particle, in particle order, as ``(N, q)``."""
    with np.load(directory / "embeddings.npz", allow_pickle=False) as saved:
        return saved["z"].astype(np.float64)[np.argsort(saved["particle_ids"])]


def x_positions(arms: tuple[Arm, ...] = ARMS) -> list[float]:
    """Arm positions on the x axis: a gap between the groups."""
    positions, last_group, x = [], None, 0.0
    for arm in arms:
        x += 0.0 if last_group is None else (1.0 if arm.group == last_group else 1.8)
        positions.append(x)
        last_group = arm.group
    return positions


def draw_dots(ax, values: dict[tuple[str, int], float], positions: list[float], arms: tuple[Arm, ...] = ARMS) -> None:
    """Per-seed dots (shape = seed), same-seed lines inside a group, and the mean of each arm as a bar."""
    for seed, offset in zip(SEEDS, (-0.14, 0.0, 0.14)):
        for group in dict.fromkeys(arm.group for arm in arms):
            idx = [i for i, arm in enumerate(arms) if arm.group == group]
            ax.plot([positions[i] + offset for i in idx], [values[arms[i].name, seed] for i in idx],
                    color=GRID, lw=1.0, zorder=1)  # fmt: skip
        for arm, x in zip(arms, positions):
            ax.plot(x + offset, values[arm.name, seed], SEED_MARKERS[seed], color=arm.color, ms=8,
                    mec=SURFACE, mew=1.5, zorder=3)  # fmt: skip
    for arm, x in zip(arms, positions):
        mean = np.mean([values[arm.name, s] for s in SEEDS])
        ax.plot([x - 0.3, x + 0.3], [mean, mean], color=arm.color, lw=2.2, zorder=2, solid_capstyle="butt")
        ax.text(x + 0.33, mean, f"{mean:.3f}", va="center", ha="left", fontsize=9, color=INK_2)


def figure_r2(roots: dict[int, Path], labels: np.ndarray, out: Path) -> None:
    """R^2 of every arm and seed above its rank-based version (the ranks of the latents replace the latents)."""
    r2, r2_rank = {}, {}
    for arm in ARMS:
        for seed in SEEDS:
            z = load_latent(run_dir(roots, arm, seed))
            r2[arm.name, seed] = between_state_r2(z, labels)
            r2_rank[arm.name, seed] = between_state_r2(rankdata(z[:, 0])[:, None], labels)
    positions = x_positions()
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax, values, title in (
        (axes[0], r2, "Share of the latent variance explained by the true state (between-state R²)"),
        (axes[1], r2_rank, "The same on the ranks of the latents (robust to outliers)"),
    ):
        draw_dots(ax, values, positions)
        ax.set_ylim(0.04, 0.30)
        ax.set_ylabel("R²")
        ax.set_title(title, loc="left", color=INK_2, fontsize=10)
    for group in dict.fromkeys(arm.group for arm in ARMS):
        xs = [x for arm, x in zip(ARMS, positions) if arm.group == group]
        axes[0].text(np.mean(xs), 1.14, group, transform=axes[0].get_xaxis_transform(), ha="center", va="bottom",
                     fontsize=10, color=INK, weight="bold")  # fmt: skip
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([arm.label for arm in ARMS], fontsize=9)
    axes[1].grid(axis="x", visible=False)
    axes[0].grid(axis="x", visible=False)
    handles = [
        plt.Line2D([], [], marker=m, ls="", color=GRAY, ms=8, label=f"seed {s}") for s, m in SEED_MARKERS.items()
    ]
    axes[0].legend(handles=handles, loc="upper left", ncol=3, fontsize=9, bbox_to_anchor=(0.0, 0.93))
    fig.suptitle("Latent separation of the eleven states, q = 1, 600 iterations (horizontal bars: mean of three seeds)",
                 x=0.07, ha="left", fontsize=12, y=1.02)  # fmt: skip
    fig.savefig(out / "r2_by_arm.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure_states(roots: dict[int, Path], labels: np.ndarray, out: Path) -> None:
    """The latent coordinate per true state (median, interquartile range), standardized and sign-aligned."""
    fig, axes = plt.subplots(len(KEY_ARMS), len(SEEDS), figsize=(11.5, 9.0), sharex=True, sharey=True)
    states = np.unique(labels)
    for row, arm in enumerate(KEY_ARMS):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            z = load_latent(run_dir(roots, arm, seed))[:, 0]
            r2 = between_state_r2(z[:, None], labels)
            z = z * np.sign(spearmanr(z, labels).statistic) / z.std()
            q1, med, q3 = (np.array([np.percentile(z[labels == s], p) for s in states]) for p in (25, 50, 75))
            ax.axhline(0, color=GRID, lw=1.0)
            ax.vlines(states, q1, q3, color=arm.color, lw=3.0, alpha=0.55)
            ax.plot(states, med, color=arm.color, lw=1.2)
            ax.plot(states, med, "o", color=arm.color, ms=6, mec=SURFACE, mew=1.2)
            ax.text(0.03, 0.95, f"R² {r2:.3f}", transform=ax.transAxes, va="top", fontsize=10, color=INK)
            ax.set_ylim(-1.9, 1.9)
            ax.grid(axis="x", visible=False)
            if row == 0:
                ax.set_title(f"seed {seed}", loc="left", color=INK_2)
            if col == 0:
                ax.set_ylabel(KEY_NAMES[arm.name], color=INK, fontsize=10)
    for ax in axes[-1]:
        ax.set_xticks(states)
        ax.set_xlabel("true state")
    fig.suptitle("Latent coordinate of each true state (dot: median, bar: interquartile range)\n"
                 "latents standardized to unit spread and signed so that the trend with the state is upward",
                 x=0.07, ha="left", fontsize=11, y=0.995)  # fmt: skip
    fig.savefig(out / "latents_by_state.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def figure_distribution(roots: dict[int, Path], out: Path) -> None:
    """The distribution of the standardized latent coordinate on a log density axis, against a normal one."""
    seeds = (11, 13)
    fig, axes = plt.subplots(1, len(seeds), figsize=(11.5, 4.6), sharey=True)
    edges = np.linspace(-8, 8, 81)
    centers = (edges[1:] + edges[:-1]) / 2
    for ax, seed in zip(axes, seeds):
        ax.plot(centers, norm.pdf(centers), color=GRAY, lw=1.2, label="normal")
        for arm in KEY_ARMS:
            z = load_latent(run_dir(roots, arm, seed))[:, 0]
            density, _ = np.histogram((z - z.mean()) / z.std(), bins=edges, density=True)
            ax.plot(centers, np.where(density > 0, density, np.nan), color=arm.color, lw=1.8,
                    label=f"{KEY_NAMES[arm.name]}  (kurt. {kurtosis(z):.1f})")  # fmt: skip
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 30.0)  # headroom above the peak for the legend
        ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0])  # no tick label under the legend
        ax.set_title(f"seed {seed}", loc="left", color=INK_2)
        ax.set_xlabel("latent coordinate, in standard deviations")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", ncol=2, fontsize=8.5, columnspacing=1.2)
    axes[0].set_ylabel("density")
    fig.suptitle("A PC present from the start gives a heavy-tailed latent; one that joins late does not\n"
                 "kurt.: excess kurtosis of the latent coordinate (0 for a normal distribution)",
                 x=0.07, ha="left", fontsize=11, y=1.04)  # fmt: skip
    fig.savefig(out / "latent_distribution.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def read_trajectory(directory: Path) -> dict[str, np.ndarray]:
    """Per-iteration pose entropy, loading power and latent posterior variance of a run's iterations.jsonl."""
    rows = last_trajectory([json.loads(line) for line in open(directory / "iterations.jsonl")])

    def mean_of(row: dict, key: str) -> float:
        values = [p[key] for p in row["posterior"] if p.get(key) is not None]
        return float(np.mean(values)) if values else np.nan

    loading = np.array([r["loading_power"] for r in rows])
    variance = np.array([mean_of(r, "latent_covariance_trace_mean") for r in rows])
    return {
        "iteration": np.array([r["iteration"] for r in rows]),
        "entropy": np.array([mean_of(r, "pose_entropy_mean") for r in rows]),
        "loading": loading,
        "variance": np.where(loading > 0, variance, np.nan),  # no latent exists before the PC joins
    }


def figure_dynamics(roots: dict[int, Path], out: Path) -> None:
    """Pose entropy, loading power and latent posterior variance against the iteration, for two seeds."""
    seeds = (11, 13)
    panels = (("entropy", "pose entropy (mean over images)", False), ("loading", "loading power  Σ|W|²", True),
              ("variance", "latent posterior variance (prior: 1)", False))  # fmt: skip
    fig, axes = plt.subplots(len(seeds), len(panels), figsize=(13.5, 7.4), sharex=True)
    for row, seed in enumerate(seeds):
        runs = {arm.name: read_trajectory(run_dir(roots, arm, seed)) for arm in KEY_ARMS}
        for col, (key, title, log) in enumerate(panels):
            ax = axes[row, col]
            for arm in KEY_ARMS:
                data = runs[arm.name]
                y = np.where(data[key] > 0, data[key], np.nan) if log else data[key]
                ax.plot(data["iteration"], y, color=arm.color, lw=1.8, label=KEY_NAMES[arm.name])
            for x, name in ((200, "radius 8"), (400, "radius 16")):
                ax.axvline(x, color=GRID, lw=1.0, zorder=0)
                if row == 1 and col == 0:
                    ax.text(x + 4, 0.03, name, transform=ax.get_xaxis_transform(), va="bottom", fontsize=8.5,
                            color=INK_2)  # fmt: skip
            ax.axvline(480, color=GRAY, lw=1.0, zorder=0)
            if log:
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(title, loc="left", color=INK_2)
            if col == 0:
                ax.set_ylabel(f"seed {seed}", color=INK)
            ax.grid(axis="x", visible=False)
    for ax in axes[-1]:
        ax.set_xlabel("iteration")
    axes[0, 1].text(486, 0.96, "PC joins", transform=axes[0, 1].get_xaxis_transform(), va="top", fontsize=8.5,
                    color=INK_2)  # fmt: skip
    handles = [plt.Line2D([], [], color=arm.color, lw=2.2, label=KEY_NAMES[arm.name]) for arm in KEY_ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("How the four key arms train (vertical lines: radius changes at 200 and 400, the PC at 480)\n"
                 "order-4 arms have 8 times more poses: their entropy is higher by ln 8 = 2.08 at equal sharpness",
                 x=0.07, ha="left", fontsize=11, y=0.995)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out / "training_dynamics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Draw the three figures into the output directory."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round1", type=Path, required=True, help="output_root of the first Snakemake round")
    parser.add_argument("--round2", type=Path, required=True, help="output_root of the second Snakemake round")
    parser.add_argument("--truth", type=Path, required=True, help="the fixture's evaluation/truth.npz (labels)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    roots = {1: args.round1, 2: args.round2}
    with np.load(args.truth, allow_pickle=False) as saved:
        labels = saved["labels"]
    style()
    figure_r2(roots, labels, args.out)
    figure_states(roots, labels, args.out)
    figure_distribution(roots, args.out)
    figure_dynamics(roots, args.out)
    print(f"wrote {sorted(p.name for p in args.out.glob('*.png'))} to {args.out}")


if __name__ == "__main__":
    main()
