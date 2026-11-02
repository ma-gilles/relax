"""UMAP of the latent coordinates of a `relax ppca_initial_model` run, coloured by true state.

Reads PLOTS_DIR/latents_pc_ordered.npz, which plot_ppca_run.py writes: the final latents of every particle in the plotted
basis (centred, PCs of the particle volumes in variance order, coordinates in volume units) with their particle IDs.
With --labels it reads the true state of every particle (an npz with a `labels` array indexed by particle ID; evaluation
only) and colours the points by it. UMAP runs on all coordinates with the seed of `recovar analyze`. Writes to PLOTS_DIR:
  umap.png            scatter coloured by true state (each state's number at its median), and density of the same points
  umap_embedding.npy  the (N, 2) embedding in particle-ID order

Run (CPU, about a minute for 20,000 particles):
  python scripts/plot_ppca_umap.py RUN_DIR/plots --labels FIXTURE/evaluation/truth.npz
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

UMAP_SEED = 42


def load_latents(plots: Path, labels_path: Path | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Latents ``(N, q)`` and, with ``labels_path``, the true state of each particle, both in particle-ID order."""
    with np.load(plots / "latents_pc_ordered.npz", allow_pickle=False) as saved:
        ids, z = saved["particle_ids"], saved["z"]
    if not np.array_equal(np.sort(ids), np.arange(len(ids))):
        raise ValueError("latents_pc_ordered.npz must hold every particle exactly once")
    order = np.argsort(ids)
    if labels_path is None:
        return z[order], None
    with np.load(labels_path, allow_pickle=False) as saved:
        labels = saved["labels"]
    if len(labels) != len(ids):
        raise ValueError(f"{labels_path} has {len(labels)} labels for {len(ids)} particles")
    return z[order], labels[ids[order]]


def state_colors(labels: np.ndarray) -> dict[int, tuple]:
    """A colour per state: qualitative for up to 10 states, a continuous map beyond (as plot_ppca_run.py)."""
    states = np.unique(labels)
    if len(states) <= 10:
        palette = [plt.get_cmap("tab10")(i) for i in range(len(states))]
    else:
        palette = list(plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(states))))
    return {int(state): palette[i] for i, state in enumerate(states)}


def embed(z: np.ndarray, seed: int) -> np.ndarray:
    """UMAP embedding ``(N, 2)`` of the rows of ``z``."""
    import umap

    return umap.UMAP(n_components=2, random_state=seed).fit_transform(z)


def plot_umap(embedding: np.ndarray, labels: np.ndarray | None, n_dims: int, path: Path) -> None:
    """Scatter (coloured by ``labels`` when given) and hexbin density of ``embedding``."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    if labels is None:
        axes[0].scatter(*embedding.T, s=2, c="#333333", alpha=0.35, linewidths=0, rasterized=True)
        axes[0].set_title("scatter")
    else:
        colors = state_colors(labels)
        axes[0].scatter(
            *embedding.T, s=2, c=[colors[int(s)] for s in labels], alpha=0.35, linewidths=0, rasterized=True
        )
        for state, color in colors.items():
            x, y = np.median(embedding[labels == state], axis=0)
            axes[0].text(
                x,
                y,
                str(state),
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="black",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": color, "alpha": 0.85},
            )
        axes[0].set_title("coloured by true state (number at each state's median)")
    axes[1].hexbin(*embedding.T, gridsize=80, bins="log", mincnt=1, cmap="viridis")
    axes[1].set_title("density")
    for ax in axes:
        ax.set(xlabel="UMAP1", ylabel="UMAP2", xticks=[], yticks=[])
        for axis, values in zip(("x", "y"), embedding.T):
            low, high = np.percentile(values, [0.2, 99.8])
            pad = 0.1 * (high - low)
            (ax.set_xlim if axis == "x" else ax.set_ylim)(low - pad, high + pad)
    fig.suptitle(f"UMAP of {n_dims}-dimensional latents, {len(embedding):,} particles")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parse the command line, embed the latents and write umap.png and umap_embedding.npy."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plots", type=Path, help="directory with latents_pc_ordered.npz; the figures go there")
    parser.add_argument(
        "--labels", type=Path, help="npz with a 'labels' array indexed by particle ID (evaluation only)"
    )
    parser.add_argument("--seed", type=int, default=UMAP_SEED, help=f"UMAP random state (default {UMAP_SEED})")
    args = parser.parse_args()

    z, labels = load_latents(args.plots, args.labels)
    if z.shape[1] < 2:
        raise ValueError("UMAP needs at least two latent dimensions; use the PC histogram for q = 1")
    embedding = embed(z, args.seed)
    np.save(args.plots / "umap_embedding.npy", embedding)
    plot_umap(embedding, labels, z.shape[1], args.plots / "umap.png")
    print(f"Wrote umap.png and umap_embedding.npy to {args.plots}")


if __name__ == "__main__":
    main()
