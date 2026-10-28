"""Mean and first principal component of the true state maps of a PPCA fixture, projected along the three axes.

The ground-truth counterpart of the volume plot of a PPCA run: the PCA of the state maps weighted by their particle
counts, in the fixture's own frame (a run's maps sit in an arbitrary frame of their own, so compare shapes and signs,
not orientations). Protein positive; the PC is signed so that the last state has a positive coordinate.

Run (CPU, seconds):
  python scripts/plot_ppca_truth_pc.py FIXTURE/evaluation/truth.npz -o OUT.png [--radius 16]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from recovar.core import fourier_transform_utils as ftu


def state_maps(truth: Path, radius: float | None) -> tuple[np.ndarray, np.ndarray]:
    """The real-space state maps ``(S, n, n, n)`` and the share of the particles in each state.

    ``radius`` band-limits the centered Fourier volumes to that shell (a run's final Fourier radius); None keeps all.
    """
    with np.load(truth, allow_pickle=False) as saved:
        fourier, labels = saved["volumes_fourier"], saved["labels"]
    n = round(fourier.shape[1] ** (1 / 3))
    shell = np.sqrt(((np.indices((n, n, n)) - n // 2) ** 2).sum(axis=0))
    keep = np.ones((n, n, n)) if radius is None else (shell <= radius).astype(float)
    maps = np.stack([np.real(np.asarray(ftu.get_idft3(f.reshape(n, n, n) * keep))) for f in fourier]).astype(np.float64)
    counts = np.bincount(labels, minlength=len(maps)).astype(np.float64)
    return maps, counts / counts.sum()


def first_pc(maps: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The weighted mean map, the unit-norm first PC of the state maps and each state's coordinate along it."""
    flat = maps.reshape(len(maps), -1)
    mean = weights @ flat
    centered = (flat - mean) * np.sqrt(weights)[:, None]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc = vt[0]
    coords = (flat - mean) @ pc
    sign = 1.0 if coords[-1] >= 0 else -1.0
    return mean.reshape(maps.shape[1:]), (sign * pc).reshape(maps.shape[1:]), sign * coords


def variance_shares(maps: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The share of the weighted variance of the state maps that lies along each PC, largest first."""
    flat = maps.reshape(len(maps), -1)
    centered = (flat - weights @ flat) * np.sqrt(weights)[:, None]
    values = np.linalg.svd(centered, compute_uv=False) ** 2
    return values / values.sum()


def main() -> None:
    """Draw the mean and first PC projections and print the state coordinates."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("truth", type=Path, help="the fixture's evaluation/truth.npz")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--radius", type=float, help="band-limit the state maps to this Fourier shell (default: none)")
    args = parser.parse_args()
    maps, weights = state_maps(args.truth, args.radius)
    mean, pc, coords = first_pc(maps, weights)
    grid = np.indices(mean.shape) - mean.shape[0] // 2
    inside = np.sqrt((grid**2).sum(axis=0)) < mean.shape[0] // 3
    sign = 1.0 if np.sum(mean[inside] ** 3) >= 0 else -1.0
    mean, pc = sign * mean, sign * pc
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2), layout="constrained")
    for row, (volume, name, cmap) in enumerate(((mean, "mean", "gray"), (pc, "PC1", "RdBu_r"))):
        limit = np.abs(volume.sum(axis=0)).max()
        for col, axis in enumerate((0, 1, 2)):
            image = volume.sum(axis=axis)
            shown = axes[row, col].imshow(image, cmap=cmap, vmin=-limit if row else None, vmax=limit)
            axes[row, col].set(xticks=[], yticks=[], title=f"{name}, sum along {'xyz'[axis]}")
        fig.colorbar(shown, ax=axes[row, :], shrink=0.8)
    limit_note = "" if args.radius is None else f", band-limited to Fourier radius {args.radius:g}"
    fig.suptitle(f"Ground truth: mean and first PC of the 11 state maps{limit_note} (fixture frame; protein positive)",
                 fontsize=10)  # fmt: skip
    fig.savefig(args.out, dpi=150)
    print("state coordinates along PC1:", np.round(coords / np.abs(coords).max(), 2).tolist())
    shares = variance_shares(maps, weights)
    print(
        "share of the state variance on PC1..PC4:",
        np.round(shares[:4], 3).tolist(),
        "cumulative",
        np.round(np.cumsum(shares[:4]), 3).tolist(),
    )


if __name__ == "__main__":
    main()
