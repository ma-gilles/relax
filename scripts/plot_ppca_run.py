"""Top-PC scatter plots, UMAP, and mean / eigenvolume projections of a `relax ppca_initial_model` run.

Reads the run's last checkpoint (mean and loadings) and embeddings.npz (pose-marginal posterior means of z),
then writes to <run>/plots (or -o). A run that is still going has no embeddings.npz (it is written after the last
iteration), so only the figures that need the checkpoint and iterations.jsonl are written, from the latest
checkpoint (see "Unfinished run"):
  pc_scatter.png, pc_density.png   pairs of the top PCs, scatter and hexbin
  pc_histogram.png                 with --n-pcs 1 (or a one-dimensional model) instead of the two pair plots:
                                   histogram of the first coordinate
  umap.png, umap_embedding.npy     UMAP of all coordinates (skip with --skip-umap; skipped for one dimension)
  volume_projections.png           projections of the mean and the top PCs along the three axes
  latents_pc_ordered.npz           particle_ids, coordinates in the plotted basis, eigenvolume norms and coordinate stds
  mrc/mean.mrc, mrc/pc01.mrc ...   with --save-mrc: the mean and all q PCs as float32 maps (see below)
  gates_spectrogram.png            with --gates: the VDAM shell gates over iterations (plot_ppca_gates.py, from
                                   <run>/iterations.jsonl; for a finished run its last run must end at the
                                   checkpoint's iteration)

Basis: by default the latents are centred (their mean moves into the mean map, mu + W zbar) and the PCs are
those of the particle volumes mu + W z_i: unit-norm eigenvolumes ordered by variance, with uncorrelated
coordinates in volume units (as in RECOVAR). --raw keeps the learned loadings, latents and mean as saved.
Volumes are shown with protein positive (the sign is chosen from the mean map) and the same sign on every
channel, so a positive coordinate adds the PC.

MRC export (--save-mrc): the mean and every PC of the plotted basis (not only --n-pcs), written by
relax.io.map_io.write_map, so they are in RELION's map convention like every relax map (relax's internal
frame is the negated (2, 1, 0) transpose of the file array). The same linear conversion applies to all maps,
so mean + c * PC holds in the files with c the plotted coordinate. The files keep relax's sign even when the
plots flip it to show protein positive: a protein-dark training stack gives protein-negative maps (a note is
printed). The default PCs have unit norm; the coordinate stds are in latents_pc_ordered.npz. The pixel size
is read from the run's particle STAR (--voxel-size overrides it).

Unfinished run: the latest checkpoint is not the final iteration, so there are no latents. Skipped, with a message
saying why: pc_scatter / pc_density / pc_histogram, umap, latents_pc_ordered.npz and --labels. Still written:
volume_projections.png (title says the iteration), the MRC maps, and the gates (every complete iteration logged so
far, which can be past the checkpoint; a half-written last line of iterations.jsonl is ignored). Without latents
the default basis is the SVD of the loadings, assuming the prior z ~ N(0, I): unit-norm PCs ordered by singular
value (the row label shows it, the std of the coordinate under that prior), each signed so its largest voxel is
positive (the sign is otherwise arbitrary). --raw shows the learned columns instead. The same applies to a
finished run whose embeddings.npz is missing.

Run (CPU only, about a minute with UMAP; from a relax environment):
  python scripts/plot_ppca_run.py RUN_DIR \
      [--labels FIXTURE/evaluation/truth.npz] [--n-pcs 4] [--skip-umap] [--raw] \
      [--save-mrc [--voxel-size A]] [--gates] [-o OUT_DIR]
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # a plotting script has no business on a GPU

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from recovar.core import fourier_transform_utils as ftu

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_ppca_gates import plot_run_gates  # noqa: E402


def latest_checkpoint(run: Path) -> Path:
    """The highest-numbered checkpoint_NNNN.npz of ``run``."""
    candidates = sorted(run.glob("checkpoint_[0-9]*.npz"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_*.npz in {run}")
    return candidates[-1]


def load_model(run: Path) -> tuple[np.ndarray, dict]:
    """Return ``theta`` (half-Fourier voxels, q+1) and the metadata of the latest checkpoint of ``run``."""
    checkpoint = latest_checkpoint(run)
    with np.load(checkpoint, allow_pickle=False) as saved:
        return saved["theta"].copy(), json.loads(str(saved["metadata"]))


def is_final(meta: dict) -> bool:
    """Whether the checkpoint with metadata ``meta`` is the last iteration of its run."""
    return meta["iteration"] == meta["config"]["iterations"]


def latents_unavailable(run: Path, meta: dict) -> str | None:
    """Why the final latents of ``run`` cannot be used with its latest checkpoint, or None if they can.

    embeddings.npz holds the final model's latents, so it matches the checkpoint only when that is the final one.
    """
    if not is_final(meta):
        return (
            f"the run is at iteration {meta['iteration']} of {meta['config']['iterations']}, and embeddings.npz "
            "holds the final model's latents"
        )
    if not (run / "embeddings.npz").exists():
        if meta["config"].get("skip_final_embeddings"):
            return "the run was started with skip_final_embeddings, so it has no embeddings.npz"
        return "embeddings.npz is not there: the final embedding pass is still running, or it failed"
    return None


def load_latents(run: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``z`` (N, q) in particle-ID order and the IDs from embeddings.npz of a finished run."""
    with np.load(run / "embeddings.npz", allow_pickle=False) as saved:
        ids, z = saved["particle_ids"].copy(), saved["z"].copy()
    if not np.array_equal(np.sort(ids), np.arange(len(ids))):
        raise ValueError("embeddings.npz must hold every particle exactly once")
    order = np.argsort(ids)
    return z[order], ids[order]


def box_size(n_voxels: int) -> int:
    """The box ``n`` of a half-Fourier volume with ``n * n * (n // 2 + 1)`` voxels."""
    n = round((2 * n_voxels) ** (1 / 3))
    for candidate in range(max(n - 2, 2), n + 3):
        if candidate * candidate * (candidate // 2 + 1) == n_voxels:
            return candidate
    raise ValueError(f"{n_voxels} is not a half-Fourier volume size")


def to_real(theta: np.ndarray, n: int) -> np.ndarray:
    """Real-space volumes ``(q+1, n, n, n)`` of the half-Fourier columns of ``theta``."""
    half = (theta.shape[1], n, n, n // 2 + 1)
    return np.asarray(ftu.get_idft3_real(theta.T.reshape(half), (n, n, n)), np.float32)


def voxel_size_of(run: Path, override: float | None) -> float:
    """Pixel size in Angstrom: ``override``, else the optics table of the particle STAR named in run.json."""
    if override is not None:
        return override
    star = json.loads((run / "run.json").read_text())["input"]["particles"]
    if star is None:
        raise ValueError("A subtomogram run names no particle STAR; pass --voxel-size")
    from recovar.data_io.starfile import read_star, star_column

    _, optics = read_star(star)
    if optics is None or len(optics) != 1:
        raise ValueError(f"{star} needs exactly one optics group; pass --voxel-size")
    return float(star_column(optics, "rlnImagePixelSize", required=True).iloc[0])


def save_mrcs(mean: np.ndarray, pcs: np.ndarray, voxel_size: float, directory: Path) -> None:
    """Write the mean and every PC as RELION-convention maps (mean.mrc, pc01.mrc, ...) into ``directory``."""
    from relax.io.map_io import write_map

    directory.mkdir(parents=True, exist_ok=True)
    write_map(directory / "mean.mrc", mean, voxel_size)
    for k, pc in enumerate(pcs, 1):
        write_map(directory / f"pc{k:02d}.mrc", pc, voxel_size)


def volume_sign(mean: np.ndarray) -> int:
    """+1 if the mean's cubed density inside a central sphere is positive (protein positive), else -1."""
    n = mean.shape[0]
    grid = np.indices(mean.shape) - n // 2
    inside = np.sqrt((grid**2).sum(axis=0)) < n // 3
    return 1 if np.sum(mean[inside].astype(np.float64) ** 3) >= 0 else -1


def apply_gauge(
    mean: np.ndarray, loadings: np.ndarray, z: np.ndarray, raw: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean map, eigenvolumes ``(q, n, n, n)`` and latents ``(N, q)`` to plot.

    Default: centre z (mu + W zbar), then take the PCA of the particle volumes mu + W z_i: unit-norm PCs in
    variance order, coordinates in volume units, each PC signed so its coordinate has positive skewness.
    ``raw`` returns the inputs unchanged.
    """
    if raw:
        return mean, loadings, z
    zbar = z.mean(axis=0)
    zc = z - zbar
    mean = mean + np.tensordot(zbar, loadings, axes=1)
    q = loadings.shape[0]
    a = loadings.reshape(q, -1).T.astype(np.float64)  # (voxels, q)
    # PCA of the particle volumes mu + W z_i: their covariance is W cov(z) W^T = B B^T
    lam, rot = np.linalg.eigh(np.atleast_2d(np.cov(zc, rowvar=False)))
    b = a @ (rot * np.sqrt(np.maximum(lam, 0)))
    u = np.linalg.svd(b, full_matrices=False)[0]
    coords = zc @ (a.T @ u)  # projection of each particle volume on the unit-norm eigenvolumes
    flip = np.where(np.mean(coords**3, axis=0) >= 0, 1.0, -1.0)
    pcs = (u * flip).T.reshape(loadings.shape).astype(np.float32)
    return mean, pcs, (coords * flip).astype(np.float32)


def loading_gauge(mean: np.ndarray, loadings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the mean map, eigenvolumes ``(q, n, n, n)`` and their singular values, for a model without latents.

    Under the prior z ~ N(0, I) the particle volumes have covariance W W^T, so the unit-norm PCs are the left
    singular vectors of W in singular-value order, and a coordinate's std is its singular value. Each PC is
    signed so its largest voxel is positive (the sign is otherwise arbitrary).
    """
    q = loadings.shape[0]
    u, singular_values, _ = np.linalg.svd(loadings.reshape(q, -1).T.astype(np.float64), full_matrices=False)
    flip = np.where(u[np.argmax(np.abs(u), axis=0), np.arange(q)] >= 0, 1.0, -1.0)
    return mean, (u * flip).T.reshape(loadings.shape).astype(np.float32), singular_values


def label_colors(labels: np.ndarray | None) -> tuple[np.ndarray | None, dict]:
    """Per-particle RGBA colours and a ``{label: colour}`` legend map; ``(None, {})`` without labels."""
    if labels is None:
        return None, {}
    values = np.unique(labels)
    # Qualitative colours for up to 10 states, a continuous map beyond
    if len(values) <= 10:
        palette = [plt.get_cmap("tab10")(i) for i in range(len(values))]
    else:
        palette = list(plt.get_cmap("turbo")(np.linspace(0.05, 0.95, len(values))))
    lookup = {int(v): palette[i] for i, v in enumerate(values)}
    return np.array([lookup[int(v)] for v in labels]), lookup


def draw_points(ax, xy: np.ndarray, colors: np.ndarray | None, density: bool) -> None:
    """Hexbin density (``density``) or scatter of ``xy``, coloured per particle when ``colors`` is given."""
    if density:
        ax.hexbin(xy[:, 0], xy[:, 1], gridsize=80, bins="log", mincnt=1, cmap="viridis")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c=colors if colors is not None else "#333333",
                   alpha=0.35, linewidths=0, rasterized=True)
    for axis, values in zip(("x", "y"), xy.T):
        low, high = np.percentile(values, [0.2, 99.8])
        pad = 0.1 * (high - low)
        (ax.set_xlim if axis == "x" else ax.set_ylim)(low - pad, high + pad)


def add_legend(fig, lookup: dict) -> None:
    """Legend of the true-state colours to the right of ``fig``."""
    if lookup:
        handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f"state {k}") for k, c in lookup.items()]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)


def plot_pc_pairs(
    z: np.ndarray, stds: np.ndarray, n_pcs: int, path: Path, labels: np.ndarray | None, density: bool
) -> None:
    """Grid of latent scatters (or hexbins) over all pairs of the top ``n_pcs`` PCs."""
    pairs = list(itertools.combinations(range(n_pcs), 2))
    cols = min(3, len(pairs))
    rows = -(-len(pairs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False, layout="constrained")
    colors, lookup = label_colors(None if density else labels)
    for ax, (a, b) in zip(axes.flat, pairs):
        draw_points(ax, z[:, [a, b]], colors, density)
        ax.set(xlabel=f"PC{a + 1} coordinate  (std {stds[a]:.2e})", ylabel=f"PC{b + 1} coordinate  (std {stds[b]:.2e})")
        ax.axvline(0, color="gray", lw=0.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))  # offset text, not long tick labels
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    kind = "density" if density else ("coloured by true state" if labels is not None else "scatter")
    fig.suptitle(f"{len(z):,} particles, posterior-mean coordinates, top {n_pcs} PCs ({kind})")
    add_legend(fig, lookup)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pc_histogram(coords: np.ndarray, std: float, path: Path, labels: np.ndarray | None) -> None:
    """Histogram of a 1-D coordinate: all particles, and with ``labels`` one outline per true state.

    The bins span the 0.2-99.8 percentile range, shared by both panels.
    """
    panels = 1 if labels is None else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.6 * panels, 4.2), squeeze=False, layout="constrained")
    bins = np.linspace(*np.percentile(coords, [0.2, 99.8]), 61)
    axes[0, 0].hist(coords, bins=bins, color="#333333")
    axes[0, 0].set_title("all particles")
    colors, lookup = label_colors(labels)
    if labels is not None:
        for state, color in lookup.items():
            axes[0, 1].hist(coords[labels == state], bins=bins, histtype="step", color=color, lw=1.3)
        axes[0, 1].set_title("by true state")
    for ax in axes.flat:
        ax.axvline(0, color="gray", lw=0.5)
        ax.set(xlabel=f"PC1 coordinate  (std {std:.2e})", ylabel="particles")
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig.suptitle(f"{len(coords):,} particles, posterior-mean coordinate along PC1")
    add_legend(fig, lookup)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_umap(z: np.ndarray, labels: np.ndarray | None, path: Path) -> np.ndarray:
    """UMAP of all latent dimensions (same seed as recovar analyze); saves scatter and hexbin, returns it."""
    import umap

    embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(z)
    colors, lookup = label_colors(labels)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    draw_points(axes[0], embedding, colors, density=False)
    draw_points(axes[1], embedding, None, density=True)
    axes[0].set_title("coloured by true state" if labels is not None else "scatter")
    axes[1].set_title("density")
    for ax in axes:
        ax.set(xlabel="UMAP1", ylabel="UMAP2", xticks=[], yticks=[])
    fig.suptitle(f"UMAP of {z.shape[1]}-dimensional latents, {len(z):,} particles")
    add_legend(fig, lookup)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return embedding


def plot_volume_projections(
    mean: np.ndarray, pcs: np.ndarray, stds: np.ndarray | None, std_label: str, sign: int, meta: dict, path: Path
) -> None:
    """Projections (sums along each axis) of the mean (grey) and the PCs (signed, symmetric limits per row).

    Each PC row is labelled with its norm and, unless ``stds`` is None, ``std_label`` and its value in ``stds``.
    The title names the iteration of an unfinished run.
    """
    volumes = [mean, *pcs]
    fig, axes = plt.subplots(len(volumes), 3, figsize=(8, 2.5 * len(volumes)), squeeze=False, layout="constrained")
    for row, volume in enumerate(volumes):
        views = [sign * volume.sum(axis=axis) for axis in range(3)]
        if row == 0:
            low, high = np.percentile(np.concatenate([v.ravel() for v in views]), [0.2, 99.8])
            cmap = "gray"
        else:
            high = max(float(np.percentile(np.abs(np.concatenate([v.ravel() for v in views])), 99.8)), 1e-30)
            low, cmap = -high, "RdBu_r"
        for axis, view in enumerate(views):
            image = axes[row, axis].imshow(view, cmap=cmap, vmin=low, vmax=high)
            axes[row, axis].set(xticks=[], yticks=[])
            if row == 0:
                axes[row, axis].set_title(f"sum along map {'xyz'[axis]}")  # internal axis a is the file's axis 2 - a
        fig.colorbar(image, ax=axes[row].tolist(), shrink=0.8, format="%.1e")
        name = "mean" if row == 0 else f"PC{row}"
        text = f"{name}\nnorm {np.linalg.norm(volume):.2e}"
        if row > 0 and stds is not None:
            text += f"\n{std_label} {stds[row - 1]:.2e}"
        axes[row, 0].set_ylabel(text, rotation=0, ha="right", va="center")
    if is_final(meta):
        radius = f"final Fourier radius {meta['radius']}"
    else:
        radius = f"Fourier radius {meta['radius']} at iteration {meta['iteration']} of {meta['config']['iterations']} (unfinished)"
    fig.suptitle(
        f"Mean and eigenvolume projections | box {mean.shape[0]}, {radius} | "
        "protein positive; a positive latent adds the PC; each row has its own scale"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Parse the command line and write the figures."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory with checkpoint_*.npz (and embeddings.npz once finished)")
    parser.add_argument("-o", "--output", type=Path, help="output directory (default: <run>/plots)")
    parser.add_argument(
        "--n-pcs", type=int, default=4,
        help="top PCs for the pair plots and volume rows (default 4); 1 gives a histogram instead of pair plots",
    )
    parser.add_argument("--labels", type=Path, help="npz with a 'labels' array indexed by particle ID (evaluation only)")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--raw", action="store_true", help="keep the learned basis and mean (no centring or ordering)")
    parser.add_argument("--save-mrc", action="store_true", help="also write the mean and all PCs as MRC maps to <output>/mrc")
    parser.add_argument("--voxel-size", type=float, help="pixel size in Angstrom for the MRC headers (default: from the run's STAR)")
    parser.add_argument("--gates", action="store_true", help="also plot the VDAM shell gates from <run>/iterations.jsonl")
    args = parser.parse_args()
    voxel_size = voxel_size_of(args.run, args.voxel_size) if args.save_mrc else None

    theta, meta = load_model(args.run)
    q = theta.shape[1] - 1
    n_pcs = min(args.n_pcs, q)
    if n_pcs < 1:
        raise ValueError("--n-pcs must be at least 1")
    skipped = latents_unavailable(args.run, meta)
    z = ids = labels = None
    if skipped is None:
        z, ids = load_latents(args.run)
        if args.labels is not None:
            with np.load(args.labels, allow_pickle=False) as saved:
                labels = saved["labels"][ids]
    else:
        print(f"No latents ({skipped}): skipping the latent plots, umap, latents_pc_ordered.npz and --labels")
    output = args.output or args.run / "plots"
    output.mkdir(parents=True, exist_ok=True)
    if args.gates:
        plot_run_gates(
            args.run / "iterations.jsonl", output / "gates_spectrogram.png",
            last_iteration=meta["iteration"] if is_final(meta) else None, in_progress=not is_final(meta),
        )

    volumes = to_real(theta, box_size(theta.shape[0]))
    if z is not None:
        mean, pcs, z_plot = apply_gauge(volumes[0], volumes[1:], z, args.raw)
        norms, stds, std_label = np.linalg.norm(pcs.reshape(q, -1), axis=1), z_plot.std(axis=0), "coord. std"
        np.savez(
            output / "latents_pc_ordered.npz", particle_ids=ids, z=z_plot, eigenvolume_norms=norms, coordinate_stds=stds
        )
    elif args.raw:
        mean, pcs, stds, std_label = volumes[0], volumes[1:], None, ""
    else:
        mean, pcs, stds = loading_gauge(volumes[0], volumes[1:])
        std_label = "prior std"

    if z is None:
        pass
    elif n_pcs == 1:
        plot_pc_histogram(z_plot[:, 0], stds[0], output / "pc_histogram.png", labels)
    else:
        plot_pc_pairs(z_plot, stds, n_pcs, output / "pc_scatter.png", labels, density=False)
        plot_pc_pairs(z_plot, stds, n_pcs, output / "pc_density.png", labels, density=True)
    sign = volume_sign(mean)
    plot_volume_projections(
        mean, pcs[:n_pcs], None if stds is None else stds[:n_pcs], std_label, sign, meta, output / "volume_projections.png"
    )
    if args.save_mrc:
        save_mrcs(mean, pcs, voxel_size, output / "mrc")
        if sign != -1:  # relax's map convention negates the internal frame
            print(
                "Note: the mean is positive in relax's internal frame, so the MRC files (RELION convention, "
                "negated) are protein-negative, like the input images; the plots show protein positive"
            )
    if z is None or args.skip_umap:
        pass
    elif z_plot.shape[1] < 2:
        print("UMAP skipped: the model has one latent dimension")
    else:
        np.save(output / "umap_embedding.npy", plot_umap(z_plot, labels, output / "umap.png"))
    print(f"Wrote figures to {output} (checkpoint iteration {meta['iteration']} of {meta['config']['iterations']})")


if __name__ == "__main__":
    main()
