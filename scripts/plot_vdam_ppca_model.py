"""Display the learned PPCA mean, signed loadings and particle coordinates.

No spatial or latent alignment is applied. See
docs/development/vdam_ppca_visual_comparison.md. Labels are evaluation-only.
"""

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from recovar.core import fourier_transform_utils as ftu
from recovar.utils.helpers import write_mrc

from relax.ppca_initial_model.checkpoint import file_hash


def particle_labels(ids, coordinates, labels):
    """Join evaluation labels by particle ID, never by embedding row order."""
    ids, coordinates, labels = np.asarray(ids), np.asarray(coordinates), np.asarray(labels)
    if coordinates.ndim != 2 or coordinates.shape[0] != len(ids) or coordinates.shape[1] < 1:
        raise ValueError("Expected one nonempty coordinate vector per particle ID")
    if not np.isfinite(coordinates).all():
        raise ValueError("Nonfinite latent coordinates")
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or labels.ndim != 1:
        raise ValueError("Expected integer particle IDs and a label vector")
    if len(ids) != len(labels) or not np.array_equal(np.sort(ids), np.arange(len(labels))):
        raise ValueError("Expected all particles exactly once in the final embeddings")
    return labels[ids]


def save_model_diagnostics(theta, ids, coordinates, labels, *, n, voxel, radius, output, pdf=None):
    """Export fields and matching latent axes without changing their gauge."""
    output = Path(output)
    coordinates = np.asarray(coordinates)
    colors_by_particle = particle_labels(ids, coordinates, labels)
    q = coordinates.shape[1]
    if theta.shape != (n * n * (n // 2 + 1), q + 1) or not np.isfinite(theta).all():
        raise ValueError("Checkpoint geometry/latent dimension mismatch")
    fields = np.asarray(ftu.get_idft3_real(theta.T.reshape(q + 1, n, n, n // 2 + 1), (n, n, n)))
    # The checkpoint already carries the controller's final support/mask.
    # Show its actual fields: no extra fit, sign change, SVD or normalization.
    views = [[v.sum(axis=a) for a in range(3)] + [np.take(v, n // 2, axis=a) for a in range(3)] for v in fields]
    fig = plt.figure(figsize=(17.5, 2.7 * (q + 1)))
    # Reserve a tick-label gutter after each colorbar so labels cannot cover a map.
    grid = fig.add_gridspec(q + 1, 10, width_ratios=[1, 1, 1, 0.055, 0.5, 1, 1, 1, 0.055, 0.5],
                            left=0.12, right=0.98, bottom=0.11, top=0.91, wspace=0.08, hspace=0.18)
    axes = np.array([[fig.add_subplot(grid[row, col + 2 * (col >= 3)]) for col in range(6)]
                     for row in range(q + 1)])
    limits = {}
    for group, columns in (("projections", range(3)), ("slices", range(3, 6))):
        mean_values = np.concatenate([views[0][c].ravel() for c in columns])
        mean_low, mean_high = np.percentile(mean_values, [0.2, 99.8])
        pc_values = np.concatenate([views[k][c].ravel() for k in range(1, q + 1) for c in columns])
        bound = max(float(np.percentile(np.abs(pc_values), 99.8)), np.finfo(np.float32).tiny)
        limits[group] = {"mean": [float(mean_low), float(mean_high)], "pcs": [-bound, bound]}
        for row in range(q + 1):
            for col in columns:
                low, high = (mean_low, mean_high) if row == 0 else (-bound, bound)
                im = axes[row, col].imshow(views[row][col], origin="upper", cmap="gray" if row == 0 else "RdBu_r", vmin=low, vmax=high)
                axes[row, col].set(xticks=[], yticks=[])
                if row == 0:
                    axes[row, col].set_title(f"Projection {col}" if col < 3 else f"Central slice {col - 3}")
            fig.colorbar(im, cax=fig.add_subplot(grid[row, 3 if group == "projections" else 8]), format="%.1e")
    for row in range(q + 1):
        label = "Mean" if row == 0 else f"PC{row - 1}\nlearned loading"
        axes[row, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=14)
        write_mrc(str(output / ("mean_unaligned.mrc" if row == 0 else f"pc{row - 1}_unaligned.mrc")), fields[row], voxel)
    fig.suptitle("Learned PPCA mean and signed PCs | original spatial frame and latent basis", fontsize=14)
    fig.text(0.5, 0.025, f"Box {n}, pixel {voxel:g} Å; final radius {radius} ({n * voxel / radius:.0f} Å). No alignment, sign changes or amplitude normalization.\n"
             "Red = positive, blue = negative loading; a positive latent coordinate adds the displayed PC. All PCs share scales within projections / slices.", ha="center", fontsize=9)
    fig.savefig(output / "model_components_six_views.png", dpi=150)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)

    pairs = list(itertools.combinations(range(q), 2)) if q > 1 else [(0, None)]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(12, 4.8 * len(pairs)), squeeze=False)
    centers = {str(k): coordinates[colors_by_particle == k].mean(axis=0).tolist() for k in np.unique(labels)}
    for row, (a, b) in enumerate(pairs):
        for col, ax in enumerate(axes[row]):
            groups = [None] if col == 0 else np.unique(colors_by_particle)
            for index, group in enumerate(groups):
                chosen = np.ones(len(ids), dtype=bool) if group is None else colors_by_particle == group
                color = "#333333" if group is None else plt.get_cmap("tab10")(index % 10)
                label = "All particles" if group is None else f"True state {group} (n={chosen.sum()})"
                if b is None:
                    ax.hist(coordinates[chosen, a], bins=40, color=color, alpha=0.45, label=label)
                else:
                    ax.scatter(coordinates[chosen, a], coordinates[chosen, b], s=10, c=[color], alpha=0.3 if col == 0 else 0.45, linewidths=0, rasterized=True, label=label)
                    if group is not None:
                        center = centers[str(group)]
                        ax.scatter(center[a], center[b], marker="X", s=110, c=[color], edgecolors="black", linewidths=0.8)
            ax.set_xlabel(f"PC{a} latent coordinate")
            ax.set_ylabel("Particles" if b is None else f"PC{b} latent coordinate")
            ax.set_title("Inferred coordinates; no labels" if col == 0 else "Same coordinates; evaluation-only true labels")
            ax.axvline(0, color="gray", lw=0.5)
            if b is not None:
                ax.axhline(0, color="gray", lw=0.5)
                ax.set_aspect("equal", adjustable="box")
            ax.legend(fontsize=8)
        axes[row, 1].set_xlim(axes[row, 0].get_xlim())
        axes[row, 1].set_ylim(axes[row, 0].get_ylim())
    fig.suptitle(f"All {len(ids):,} final pose-marginal posterior means | {q} latent dimensions", fontsize=14)
    fig.text(0.5, 0.012, "Native coordinates matching the PC maps; no PCA refit, whitening or alignment. X marks true-class coordinate means.\n"
             "Posterior means do not display per-particle posterior uncertainty. True labels are used only in the colored diagnostic.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.1 / len(pairs), 1, 0.95))
    fig.savefig(output / "latent_coordinates.png", dpi=160)
    if pdf is not None:
        pdf.savefig(fig)
    plt.close(fig)
    result = {"n_particles": len(ids), "q": q, "spatial_alignment": "none", "latent_transform": "none; raw learned loading basis and matching z", "pairs": pairs,
              "component_l2_norms": np.linalg.norm(fields.reshape(q + 1, -1), axis=1).tolist(),
              "display_limits": limits, "latent_covariance": np.atleast_2d(np.cov(coordinates, rowvar=False)).tolist(),
              "latent_quantiles": np.quantile(coordinates, [0, 0.01, 0.5, 0.99, 1], axis=0).tolist(),
              "true_class_coordinate_means": centers,
              "files": ["model_components_six_views.png", "latent_coordinates.png"],
              "interpretation": "Learned loadings need not be orthogonal variance-ordered PCs. Their axes correspond exactly to the plotted coordinates; labels never modify model or embeddings."}
    (output / "model_diagnostics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("embeddings")
    parser.add_argument("training_manifest")
    parser.add_argument("evaluation_dir")
    parser.add_argument("fixture_verification")
    parser.add_argument("--comparison-report", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.comparison_report).read_text())
    checks = {"checkpoint_sha256": args.checkpoint, "embeddings_sha256": args.embeddings,
              "training_manifest_sha256": args.training_manifest, "fixture_verification_sha256": args.fixture_verification}
    for key, path in checks.items():
        if file_hash(path) != report["input_sha256"][key]:
            raise ValueError(f"Comparison input mismatch: {key}")
    truth = Path(args.evaluation_dir) / "truth.npz"
    verification = json.loads(Path(args.fixture_verification).read_text())
    if file_hash(truth) != verification["files"]["evaluation/truth.npz"]:
        raise ValueError("Evaluation truth identity mismatch")
    with np.load(args.checkpoint, allow_pickle=False) as saved:
        theta, metadata = saved["theta"].copy(), json.loads(str(saved["metadata"]))
    if metadata["iteration"] != metadata["config"]["iterations"] or metadata["identity"]["manifest_sha256"] != file_hash(args.training_manifest):
        raise ValueError("Expected a final checkpoint for this training manifest")
    with np.load(args.embeddings, allow_pickle=False) as saved:
        ids, coordinates = saved["particle_ids"].copy(), saved["z"].copy()
    with np.load(truth, allow_pickle=False) as saved:
        labels = saved["labels"].copy()
    training = json.loads(Path(args.training_manifest).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with PdfPages(output / "model_and_latents.pdf") as pdf:
        save_model_diagnostics(theta, ids, coordinates, labels, n=int(training["box"]), voxel=float(training["voxel_size"]), radius=int(report["active_radius"]), output=output, pdf=pdf)
    (output / "input_identity.json").write_text(json.dumps({**{k: file_hash(v) for k, v in checks.items()}, "comparison_report_sha256": file_hash(args.comparison_report), "truth_sha256": file_hash(truth)}, indent=2) + "\n")
    (output / "README.md").write_text("# Learned model and particle coordinates\n\n[PDF](model_and_latents.pdf)\n\n![Mean and PCs](model_components_six_views.png)\n\n![Latent coordinates](latent_coordinates.png)\n")
    print(f"Saved model fields and latent coordinates to {output}")


if __name__ == "__main__":
    main()
