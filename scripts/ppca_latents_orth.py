"""Orthonormal-PC latents (and a UMAP) for an existing export, on CPU, without a new pose pass.

    python scripts/ppca_latents_orth.py CHECKPOINT.npz EXPORT.npz [--out OUT.npz] [--umap UMAP.png]

EXPORT.npz is any file with the checkpoint's per-particle ``z`` (P, q): scripts/ppca_tilt_export.py's export or the
run's embeddings.npz. Writes OUT.npz (default EXPORT_orth.npz) with ``z_orth``, ``pc_scales`` and ``rotation``
(relax.ppca_initial_model.latents.orthonormal_latents) and, with --umap, ``umap`` (P, 2) plus a scatter plot.
UMAP settings: n_neighbors 30, min_dist 0, random_state 0, on the centred ``z_orth`` (no per-axis standardization).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from relax.ppca_initial_model.latents import box_from_theta, orthonormal_latents  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("export", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--umap", type=Path, help="also embed z_orth with UMAP and save a scatter plot here")
    a = parser.parse_args(argv)
    with np.load(a.checkpoint, allow_pickle=False) as arrays:
        theta = arrays["theta"]
    with np.load(a.export, allow_pickle=False) as arrays:
        z = arrays["z"]
    z_orth, scales, rotation = orthonormal_latents(theta, z, box_from_theta(theta))
    print("PC scales relative to the first:", np.round(scales / scales[0], 3), flush=True)
    out = {"z_orth": z_orth, "pc_scales": scales, "rotation": rotation}
    if a.umap is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap

        emb = umap.UMAP(n_neighbors=30, min_dist=0.0, random_state=0).fit_transform(z_orth - z_orth.mean(0))
        out["umap"] = emb
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(emb[:, 0], emb[:, 1], s=0.3, c="#3a5a80", rasterized=True)
        ax.set_title(f"UMAP of orthonormal-PC latents ({len(emb)} particles)")
        ax.set_xticks([]), ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(a.umap, dpi=120)
    path = a.out or a.export.with_name(a.export.stem + "_orth.npz")
    np.savez(path, **out)
    print(f"wrote {path}" + (f" and {a.umap}" if a.umap else ""), flush=True)


if __name__ == "__main__":
    main()
