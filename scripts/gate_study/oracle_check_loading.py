"""Known-truth check of the oracle's loading channel (gate study): one true loading reconstructed at true poses.

Two states S0, S1 of a simulated suite dataset make an exact rank-1 PPCA: mean m = (v0 + v1)/2, loading w = (v1 - v0)/2
and latent z = -1 for S0's particles, +1 for S1's (unit variance, as the model's standardised latent). With true
rotations and known z, the per-voxel PPCA normal equations are the 2 x 2 system

    [[F, G], [G, F]] [m; w] = [B0; B1],   F = sum CTF^2, G = sum z CTF^2, B0 = sum CTF y, B1 = sum z CTF y,

back-projected with linear interpolation (recovar adjoint_slice_volume, as scripts/gate_study/oracle_check.py; a per-shell
sigma^2 cancels). Solved per voxel in two halves; the loading's half-map FSC (unmasked and with the model's mask) gives
the measured full-data SNR 2 FSC / (1 - FSC), compared with the oracle's prediction n kappa snr_img P_w / P_probe
(oracle_gate's machinery; P_w the power of w), so the ratio is the loading channel's noise factor (the mean channel's is
about 3.0 unmasked and 5.6 masked).

Run: python scripts/gate_study/oracle_check_loading.py DATASET_DIR STAR --states S0 S1 [--max-images N] -o out.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp  # noqa: E402
from recovar.core import fourier_transform_utils as ftu  # noqa: E402
from recovar.utils.helpers import load_mrc  # noqa: E402

from relax.commands.ppca_initial_model import load_training  # noqa: E402
from relax.ppca_initial_model.initialization import support_mask  # noqa: E402
from scripts.gate_study.oracle_check import RADIUS, half_reconstructions, masked  # noqa: E402
from scripts.gate_study.oracle_gate import oracle_tables, shell_power_3d_full  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("star", type=Path)
    parser.add_argument("--states", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--q", type=int, default=10)
    parser.add_argument("--max-images", type=int, help="per state")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = args.dataset / "evaluation"
    with np.load(evaluation / "truth.npz") as truth:
        labels, rotations = truth["labels"], truth["rotations"]
    dataset, _, _ = load_training(str(args.star), "auto")
    shape = dataset.volume_shape
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(shape)).reshape(-1)
    nshells = int(shells.max()) + 1
    halves = []  # per state: [(B, F) half 0, (B, F) half 1]
    counts = []
    for s in args.states:
        ids = np.random.default_rng(s).permutation(np.flatnonzero(labels == s))[: args.max_images]
        counts.append(len(ids))
        halves.append(half_reconstructions(dataset, rotations, ids))
    solutions = []
    for h in range(2):
        (b0, f0), (b1, f1) = halves[0][h], halves[1][h]
        F, G, B0, B1 = f0 + f1, f1 - f0, b0 + b1, b1 - b0
        det = F * F - G * G
        ok = (det > 1e-6 * np.maximum(F * F, 1e-30)) & (shells <= RADIUS)
        safe = np.where(ok, det, 1)
        m = np.where(ok, (F * B0 - G * B1) / safe, 0)
        w = np.where(ok, (F * B1 - G * B0) / safe, 0)
        solutions.append((m, w, ok))
    keep = solutions[0][2] & solutions[1][2]
    v = [load_mrc(str(evaluation / f"state{s:03d}.mrc")).astype(np.float32) for s in args.states]
    w_true = (v[1] - v[0]) / 2
    manifest = json.loads((evaluation / "manifest.json").read_text())
    training = json.loads((args.dataset / "training" / "manifest.json").read_text())
    mask = np.asarray(support_mask(shape[0], float(training["particle_diameter_ang"]) / float(manifest["voxel_size"])))

    def fsc(a, b):
        cross = np.bincount(shells, np.real(a * np.conj(b)) * keep, nshells)
        pa = np.bincount(shells, np.abs(a) ** 2 * keep, nshells)
        pb = np.bincount(shells, np.abs(b) ** 2 * keep, nshells)
        return np.where(pa * pb > 0, cross / np.sqrt(np.maximum(pa * pb, 1e-300)), np.nan)

    out = {"n_images": sum(counts), "states": args.states}
    for name, idx in (("mean", 0), ("loading", 1)):
        a, b = solutions[0][idx], solutions[1][idx]
        out[f"fsc_{name}"] = fsc(a, b)
        out[f"fsc_{name}_masked"] = fsc(masked(a, shape, mask), masked(b, shape, mask))
    tables = oracle_tables(args.dataset, args.q)
    shells3 = np.asarray(ftu.get_grid_of_radial_distances(shape)).reshape(-1)
    base = sum(counts) * tables["kappa"] * tables["snr_img"] / np.maximum(tables["probe_state_power_3d"], 1e-300)
    out["snr_loading_predicted"] = (base * shell_power_3d_full(w_true, shells3, len(base)))[:nshells]
    out["snr_mean_predicted"] = (base * shell_power_3d_full((v[0] + v[1]) / 2, shells3, len(base)))[:nshells]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **out)
    print(f"{args.dataset.name} states {args.states}: n {sum(counts)}")
    print("shell  FSC_w   FSC_w_m  snr_w_pred  ratio_u  ratio_m |  FSC_m   ratio_u  ratio_m (mean channel)")
    for r in range(1, RADIUS + 1, 2):
        row = []
        for name in ("loading", "mean"):
            f_u, f_m = out[f"fsc_{name}"][r], out[f"fsc_{name}_masked"][r]
            snr = out[f"snr_{name}_predicted"][r]
            ratio = lambda f: 2 * f / (1 - f) / snr if 0.1 < f < 0.9 else np.nan  # noqa: E731
            row.append((f_u, f_m, snr, ratio(f_u), ratio(f_m)))
        (a1, a2, a3, a4, a5), (b1, _, _, b4, b5) = row
        print(f"{r:5d} {a1:7.3f} {a2:7.3f} {a3:11.3g} {a4:8.2f} {a5:8.2f} | {b1:7.3f} {b4:8.2f} {b5:8.2f}")


if __name__ == "__main__":
    main()
