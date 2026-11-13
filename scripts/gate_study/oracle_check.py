"""Known-truth check of the oracle gate tables (gate study): measured full-data weights against the predicted ones.

For one state of a simulated suite dataset (a homogeneous set's only state, or one state of a heterogeneous set,
reconstructed from that state's particles alone), back-project the training images at their true rotations with the
CTF and linear interpolation (``recovar.core.adjoint_slice_volume``, as scripts/write_ppca_truth_mean.py), in two halves
of n/2 particles: theta_hat = B / F per voxel (the noise is radially symmetric, so a per-shell sigma^2 cancels). Then
per shell r, against the truth s (that state's map, its sign fitted on the low shells, masked as the model is):

- measured optimal weight of the full reconstruction, g*(r) = Re<theta_hat, s> / <|theta_hat|^2> (the scalar that
  minimises |g theta_hat - s|^2 over the shell), unmasked and with the model's real-space mask applied to both;
- measured half-map FSC, unmasked and masked;
- the predictions of scripts/gate_study/oracle_gate.py for that state and n: snr = n kappa snr_img P_state / P_probe,
  W = snr / (1 + snr) (task), snr / f (mask-aware), half-map FSC = (snr/2) / (1 + snr/2) and its masked version.

The implied full-data SNR of a measured weight is g / (1 - g); its ratio to the predicted snr per shell is the correction
an oracle needs (1 if the prediction is right).

Run (CPU or GPU): python scripts/gate_study/oracle_check.py DATASET_DIR STAR --state K [--max-images N] -o check.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp  # noqa: E402
from recovar import core  # noqa: E402
from recovar.core import fourier_transform_utils as ftu  # noqa: E402
from recovar.utils.helpers import load_mrc  # noqa: E402

from relax.commands.ppca_initial_model import load_training  # noqa: E402
from relax.ppca_initial_model.initialization import support_mask  # noqa: E402
from scripts.gate_study.oracle_gate import oracle_tables, shell_power_3d_full  # noqa: E402

RADIUS = 31


def half_reconstructions(dataset, rotations, ids, batch=200):
    """B and F (half volume, linear interpolation, max radius RADIUS) of the two halves (alternate ids)."""
    half = ftu.volume_shape_to_half_volume_shape(dataset.volume_shape)
    out = []
    for part in (ids[0::2], ids[1::2]):
        rhs = jnp.zeros(int(np.prod(half)), jnp.complex64)
        lhs = jnp.zeros(rhs.shape, jnp.float32)
        for images, _r, _t, ctf_params, _n, particle_ids, _i in dataset.iter_batches(batch, indices=part, by_image=True):
            count = len(images)
            spectra = dataset.process_images_half(images, apply_image_mask=False).reshape(count, -1).astype(jnp.complex64)
            ctf = dataset.ctf_evaluator(ctf_params, dataset.image_shape, dataset.voxel_size, half_image=True).astype(jnp.float32)
            args = (rotations[np.asarray(particle_ids)].astype(np.float32), dataset.image_shape, dataset.volume_shape)
            kwargs = {"half_image": True, "half_volume": True, "max_r": RADIUS}
            rhs += core.adjoint_slice_volume(spectra * ctf, *args, "linear_interp", **kwargs)
            lhs += core.adjoint_slice_volume(ctf**2, *args, "linear_interp", **kwargs).real
        out.append((np.asarray(rhs), np.asarray(lhs)))
    return out


def masked(spectrum, shape, mask):
    real = ftu.get_idft3_real(jnp.asarray(spectrum)[None].reshape((1,) + ftu.volume_shape_to_half_volume_shape(shape)), shape)
    return np.asarray(ftu.get_dft3_real(real * mask)).reshape(-1)


def shell_reduce(values, shells, nshells):
    return np.bincount(shells, values, nshells)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("star", type=Path)
    parser.add_argument("--state", type=int, default=0)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    evaluation = args.dataset / "evaluation"
    with np.load(evaluation / "truth.npz") as truth:
        labels, rotations = truth["labels"], truth["rotations"]
    ids = np.flatnonzero(labels == args.state)
    ids = np.random.default_rng(0).permutation(ids)[: args.max_images or len(ids)]
    n = len(ids)
    dataset, _, _ = load_training(str(args.star), "auto")
    shape = dataset.volume_shape
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(shape)).reshape(-1)
    nshells = int(shells.max()) + 1
    manifest = json.loads((evaluation / "manifest.json").read_text())
    training = json.loads((args.dataset / "training" / "manifest.json").read_text())
    mask = np.asarray(support_mask(shape[0], float(training["particle_diameter_ang"]) / float(manifest["voxel_size"])))

    (b0, f0), (b1, f1) = half_reconstructions(dataset, rotations, ids)
    keep = (f0 > 0) & (f1 > 0) & (shells <= RADIUS)
    full = np.where(keep, (b0 + b1) / np.where(keep, f0 + f1, 1), 0)
    h0 = np.where(keep, b0 / np.where(keep, f0, 1), 0)
    h1 = np.where(keep, b1 / np.where(keep, f1, 1), 0)
    state_map = load_mrc(str(evaluation / f"state{args.state:03d}.mrc")).astype(np.float32)
    s = np.asarray(ftu.get_dft3_real(jnp.asarray(state_map)[None])).reshape(-1)
    low = keep & (shells <= 8)
    sign = 1.0 if np.real(np.vdot(s[low], full[low])) > 0 else -1.0
    s = sign * s

    def measures(theta, h0_, h1_, truth_):
        sel = keep
        num = shell_reduce(np.real(np.conj(theta) * truth_) * sel, shells, nshells)
        den = shell_reduce(np.abs(theta) ** 2 * sel, shells, nshells)
        g = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
        cross = shell_reduce(np.real(h0_ * np.conj(h1_)) * sel, shells, nshells)
        p0 = shell_reduce(np.abs(h0_) ** 2 * sel, shells, nshells)
        p1 = shell_reduce(np.abs(h1_) ** 2 * sel, shells, nshells)
        fsc = np.where(p0 * p1 > 0, cross / np.sqrt(np.maximum(p0 * p1, 1e-300)), np.nan)
        return g, fsc

    g_u, fsc_u = measures(full, h0, h1, s)
    mm = lambda x: masked(x, shape, mask)  # noqa: E731
    g_m, fsc_m = measures(mm(full), mm(h0), mm(h1), mm(s))

    # Predictions for this state and n (oracle_gate's machinery with the state's own power).
    tables = oracle_tables(args.dataset, args.q)
    shells3 = np.asarray(ftu.get_grid_of_radial_distances(shape)).reshape(-1)
    p_state = shell_power_3d_full(state_map, shells3, len(tables["kappa"]))
    snr = n * tables["kappa"] * tables["snr_img"] * p_state / np.maximum(tables["probe_state_power_3d"], 1e-300)
    snr = snr[:nshells]
    f = tables["support_fraction"]
    wiener = lambda x: x / (1 + x)  # noqa: E731
    out = {
        "n_images": n, "state": args.state, "sign": sign, "support_fraction": f,
        "measured_weight": g_u, "measured_weight_masked": g_m, "measured_fsc_half": fsc_u, "measured_fsc_half_masked": fsc_m,
        "snr_predicted": snr, "w_task": wiener(snr), "w_masked": wiener(snr / f),
        "fsc_half_predicted": wiener(snr / 2), "fsc_half_masked_predicted": wiener(snr / 2 / f),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **out)
    implied = lambda g: g / np.maximum(1 - g, 1e-6)  # noqa: E731
    print(f"{args.dataset.name} state {args.state}: n {n}, sign {sign:+.0f}, support fraction {f:.3f}")
    print("shell  g*      W_task  g*_mask W_mask   FSCh    pred    FSChm   predm   ratio_u ratio_m")
    for r in range(1, RADIUS + 1, 2):
        print(
            f"{r:5d} {g_u[r]:7.3f} {out['w_task'][r]:7.3f} {g_m[r]:7.3f} {out['w_masked'][r]:7.3f} "
            f"{fsc_u[r]:7.3f} {out['fsc_half_predicted'][r]:7.3f} {fsc_m[r]:7.3f} {out['fsc_half_masked_predicted'][r]:7.3f} "
            f"{implied(g_u[r]) / snr[r]:7.2f} {implied(g_m[r]) / snr[r]:7.2f}"
        )


if __name__ == "__main__":
    main()
