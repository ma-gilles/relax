"""Oracle shrinkage weights of the VDAM gate for a simulated PPCA suite dataset (gate study, Phase 0).

Uses the dataset's ``evaluation/`` folder only (ground truth; never a training input). Per Fourier shell r of the
model's box, under the model with known poses:

- image SNR, from ``noise_probe.npz`` (the clean CTF-modulated projections ``signal`` of the first particles and their
  noise ``residual_0``; the training image is exactly their sum): ``snr_img(r) = <|signal^|^2>_r / <|noise^|^2>_r``;
- the full-data weight per voxel: with uniform views every 2D pixel of radius r lands on the 3D shell r, so a voxel
  collects ``kappa(r) = n2D(r) / n3D(r)`` observations per image (full planes and full volume, rounded radii), and
  ``F_all(r) <CTF^2-weighted signal> = N kappa(r) snr_img(r)`` in units of the probe's mean signal power;
- the signal power of each model channel: the population-weighted mean ``mu`` and the loadings ``W_j = sqrt(lambda_j)
  u_j`` (top q eigenvectors of the population-weighted state covariance, from the state maps), as 3D shell powers
  relative to the probe particles' own state power ``P_probe(r) = <|V_label|^2>_r``.

Then ``snr_c(r) = N kappa(r) snr_img(r) P_c(r) / P_probe(r)`` and the oracle Wiener weight ``W_c = snr_c / (1 + snr_c)``
(the task's ``F_all / (F_all + 1/P)`` at fudge 1). The shared loading channel takes ``snr_L = mean_j snr_j`` (the
MSE-optimal common scalar for q components of equal noise). Also written: the code's own target (half-map FSC,
``snr / 2``), the real-space support fraction ``f`` of the model's mask (masking every update removes noise power by about
``f``; the mask-aware weight uses ``snr / f``), and the true powers in the model's units (``ftu.get_dft3_real`` of the
masked real-space maps, shell means over the half volume, as ``stochastic_update`` logs ``signal``).

Run (CPU, a few minutes): python scripts/gate_study/oracle_gate.py DATASET_DIR --q Q -o oracle.npz
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

from relax.ppca_initial_model.initialization import support_mask  # noqa: E402


def shell_power_2d(images: np.ndarray, shells2: np.ndarray, nshells: int) -> np.ndarray:
    """Mean |FFT2|^2 per rounded 2D shell over a stack of real images (full plane)."""
    power = np.zeros(nshells)
    for start in range(0, len(images), 128):
        spectra = np.fft.fftshift(np.fft.fft2(images[start : start + 128].astype(np.float64)), axes=(-2, -1))
        power += np.bincount(shells2, (np.abs(spectra) ** 2).sum(0).reshape(-1), nshells)
    counts = np.bincount(shells2, minlength=nshells)
    return power / np.maximum(counts, 1) / len(images)


def shell_power_3d_full(volume: np.ndarray, shells3: np.ndarray, nshells: int) -> np.ndarray:
    """Mean |FFT3|^2 per rounded 3D shell (full volume, numpy normalisation)."""
    spectrum = np.fft.fftshift(np.fft.fftn(volume.astype(np.float64)))
    counts = np.bincount(shells3, minlength=nshells)
    return np.bincount(shells3, (np.abs(spectrum) ** 2).reshape(-1), nshells) / np.maximum(counts, 1)


def model_unit_power(volume: np.ndarray, mask: np.ndarray, half_shells: np.ndarray, nshells: int) -> np.ndarray:
    """Shell mean of |theta|^2 for a real-space map in the model's units: ftu.get_dft3_real of the masked map, averaged
    over the half-volume voxels of each rounded shell (stochastic_update's shell_average of ``signal``)."""
    spectrum = np.asarray(ftu.get_dft3_real(jnp.asarray(volume * mask, jnp.float32)[None])).reshape(-1)
    counts = np.bincount(half_shells, minlength=nshells)
    return np.bincount(half_shells, np.abs(spectrum) ** 2, nshells) / np.maximum(counts, 1)


def population_pcs(states: np.ndarray, weights: np.ndarray, q: int):
    """Weighted mean and the top-q loadings W_j = sqrt(lambda_j) u_j of the population-weighted state covariance."""
    flat = states.reshape(len(states), -1).astype(np.float64)
    mean = weights @ flat
    centred = np.sqrt(weights)[:, None] * (flat - mean)
    gram = centred @ centred.T
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:q]
    values = np.clip(values[order], 0, None)
    loadings = vectors[:, order].T @ centred  # rows: sqrt(lambda_j) u_j
    return mean.reshape(states.shape[1:]), loadings.reshape((q,) + states.shape[1:]), values


def oracle_tables(dataset: Path, q: int) -> dict:
    evaluation = dataset / "evaluation"
    manifest = json.loads((evaluation / "manifest.json").read_text())
    training = json.loads((dataset / "training" / "manifest.json").read_text())
    counts = np.asarray(manifest["counts"], np.float64)
    weights = counts / counts.sum()
    n_images = int(manifest["n_images"])
    states = np.stack([load_mrc(str(evaluation / f"state{k:03d}.mrc")) for k in range(len(counts))]).astype(np.float32)
    box = states.shape[-1]
    shape = (box,) * 3
    shells3 = np.asarray(ftu.get_grid_of_radial_distances(shape)).reshape(-1)
    shells2 = np.asarray(ftu.get_grid_of_radial_distances((box, box))).reshape(-1)
    half_shells = np.asarray(ftu.get_grid_of_radial_distances_real(shape)).reshape(-1)
    nshells = int(half_shells.max()) + 1
    n3 = np.bincount(shells3, minlength=nshells).astype(np.float64)
    n2 = np.bincount(shells2, minlength=nshells).astype(np.float64)
    kappa = np.where(n3 > 0, n2 / np.maximum(n3, 1), 0.0)

    probe = np.load(evaluation / "noise_probe.npz")
    with np.load(evaluation / "truth.npz") as truth:
        labels = truth["labels"][probe["particle_ids"]]
    signal_2d = shell_power_2d(probe["signal"], shells2, nshells)
    noise_2d = shell_power_2d(probe["residual_0"], shells2, nshells)
    snr_img = np.where(noise_2d > 0, signal_2d / np.maximum(noise_2d, 1e-300), 0.0)

    state_power = np.stack([shell_power_3d_full(v, shells3, nshells) for v in states])
    probe_power = state_power[labels].mean(0)
    mean, loadings, eigenvalues = population_pcs(states, weights, q)
    mean_power = shell_power_3d_full(mean, shells3, nshells)
    loading_power = np.stack([shell_power_3d_full(w, shells3, nshells) for w in loadings])
    data_snr = n_images * kappa * snr_img / np.maximum(probe_power, 1e-300)
    snr_mean = data_snr * mean_power
    snr_pc = data_snr[None] * loading_power
    snr_loading = snr_pc.mean(0)

    diameter_px = float(training["particle_diameter_ang"]) / float(manifest["voxel_size"])
    mask = np.asarray(support_mask(box, diameter_px))
    support_fraction = float(np.mean(mask**2))
    code_mean = model_unit_power(mean, mask, half_shells, nshells)
    code_loading = np.mean([model_unit_power(w, mask, half_shells, nshells) for w in loadings], axis=0)

    def wiener(snr):
        return snr / (1.0 + snr)

    return {
        "dataset": str(dataset),
        "n_images": n_images,
        "q": q,
        "box": box,
        "voxel_size": float(manifest["voxel_size"]),
        "kappa": kappa,
        "snr_img": snr_img,
        "probe_signal_power_2d": signal_2d,
        "probe_noise_power_2d": noise_2d,
        "probe_state_power_3d": probe_power,
        "mean_power_3d": mean_power,
        "loading_power_3d": loading_power,
        "eigenvalues": eigenvalues,
        "snr_mean": snr_mean,
        "snr_pc": snr_pc,
        "snr_loading": snr_loading,
        # The oracle of the task: full-data Wiener weight at fudge 1, [shell, channel] (mean, shared loadings).
        "oracle": np.stack([wiener(snr_mean), wiener(snr_loading)], axis=-1),
        "oracle_pc": wiener(snr_pc),
        # The code's own target (half-map FSC: noise of N/2 particles) and the mask-aware Wiener weight.
        "half_map_fsc": np.stack([wiener(snr_mean / 2), wiener(snr_loading / 2)], axis=-1),
        "support_fraction": support_fraction,
        "oracle_masked": np.stack([wiener(snr_mean / support_fraction), wiener(snr_loading / support_fraction)], axis=-1),
        # True powers in the model's units (masked), for the logged ``signal``; and the matching full-data noise
        # variance of one reconstruction (P / snr), the scaled disagreement's target being 2 V_half = 4 P / snr.
        "code_signal": np.stack([code_mean, code_loading], axis=-1),
        "code_full_data_noise": np.stack(
            [code_mean / np.maximum(snr_mean, 1e-300), code_loading / np.maximum(snr_loading, 1e-300)], axis=-1
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--q", type=int, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    tables = oracle_tables(args.dataset, args.q)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **{k: np.asarray(v) for k, v in tables.items()})
    o = tables["oracle"]
    print(f"{args.dataset.name}: support fraction {tables['support_fraction']:.3f}")
    for r in (1, 2, 4, 8, 11, 16, 20, 24, 28, 31):
        print(
            f"  shell {r:2d}: snr mean {tables['snr_mean'][r]:10.3g} loading {tables['snr_loading'][r]:10.3g}  "
            f"W mean {o[r, 0]:.3f} loading {o[r, 1]:.3f}  half-map {tables['half_map_fsc'][r, 1]:.3f}  "
            f"masked {tables['oracle_masked'][r, 1]:.3f}"
        )


if __name__ == "__main__":
    main()
