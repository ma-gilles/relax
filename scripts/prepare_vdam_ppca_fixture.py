"""Reproducible Ribosembly fixture; truth is confined to evaluation/.

Uses the simulator's MRC downsampling, common volume normalization and
noise_level/50000 white spectrum. No per-particle normalization is applied.
Pilot-scale generation must run under Slurm. See vdam_ppca_implementation_plan.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(output, source, counts, seed=1729, box=64, noise_level=0.01):
    import os

    import jax.numpy as jnp
    from recovar import core
    from recovar.core import fourier_transform_utils as ftu
    from recovar.data_io.cryoem_dataset import CryoEMDataset, ImageMetadata
    from recovar.simulation import simulator
    from recovar.utils import helpers
    from scipy.spatial.transform import Rotation

    if sum(counts) > 256 and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Pilot simulation requires Slurm")
    if not np.isfinite(noise_level) or noise_level <= 0:
        raise ValueError("noise_level must be finite and positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "SAFE_TO_DELETE").touch()
    training, evaluation = output / "training", output / "evaluation"
    training.mkdir()
    evaluation.mkdir()
    maps = [Path(source) / name for name in ["000_8c9c.mrc", "007_8c95.mrc", "015_8c8x.mrc"]]
    volumes, voxel_size = simulator.generate_volumes_from_mrcs(maps, box)
    # Same common normalization as generate_synthetic_dataset, not per-state.
    scale = np.float32(1 / np.mean(np.linalg.norm(volumes, axis=-1)))
    volumes = np.asarray(volumes * scale, np.complex64)
    rng = np.random.default_rng(seed)
    labels = rng.permutation(np.repeat(np.arange(3), counts))
    n = len(labels)
    rotations = Rotation.random(n, random_state=rng).as_matrix().astype(np.float32)
    # Existing simulator's standard random-sampling workflow uses zero shifts.
    translations = np.zeros((n, 2), np.float32)
    ctf = np.zeros((n, int(core.CTFParamIndex.TILT_ANGLE) + 1), np.float32)
    for field, value in [("DFU", 15000), ("DFV", 15000), ("VOLT", 300), ("CS", 2.7), ("W", 0.07), ("CONTRAST", 1)]:
        ctf[:, getattr(core.CTFParamIndex, field)] = value
    data = CryoEMDataset(
        None, voxel_size, ImageMetadata(rotations, translations, ctf), ctf_evaluator=core.CTFEvaluator(), grid_size=box
    )
    noise = np.asarray(simulator.get_noise_model("white", box) / 50000 * noise_level, np.float32)
    particles = simulator.simulate_data(
        data,
        volumes,
        noise,
        64,
        labels,
        np.ones(n, np.float32),
        np.ones(n, np.float32),
        seed=seed,
        noise_rng_batch_size=64,
        noise_transform_batch_size=64,
    )
    helpers.write_mrc_stack(str(training / "particles.mrcs"), particles, voxel_size=voxel_size, dtype=np.float32)
    simulator.save_ctf_params(str(training), box, ctf, voxel_size)
    # Loader-required poses are explicitly neutral; never contain simulation poses.
    helpers.pickle_dump(
        (np.broadcast_to(np.eye(3, dtype=np.float32), (n, 3, 3)).copy(), np.zeros((n, 2), np.float32)),
        str(training / "neutral_poses.pkl"),
    )
    np.save(training / "particle_ids.npy", np.arange(n, dtype=np.int64))
    np.savez(
        evaluation / "truth.npz",
        labels=labels,
        rotations=rotations,
        translations=translations,
        noise_variance=noise,
        volumes_fourier=volumes,
    )
    real_volumes = np.asarray(ftu.get_idft3(jnp.asarray(volumes.reshape(3, box, box, box)))).real
    for k, volume in enumerate(real_volumes):
        helpers.write_mrc(str(evaluation / f"state{k}.mrc"), volume, voxel_size=voxel_size)
    normalized = real_volumes.reshape(3, -1)
    normalized = normalized / np.linalg.norm(normalized, axis=1)[:, None]
    probe_count = min(n, 128)
    probe = CryoEMDataset(
        None,
        voxel_size,
        ImageMetadata(rotations[:probe_count], translations[:probe_count], ctf[:probe_count]),
        ctf_evaluator=core.CTFEvaluator(),
        grid_size=box,
    )
    signal = simulator.simulate_data(
        probe,
        volumes,
        noise * 0,
        64,
        labels[:probe_count],
        np.ones(probe_count, np.float32),
        np.ones(probe_count, np.float32),
        seed=seed,
        noise_rng_batch_size=64,
        noise_transform_batch_size=64,
    )
    measured_noise = particles[:probe_count] - signal
    noise_power = np.mean(np.abs(np.asarray(ftu.get_dft2_real(measured_noise))) ** 2, axis=0)
    shell_grid = np.asarray(ftu.get_grid_of_radial_distances_real((box, box)), np.int32)
    measured_spectrum = np.bincount(shell_grid.ravel(), weights=noise_power.ravel()) / np.bincount(shell_grid.ravel())
    real_radius = np.sqrt(sum(x * x for x in np.meshgrid(*[np.arange(box) - box // 2] * 3, indexing="ij")))
    outer_power = np.sum(real_volumes[:, real_radius > 0.475 * box] ** 2, axis=-1) / np.sum(
        real_volumes**2, axis=(1, 2, 3)
    )
    np.savez(
        evaluation / "noise_probe.npz",
        measured_spectrum=measured_spectrum,
        noiseless_images=signal,
        particle_ids=np.arange(probe_count),
    )
    characterization = {
        "normalized_map_inner_products": (normalized @ normalized.T).tolist(),
        "source_maps": {str(path): sha256(path) for path in maps},
        "downsampled_maps": {path.name: sha256(path) for path in sorted(evaluation.glob("state*.mrc"))},
        "common_volume_scale": float(scale),
        "counts": list(map(int, counts)),
        "measured_snr_probe": float(np.mean(signal**2) / np.mean(measured_noise**2)),
        "measured_noise_coefficient_spectrum": measured_spectrum.tolist(),
        "probe_count": probe_count,
        "outer_power_fraction_r_gt_0_475_box": outer_power.tolist(),
        "simulation_arithmetic": "existing simulator, including deliberate float64 noise RNG; stored particles float32",
        "noise_level": noise_level,
        "injected_spectrum": noise.tolist(),
        "noise_rng_batch_size": 64,
        "seed": seed,
        "shift_distribution": "point mass at zero (standard simulator default)",
        "ctf": "constant defocus 15000 A, 300 kV, Cs 2.7 mm, amplitude contrast 0.07",
        "downsampling": "existing Fourier crop; common field of view preserved",
    }
    (evaluation / "manifest.json").write_text(json.dumps(characterization, indent=2) + "\n")
    manifest = {
        "schema": "recovar-ppca-training-v1",
        "n_images": n,
        "box": box,
        "voxel_size": float(voxel_size),
        "particle_diameter_ang": float(0.75 * box * voxel_size),
        "shift_range_px": 6.0,
        "particles": "particles.mrcs",
        "ctf": "ctf.pkl",
        "neutral_poses": "neutral_poses.pkl",
        "particle_ids": "particle_ids.npy",
        "contrast": 1.0,
        "image_multiplier": 1.0,
        "files": {p.name: sha256(p) for p in training.iterdir()},
    }
    (training / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="/home/mg6942/mytigress/cryobench2/Ribosembly/vols/128_org")
    parser.add_argument("--counts", type=int, nargs=3, default=[6667, 6667, 6666])
    parser.add_argument("--box", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--noise-level", type=float, default=0.01, help="Simulator noise_level parameter")
    args = parser.parse_args()
    prepare(args.output, args.source, args.counts, args.seed, args.box, args.noise_level)


if __name__ == "__main__":
    main()
