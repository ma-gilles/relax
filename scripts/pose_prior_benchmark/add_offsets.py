#!/usr/bin/env python
"""Give a pose-prior benchmark dataset translational offsets.

RECOVAR's simulator cannot make offsets: every simulated particle is centred and ``particles.star`` has zero origins.
This script writes a copy of a dataset whose particles are displaced from the box centre, as picked particles are.
Each particle draws an offset (a 2D Gaussian, independent of its pose, defocus and noise), its image is shifted by it
(a phase ramp in Fourier space, so a fractional shift is exact and white noise stays white), and the offset is
written as the particle's true ``rlnOriginXAngst`` / ``rlnOriginYAngst``. The CTF is a product in Fourier space, as
the shift is, so shifting the finished image equals shifting the particle before the CTF; the noise is shift-invariant.
Every other column, the ground-truth and start maps and the pose distribution are those of the source dataset.

Run (CPU node)::

    python scripts/pose_prior_benchmark/add_offsets.py --dataset-dir DATASET --out-dir OUT --sigma-angstrom 7 --seed 29

``OUT`` receives ``particles.star``, ``particles.<box>.mrcs``, copies of the source's ``reference_gt_relion.mrc`` and
``reference_init_relion.mrc``, and ``pose_distribution.json`` (the source's record plus an ``offsets`` entry).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import mrcfile
import numpy as np
import scipy.fft
import starfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The image content moves by ORIGIN_SIGN * origin / pixel size along x (the fast array axis) and y (the slow one): a
# particle with origin o sits at -o from the box centre and RELION moves it back by +o. Checked by reconstructing 4000
# shifted particles with relion_reconstruct from their true angles and origins: the map equals that of the unshifted
# particles (correlation 1.0000, FSC 1.000 in every shell), while the opposite sign gives noise.
ORIGIN_SIGN = -1.0
SHIFT_CHUNK = 500  # particles shifted per FFT batch
COPIED_FILES = ("reference_gt_relion.mrc", "reference_init_relion.mrc")
ORIGIN_COLUMNS = ("rlnOriginXAngst", "rlnOriginYAngst")


def draw_offsets(n_particles: int, sigma_angstrom: float, seed: int) -> np.ndarray:
    """Offsets (x, y) in A of ``n_particles`` particles: independent Gaussians of standard deviation ``sigma_angstrom``.

    Returns an (n, 2) float64 array that depends on ``seed`` and ``n_particles`` only.
    """
    return np.random.default_rng(seed).normal(0.0, sigma_angstrom, size=(n_particles, 2))


def shift_images(images: np.ndarray, shifts_px: np.ndarray, workers: int = 1) -> np.ndarray:
    """Move the content of each image by its (x, y) shift in pixels, circularly, with a Fourier phase ramp.

    ``images`` is (n, ny, nx); a positive x shift moves the content to higher column indices, as
    ``np.roll(image, s, axis=-1)`` does for an integer ``s``. Returns float32 of the same shape. The Nyquist row and
    column lose the imaginary part of a fractional shift, about 0.4% of white-noise variance at box 256.
    """
    n, ny, nx = images.shape
    spectrum = scipy.fft.rfft2(images.astype(np.float32), axes=(-2, -1), workers=workers)
    kx = np.fft.rfftfreq(nx)[None, None, :]
    ky = np.fft.fftfreq(ny)[None, :, None]
    sx = shifts_px[:, 0, None, None]
    sy = shifts_px[:, 1, None, None]
    spectrum *= (np.exp(-2j * np.pi * kx * sx) * np.exp(-2j * np.pi * ky * sy)).astype(np.complex64)
    return scipy.fft.irfft2(spectrum, s=(ny, nx), axes=(-2, -1), workers=workers).astype(np.float32)


def stack_name_of(particles: Any) -> str:
    """The single stack file ``rlnImageName`` names, whose images are numbered 1..n in row order."""
    names = particles["rlnImageName"].str.split("@", n=1, expand=True)
    stacks = names[1].unique()
    if len(stacks) != 1:
        raise ValueError(f"the particles name {len(stacks)} stacks, expected one")
    if not (names[0].astype(int).to_numpy() == np.arange(1, len(particles) + 1)).all():
        raise ValueError("the particles are not the stack's images 1..n in row order")
    return str(stacks[0])


def add_offsets(dataset_dir: Path, out_dir: Path, sigma_angstrom: float, seed: int, workers: int = 1) -> dict[str, Any]:
    """Write the offset copy of the dataset in ``dataset_dir`` to ``out_dir``; returns the record of the offsets.

    The record (also stored in ``pose_distribution.json``) holds the Gaussian's standard deviation, the seed and the
    offsets' empirical mean, rms and maximum radius in A. Refuses a source that already has non-zero origins.
    """
    if sigma_angstrom <= 0:
        raise ValueError(f"sigma_angstrom must be positive, got {sigma_angstrom}")
    blocks = starfile.read(dataset_dir / "particles.star", always_dict=True)
    optics, particles = blocks["optics"], blocks["particles"]
    if any(np.any(particles[c].to_numpy() != 0) for c in ORIGIN_COLUMNS):
        raise ValueError("the source dataset already has non-zero origins")
    stack = stack_name_of(particles)
    pixel = float(optics["rlnImagePixelSize"].iloc[0])
    offsets = draw_offsets(len(particles), sigma_angstrom, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with mrcfile.mmap(dataset_dir / stack, permissive=True, mode="r") as src:
        shape = src.data.shape
        with mrcfile.new_mmap(out_dir / stack, shape=shape, mrc_mode=2, overwrite=True) as dst:
            for start in range(0, shape[0], SHIFT_CHUNK):
                stop = min(start + SHIFT_CHUNK, shape[0])
                shifts = ORIGIN_SIGN * offsets[start:stop] / pixel
                dst.data[start:stop] = shift_images(np.asarray(src.data[start:stop]), shifts, workers)
            dst.voxel_size = pixel
    particles = particles.assign(**{c: offsets[:, i] for i, c in enumerate(ORIGIN_COLUMNS)})
    starfile.write({"optics": optics, "particles": particles}, out_dir / "particles.star", overwrite=True)
    radius = np.hypot(offsets[:, 0], offsets[:, 1])
    record = {
        "kind": "gaussian",
        "sigma_angstrom": sigma_angstrom,
        "seed": seed,
        "mean_x_angstrom": float(offsets[:, 0].mean()),
        "mean_y_angstrom": float(offsets[:, 1].mean()),
        "rms_radius_angstrom": float(np.sqrt(np.mean(radius**2))),
        "max_radius_angstrom": float(radius.max()),
    }
    poses = json.loads((dataset_dir / "pose_distribution.json").read_text())
    (out_dir / "pose_distribution.json").write_text(json.dumps({**poses, "offsets": record}, indent=2) + "\n")
    for name in COPIED_FILES:
        shutil.copy2(dataset_dir / name, out_dir / name)
    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-dir", type=Path, required=True, help="simulated dataset (simulate_dataset.py)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sigma-angstrom", type=float, required=True, help="per-axis standard deviation of the offsets")
    parser.add_argument("--seed", type=int, required=True, help="seed of the offsets")
    parser.add_argument("--workers", type=int, default=1, help="FFT threads")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    record = add_offsets(args.dataset_dir, args.out_dir, args.sigma_angstrom, args.seed, args.workers)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
