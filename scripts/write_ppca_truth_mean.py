"""Write the ground-truth mean of a simulated PPCA fixture as a map for `relax ppca_initial_model --init-mean-map`.

The mean is the particle-count-weighted mean of the fixture's state maps (evaluation/state<k>.mrc, one per count in
evaluation/manifest.json). Its sign in relax's internal frame depends on the training stack (a stack of negated simulator
images holds the negated volumes, scripts/lib/ppca_training_star.py), so the script measures it: it back-projects
particles at their true rotations (evaluation/truth.npz) as the run's random seed does, correlates the result with the
mean over the low shells, writes the map with the sign of that correlation and refuses when the agreement is weak (the
true rotations do not align the data with the maps). The map is written in RELION's convention at the training STAR's
pixel size.

Run (CPU, about a minute):
  python scripts/write_ppca_truth_mean.py FIXTURE_DIR -o truth_mean.mrc
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax.numpy as jnp  # noqa: E402
from recovar import core  # noqa: E402
from recovar.core import fourier_transform_utils as ftu  # noqa: E402
from recovar.utils.helpers import load_mrc  # noqa: E402

from relax.commands.ppca_initial_model import load_training  # noqa: E402
from relax.io.map_io import write_map  # noqa: E402
from scripts.lib.ppca_training_star import training_star_geometry  # noqa: E402

N_IMAGES = 2000  # particles back-projected for the sign
RADIUS = 16  # shells back-projected
AGREEMENT_SHELLS = 8  # the correlation is taken over shells 0..AGREEMENT_SHELLS
MIN_AGREEMENT = 0.5  # |correlation| below this: the data and the maps are not aligned


def truth_mean_volume(evaluation: Path) -> np.ndarray:
    """The count-weighted mean of the state maps of a fixture's ``evaluation`` directory, in the simulator's sign."""
    counts = np.asarray(json.loads((evaluation / "manifest.json").read_text())["counts"], np.float64)
    states = np.stack([load_mrc(str(evaluation / f"state{k}.mrc")) for k in range(len(counts))]).astype(np.float64)
    return np.tensordot(counts / counts.sum(), states, axes=1).astype(np.float32)


def band_correlation(a: np.ndarray, b: np.ndarray, shells: np.ndarray, last_shell: int) -> float:
    """Real correlation of two half-Fourier volumes over the shells ``0..last_shell`` (+1 equal, -1 opposite)."""
    keep = shells <= last_shell
    inner = np.real(np.vdot(b[keep], a[keep]))
    return float(inner / np.sqrt(np.vdot(a[keep], a[keep]).real * np.vdot(b[keep], b[keep]).real))


def mean_sign(correlation: float) -> int:
    """+1 or -1: the sign the mean takes in the training frame, from its correlation with the data's reconstruction."""
    if not np.isfinite(correlation) or abs(correlation) < MIN_AGREEMENT:
        raise ValueError(
            f"The true-rotation reconstruction correlates {correlation:.3f} with the truth mean over shells "
            f"0..{AGREEMENT_SHELLS}: the true rotations do not align the particles with the maps"
        )
    return 1 if correlation > 0 else -1


def true_rotation_reconstruction(fixture: Path) -> np.ndarray:
    """Half-Fourier reconstruction (CTF-weighted back-projection, lightly regularized) of ``N_IMAGES`` random particles
    at their true rotations, in the frame and sign the run's data define."""
    dataset, _, _ = load_training(str(fixture / "training" / "particles.star"), "auto")
    with np.load(fixture / "evaluation" / "truth.npz", allow_pickle=False) as truth:
        rotations = truth["rotations"]
    ids = np.random.default_rng(0).permutation(dataset.n_images)[:N_IMAGES]
    half = ftu.volume_shape_to_half_volume_shape(dataset.volume_shape)
    rhs = jnp.zeros(int(np.prod(half)), jnp.complex64)
    lhs = jnp.zeros(rhs.shape, jnp.float32)
    for images, _r, _t, ctf_params, _noise, particle_ids, _ids in dataset.iter_batches(100, indices=ids, by_image=True):
        count = len(images)
        spectra = dataset.process_images_half(images, apply_image_mask=False).reshape(count, -1).astype(jnp.complex64)
        ctf = dataset.ctf_evaluator(ctf_params, dataset.image_shape, dataset.voxel_size, half_image=True)
        ctf = ctf.astype(jnp.float32)
        args = (rotations[np.asarray(particle_ids)].astype(np.float32), dataset.image_shape, dataset.volume_shape)
        kwargs = {"half_image": True, "half_volume": True, "max_r": RADIUS}
        rhs += core.adjoint_slice_volume(spectra * ctf, *args, "linear_interp", **kwargs)
        lhs += core.adjoint_slice_volume(ctf**2, *args, "linear_interp", **kwargs).real
    return np.asarray(jnp.where(lhs > 0, rhs / (lhs + 1e-3 * lhs.max()), 0))


def main() -> None:
    """Write the fixture's truth mean, with the sign of the training frame, as a relax map."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", type=Path, help="fixture directory holding training/ and evaluation/")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    mean = truth_mean_volume(args.fixture / "evaluation")
    shape = mean.shape
    geometry = training_star_geometry(args.fixture / "training" / "particles.star")
    if shape != (geometry["box"],) * 3:
        raise ValueError(f"The truth mean is {shape}, the training box is {geometry['box']}")
    shells = np.asarray(ftu.get_grid_of_radial_distances_real(shape), np.int32).reshape(-1)
    spectrum = np.asarray(ftu.get_dft3_real(mean[None])).reshape(-1)
    correlation = band_correlation(true_rotation_reconstruction(args.fixture), spectrum, shells, AGREEMENT_SHELLS)
    sign = mean_sign(correlation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_map(args.output, sign * mean, voxel_size=geometry["voxel_size"])
    print(
        f"wrote {args.output}: box {geometry['box']}, {geometry['voxel_size']} A, std {mean.std():.4g}, "
        f"sign {sign:+d} (correlation {correlation:+.3f} of the true-rotation reconstruction with the simulator mean)"
    )


if __name__ == "__main__":
    main()
