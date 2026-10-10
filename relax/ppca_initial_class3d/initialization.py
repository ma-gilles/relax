"""Mixture-only signed bootstrap; the upstream single-model initializer is unchanged.

The bootstrap and noise orchestration follows ``ppca_initial_model`` so the
same seed consumes the same random draws. Its public projection, blob-model,
noise and tomography helpers remain the numerical owners. Only the mixture's
model sign follows the centre of its random-angle bootstrap reconstruction.
"""

import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.core import fourier_transform_utils as ftu
from scipy.spatial.transform import Rotation

from relax.ppca_initial_model.initialization import initial_noise
from relax.ppca_initial_model.initialization import seed_model as unsigned_seed_model
from relax.ppca_initial_model.tomo import tilt_bootstrap, tilt_noise_images
from relax.ppca_refinement.residual_statistics import full_float32
from relax.vdam.bootstrap_iref import compute_ini_high_shell


def seed_model(rhs, lhs, volume_shape, voxel_size, diameter_ang, radius, rng):
    """Upstream random blobs with the mixture bootstrap's protein sign.

    Applying one global sign after the linear Fourier/mask operations preserves
    the former signed-blob formula without changing upstream or its RNG draws.
    This is a mixture initialization convention, not a change to image contrast.
    """
    half_shape = ftu.volume_shape_to_half_volume_shape(volume_shape)
    reconstruction = jnp.where(lhs > 0, rhs / jnp.where(lhs > 0, lhs, 1), 0)
    real = np.asarray(
        ftu.get_idft3_real(reconstruction.reshape((rhs.shape[0],) + half_shape), volume_shape), np.float32
    )
    x = np.arange(volume_shape[0]) - volume_shape[0] // 2
    centre = np.sqrt(sum(a**2 for a in np.meshgrid(x, x, x, indexing="ij"))) <= diameter_ang / voxel_size / 4
    sign = np.float32(-1.0 if float(np.mean(real[0][centre])) < 0 else 1.0)
    theta = unsigned_seed_model(rhs, lhs, volume_shape, voxel_size, diameter_ang, radius, rng)
    return sign * theta


@full_float32
def initialize(dataset, *, seed, diameter_ang, batch_size=64, q=2):
    """SPA bootstrap using a signed mixture seed and the upstream noise estimator.

    The upstream initializer has no seed-model callback. Keep its orchestration
    here rather than temporarily replacing a shared function during a run.
    """
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q <= 0:
        raise ValueError("q must be a positive integer")
    channels = q + 1
    rng = np.random.default_rng(seed)
    ids = rng.permutation(dataset.n_images)[:1000]
    half_shape = ftu.volume_shape_to_half_volume_shape(dataset.volume_shape)
    rhs = jnp.zeros((channels, int(np.prod(half_shape))), jnp.complex64)
    lhs = jnp.zeros(rhs.shape, jnp.float32)
    images_for_noise = []
    radius = max(1, compute_ini_high_shell(dataset.grid_size))
    offset = 0
    for images, _r, _t, ctf_params, _noise, _pids, _ids in dataset.iter_batches(batch_size, indices=ids, by_image=True):
        count = len(images)
        images_for_noise.extend((0, np.asarray(im, np.float32)) for im in images)
        images_ft = dataset.process_images_half(images, apply_image_mask=False).reshape(count, -1).astype(jnp.complex64)
        ctf = dataset.ctf_evaluator(ctf_params, dataset.image_shape, dataset.voxel_size, half_image=True).astype(
            jnp.float32
        )
        rotations = Rotation.random(count, random_state=rng).as_matrix().astype(np.float32)
        labels = (np.arange(count) + offset) % channels
        for k in range(channels):
            select = np.flatnonzero(labels == k)
            if not len(select):
                continue
            r = core.adjoint_slice_volume(
                images_ft[select] * ctf[select], rotations[select], dataset.image_shape, dataset.volume_shape,
                "linear_interp", half_image=True, half_volume=True, max_r=radius,
            )
            l = core.adjoint_slice_volume(
                ctf[select] ** 2, rotations[select], dataset.image_shape, dataset.volume_shape,
                "linear_interp", half_image=True, half_volume=True, max_r=radius,
            )
            rhs = rhs.at[k].add(r)
            lhs = lhs.at[k].add(l.real)
        offset += count
    theta = seed_model(rhs, lhs, dataset.volume_shape, dataset.voxel_size, diameter_ang, radius, rng)
    noise = initial_noise(images_for_noise, dataset.image_shape, dataset.voxel_size, diameter_ang)
    return theta, noise, {
        "seed": seed,
        "bootstrap_ids": ids.tolist(),
        "bootstrap_radius": radius,
        "projection": "complex64",
        "accumulation": "complex64/float32",
        "noise_initializer": "deliberate host float64; coefficient spectrum float32",
    }


@full_float32
def initialize_tilts(particles, *, seed, diameter_ang, q=2):
    """ET mixture bootstrap; one shared spectrum per noise group, as upstream."""
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q <= 0:
        raise ValueError("q must be a positive integer")
    rng = np.random.default_rng(seed)
    radius = max(1, compute_ini_high_shell(particles.grid_size))
    rhs, lhs, chosen = tilt_bootstrap(particles, rng, q + 1, radius)
    theta = seed_model(rhs, lhs, particles.volume_shape, particles.voxel_size, diameter_ang, radius, rng)
    images_for_noise = tilt_noise_images(particles, rng)
    noise = initial_noise(
        images_for_noise, particles.image_shape, particles.voxel_size, diameter_ang, n_groups=particles.n_noise_groups
    )
    return theta, noise, {
        "seed": seed,
        "bootstrap_particles": chosen.tolist(),
        "noise_tilt_images": len(images_for_noise),
        "noise_groups": particles.n_noise_groups,
        "bootstrap_radius": radius,
        "projection": "complex64",
        "accumulation": "complex64/float32",
        "noise_initializer": "deliberate host float64; coefficient spectrum float32",
    }
