"""Float32 VDAM-style random seed maps, with no supplied poses or volumes.

Algorithm section 10: random-angle round-robin reconstructions set the scale
of positive-minus-half-negative blob fields; three fields map to one mean and
two normalized contrasts. Native double bootstrap is not called.
"""

import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.core import fourier_transform_utils as ftu
from scipy.spatial.transform import Rotation

from relax.ppca_initial_model.noise import relion_to_coefficient_variance
from relax.ppca_refinement.residual_statistics import full_float32
from relax.relion.initial_noise import compute_avg_unaligned_and_sigma2


def seed_maps_to_model(volumes, *, compute_dtype=jnp.float32):
    """Exact q=2 mean/contrast construction; empirical covariance divisor is 3."""
    v = jnp.asarray(volumes, compute_dtype)
    if v.shape[0] != 3:
        raise ValueError("The selected initializer requires exactly three seed maps (q=2)")
    mu = (v[0] + v[1] + v[2]) / 3
    W = jnp.stack(
        [
            (v[0] - v[1]) / jnp.sqrt(jnp.asarray(6, compute_dtype)),
            (v[0] + v[1] - 2 * v[2]) / jnp.sqrt(jnp.asarray(18, compute_dtype)),
        ]
    )
    return jnp.concatenate([mu[None], W])


def support_mask(n, diameter_px, edge=5):
    coords = jnp.arange(n, dtype=jnp.float32) - n // 2
    radius = jnp.sqrt(sum(x * x for x in jnp.meshgrid(coords, coords, coords, indexing="ij")))
    phase = jnp.clip((radius - diameter_px / 2) / edge, 0, 1)
    return (1 + jnp.cos(jnp.pi * phase)) / 2


def bandlimit_and_mask(theta, shape, radius, mask):
    """Common real-space mask and common spherical support for all channels."""
    half_shape = ftu.volume_shape_to_half_volume_shape(shape)
    real = ftu.get_idft3_real(theta.T.reshape((-1,) + half_shape), shape)
    spectrum = ftu.get_dft3_real(real * mask).reshape(theta.shape[1], -1).T
    radii = ftu.get_grid_of_radial_distances_real(shape, rounded=False).reshape(-1)
    active = radii <= radius
    return spectrum * jnp.asarray(active[:, None], spectrum.real.dtype)


def _blob_field(amplitude, diameter, rng):
    n = amplitude.shape[0]
    # Host RNG/geometry is deliberate; all field arithmetic stays float32.
    centers = (rng.normal(size=(40, 3)) * diameter / 6 + n / 2).astype(np.float32)
    grid = np.indices((n, n, n), dtype=np.float32)
    out = np.zeros((n, n, n), np.float32)
    for center in centers:
        index = center.astype(np.int64)
        if np.any(index < 0) or np.any(index >= n):
            continue
        distance = (grid - center[:, None, None, None]) * np.float32(10 / diameter)
        inside = np.all(np.abs(grid - center[:, None, None, None]) < diameter / 3, axis=0)
        out += np.exp(-np.sum(distance * distance, axis=0)) * abs(amplitude[tuple(index)]) * inside
    return out


@full_float32
def initialize(dataset, *, seed, diameter_ang, batch_size=64):
    """Return shared half-Fourier theta, inferred shell noise and seed metadata."""
    n = dataset.grid_size
    rng = np.random.default_rng(seed)
    ids = rng.permutation(dataset.n_images)[:1000]
    half_shape = ftu.volume_shape_to_half_volume_shape(dataset.volume_shape)
    rhs = jnp.zeros((3, int(np.prod(half_shape))), jnp.complex64)
    lhs = jnp.zeros(rhs.shape, jnp.float32)
    images_for_noise = []
    radius = max(1, int(np.floor(0.07 * n + 0.5)))
    offset = 0
    for images, _r, _t, ctf_params, _noise, _pids, _ids in dataset.iter_batches(batch_size, indices=ids, by_image=True):
        count = len(images)
        images_for_noise.extend((0, np.asarray(im, np.float32)) for im in images)
        images_ft = dataset.process_images_half(images, apply_image_mask=False).reshape(count, -1).astype(jnp.complex64)
        ctf = dataset.ctf_evaluator(ctf_params, dataset.image_shape, dataset.voxel_size, half_image=True).astype(
            jnp.float32
        )
        rotations = Rotation.random(count, random_state=rng).as_matrix().astype(np.float32)
        labels = (np.arange(count) + offset) % 3
        for k in range(3):
            select = np.flatnonzero(labels == k)
            if not len(select):
                continue
            r = core.adjoint_slice_volume(
                images_ft[select] * ctf[select],
                rotations[select],
                dataset.image_shape,
                dataset.volume_shape,
                "linear_interp",
                half_image=True,
                half_volume=True,
                max_r=radius,
            )
            l = core.adjoint_slice_volume(
                ctf[select] ** 2,
                rotations[select],
                dataset.image_shape,
                dataset.volume_shape,
                "linear_interp",
                half_image=True,
                half_volume=True,
                max_r=radius,
            )
            rhs = rhs.at[k].add(r)
            lhs = lhs.at[k].add(l.real)
        offset += count
    reconstruction = jnp.where(lhs > 0, rhs / jnp.where(lhs > 0, lhs, 1), 0)
    real = np.asarray(ftu.get_idft3_real(reconstruction.reshape((3,) + half_shape), dataset.volume_shape), np.float32)
    diameter = diameter_ang / dataset.voxel_size
    mask = support_mask(n, diameter)
    fields = []
    for volume in real:
        field = _blob_field(volume, diameter, rng) - np.float32(0.5) * _blob_field(volume, diameter, rng)
        sd = np.std(field, dtype=np.float32)
        if sd <= 0:
            raise ValueError("Random bootstrap produced a zero blob field")
        fields.append(field * (np.std(volume, dtype=np.float32) / sd))
    augmented = seed_maps_to_model(np.stack(fields))
    theta = ftu.get_dft3_real(augmented).reshape(3, -1).T
    theta = bandlimit_and_mask(theta, dataset.volume_shape, radius, mask)
    if not np.all(np.isfinite(np.asarray(theta))) or np.any(np.linalg.norm(np.asarray(theta[:, 1:]), axis=0) == 0):
        raise ValueError("Random initialization requires finite, nonzero loadings")
    # Existing estimator deliberately uses host float64 accumulation/metadata.
    _, sigma = compute_avg_unaligned_and_sigma2(
        iter(images_for_noise),
        ori_size=n,
        pixel_size=dataset.voxel_size,
        particle_diameter_ang=diameter_ang,
        width_mask_edge_px=5,
        do_zero_mask=False,
        nr_optics_groups=1,
    )
    noise = relion_to_coefficient_variance(sigma[0], dataset.image_shape)
    return (
        theta,
        noise,
        {
            "seed": seed,
            "bootstrap_ids": ids.tolist(),
            "bootstrap_radius": radius,
            "projection": "complex64",
            "accumulation": "complex64/float32",
            "noise_initializer": "deliberate host float64; coefficient spectrum float32",
        },
    )
