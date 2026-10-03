"""Per-particle contrast point estimates, refit outside the stream (section 16.11 of ``docs/math/vdam_ppca_algorithm.md``).

After an E-step, each particle's contrast is the least-squares scale of its images against the model
``mu + W E[z]`` projected at the particle's most probable pose, with a Gaussian prior about 1, clamped to a
range: ``c = (sum <y, m>_D + 1 / s^2) / (sum <m, m>_D + 1 / s^2)`` over the particle's images (its tilts for
subtomograms), with ``m`` the CTF-weighted projection and ``D`` the scoring metric (half-spectrum weights over
the noise, inside the current Fourier radius). The next E-step uses it as a fixed per-particle scale of the
model (the stream's ``image_scale``). One implementation for single particles and subtomogram particles.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise as noise_utils
from recovar.reconstruction.noise import make_radial_noise

from relax.helpers.half_spectrum import make_half_image_weights
from relax.helpers.preprocessing import _dense_batch_ctf_half, relion_half_translation_lattice
from relax.ppca_initial_model.tomo import TiltParticles


def least_squares_contrast(
    half,
    ctf,
    noise_half,
    rotations,
    shifts,
    owner,
    n_owners,
    theta,
    z,
    *,
    image_shape,
    volume_shape,
    radius,
    prior_sd,
    value_range,
):
    """Contrast of each owner from its images ``(n, n_half)`` at their poses; see the module docstring.

    ``rotations`` ``(n, 3, 3)`` and ``shifts`` ``(n, 2)`` (pixels) are each image's projection pose,
    ``owner`` ``(n,)`` its particle among ``n_owners``, ``theta`` ``(half volume, 1 + q)`` the model and
    ``z`` ``(n_owners, q)`` the particles' latent coordinates.
    """
    projections = core.batch_slice_volume(
        jnp.asarray(theta).T,
        jnp.asarray(rotations, jnp.float32),
        image_shape,
        volume_shape,
        "linear_interp",
        half_volume=True,
        half_image=True,
        relion_texture_interp=False,
    )  # (1 + q, n, n_half)
    owner = jnp.asarray(owner)
    model = (projections[0] + jnp.einsum("nq,qnf->nf", jnp.asarray(z, jnp.float32)[owner], projections[1:])) * ctf
    lattice = jnp.asarray(relion_half_translation_lattice(image_shape), jnp.float32)
    data = half * jnp.exp(jnp.complex64(-2j * np.pi) * (jnp.asarray(shifts, jnp.float32) @ lattice.T))
    inside = jnp.linalg.norm(lattice, axis=-1) * image_shape[0] <= radius
    metric = jnp.where(inside, make_half_image_weights(image_shape), 0) / noise_half
    numerator = jnp.zeros(n_owners).at[owner].add(jnp.sum(metric * (jnp.conj(model) * data).real, axis=-1))
    denominator = jnp.zeros(n_owners).at[owner].add(jnp.sum(metric * jnp.abs(model) ** 2, axis=-1))
    precision = 1.0 / prior_sd**2
    estimate = (np.asarray(numerator, np.float64) + precision) / (np.asarray(denominator, np.float64) + precision)
    return np.clip(estimate, *value_range).astype(np.float32)


def particle_contrast(
    dataset,
    theta,
    noise,
    *,
    ids,
    rotation_ids,
    translation_ids,
    z,
    rotations,
    translations,
    radius,
    prior_sd,
    value_range,
    chunk: int = 16,
):
    """MAP-pose contrast of the particles ``ids`` (single-particle images or :class:`TiltParticles`).

    ``rotation_ids`` / ``translation_ids`` index ``rotations`` / ``translations`` (the E-step's grids) at each
    particle's most probable pose; ``z`` are their latent posterior means. ``noise`` is the model's shell
    spectrum, one row per noise group for subtomogram particles.
    """
    ids = np.asarray(ids, np.int64)
    rotations = np.asarray(rotations, np.float64)
    translations = np.asarray(translations, np.float64)
    noise_rows = np.asarray(noise).reshape(-1, np.asarray(noise).shape[-1])
    noise_half = jnp.stack(
        [
            noise_utils.to_batched_half_pixel_noise(
                make_radial_noise(row, dataset.image_shape), dataset.image_shape
            ).reshape(-1)
            for row in noise_rows
        ]
    )
    common = dict(
        image_shape=tuple(dataset.image_shape),
        volume_shape=tuple(dataset.volume_shape),
        radius=radius,
        prior_sd=prior_sd,
        value_range=value_range,
    )
    out = np.ones(ids.size, np.float32)
    for start in range(0, ids.size, chunk):
        part = slice(start, start + chunk)
        particle_rotations = rotations[np.asarray(rotation_ids)[part]]
        particle_shifts = translations[np.asarray(translation_ids)[part]]
        if isinstance(dataset, TiltParticles):
            images, owner = dataset.particle_images(ids[part])
            _, half, ctf = dataset.read(images)
            groups = dataset.particle_group[ids[part]][owner]
            frames = np.stack([dataset.group_frames[g][f] for g, f in zip(groups, dataset.image_frame[images])])
            image_rotations = np.einsum("nab,nbc->nac", frames, particle_rotations[owner])
            image_shifts = np.einsum("nab,nb->na", frames[:, :2, :], particle_shifts[owner])
            image_noise = noise_half[dataset.particle_noise_group[ids[part]][owner]]
        else:
            batch, _, _, ctf_params, _, _, indices = next(
                dataset.iter_batches(ids[part].size, indices=ids[part], by_image=False)
            )
            half = dataset.process_images_half(batch, apply_image_mask=False).reshape(ids[part].size, -1)
            config = ForwardModelConfig.from_dataset(dataset, process_fn=dataset.process_images)
            ctf = _dense_batch_ctf_half(dataset, ctf_params, config, None, indices)
            owner = np.arange(ids[part].size)
            image_rotations, image_shifts, image_noise = particle_rotations, particle_shifts, noise_half[0]
        out[part] = least_squares_contrast(
            jnp.asarray(half, jnp.complex64),
            jnp.asarray(ctf, jnp.float32),
            image_noise,
            image_rotations,
            image_shifts,
            owner,
            ids[part].size,
            theta,
            np.asarray(z)[part],
            **common,
        )
    return out
