"""RELION's ``--zero_mask`` for the mixture's subtomogram tilt images.

relion_refine with ``--zero_mask`` (``do_zero_mask``, ml_optimiser.cpp) replaces every real-space particle
image outside the particle diameter by zeros under a raised-cosine edge of ``width_mask_edge`` (5) pixels
before any Fourier operand, so the solvent adds neither signal nor noise to the likelihood, and estimates
its start-up noise spectrum from the same masked images. The mixture gets the identical weights on every
tilt image through a wrapped :class:`~relax.ppca_initial_model.tomo.TiltParticles` reader, so the E-step,
the random start-up bootstrap and the noise estimate all see masked images. Single-particle manifests read
their images inside the streamed engine and are not masked here.
"""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np

EDGE_PX = 5  # RELION's width_mask_edge


def circular_zero_mask(image_shape, diameter_px: float, edge_px: int = EDGE_PX) -> np.ndarray:
    """RELION's ``softMaskOutsideMap`` weights on a centered ``image_shape`` grid: 1 inside ``diameter_px / 2``,
    a raised cosine over ``edge_px`` pixels, 0 beyond (mask.cpp; the volume form is
    :func:`relax.ppca_initial_model.initialization.support_mask`)."""
    if not np.isfinite(diameter_px) or diameter_px <= 0:
        raise ValueError("The zero mask needs a positive particle diameter")
    h, w = (int(s) for s in image_shape)
    y = np.arange(h, dtype=np.float32) - h // 2
    x = np.arange(w, dtype=np.float32) - w // 2
    radius = np.sqrt(y[:, None] ** 2 + x[None, :] ** 2)
    phase = np.clip((radius - np.float32(diameter_px / 2)) / np.float32(edge_px), 0, 1)
    return ((1 + np.cos(np.pi * phase)) / 2).astype(np.float32)


def zero_masked_tilt_particles(particles, process_images_half, diameter_px: float):
    """``particles`` with a reader that zero-masks the real-space tilt images before their half spectra.

    ``process_images_half`` is the image dataset's own ``process_images_half`` (the half-spectrum operand the
    unmasked reader forms), so the masked operands follow the same packing and normalization. The CTF rows
    are the reader's own.
    """
    mask = circular_zero_mask(particles.image_shape, diameter_px)
    base = particles.read

    def read(image_ids):
        raw, _, ctf = base(image_ids)
        masked = np.asarray(raw, np.float32) * mask
        half = process_images_half(jnp.asarray(masked)).reshape(masked.shape[0], -1)
        return masked, half.astype(jnp.complex64), ctf

    return dataclasses.replace(particles, read=read)


def load_tilt_training(ios, output, *, zero_mask_diameter_ang=None):
    """Subtomogram particles of a RELION 5 optimisation set and their input identity.

    Uses the package-local merged-stack adapter with upstream geometry and CTF
    handling. Keeping the tilt dataset in scope also lets its reader be wrapped
    by :func:`zero_masked_tilt_particles` when ``zero_mask_diameter_ang`` is given.
    """
    from relax.commands.ppca_initial_model import refuse_unsupported_tilt_optics, tilt_series_hash
    from relax.ppca_initial_class3d.tomo_input import load_tomo_dataset
    from relax.ppca_initial_model.checkpoint import file_hash
    from relax.ppca_initial_model.tomo import tilt_particles_from_tomo_dataset
    from relax.relion.tomo_input import read_optimisation_set

    particles_star, tomograms_star = read_optimisation_set(ios)
    refuse_unsupported_tilt_optics(particles_star)
    tomo = load_tomo_dataset(
        particles_star,
        tomograms_star,
        Path(output) / "particles_2d.star",
        datadir=str(Path(particles_star).resolve().parent),
        lazy=False,
    )
    identity = {
        "ios_sha256": file_hash(ios),
        "particles_sha256": file_hash(particles_star),
        "tomograms_sha256": file_hash(tomograms_star),
        "tilt_series_sha256": tilt_series_hash(tomograms_star),
    }
    particles = tilt_particles_from_tomo_dataset(tomo)
    if zero_mask_diameter_ang is not None:
        particles = zero_masked_tilt_particles(
            particles, tomo.images.process_images_half, zero_mask_diameter_ang / particles.voxel_size
        )
    return particles, identity
