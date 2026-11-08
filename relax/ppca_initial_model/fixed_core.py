"""A fixed, known core subtracted from every tilt image before the PPCA sees it.

Particle ``i`` holds a known core volume ``C`` (a microtubule, say) in its own frame plus what the PPCA is
to model (an attachment ``X``). Tilt ``k`` observes

    y_ik = c_i CTF_ik S_{s_ik} P(Aproj_ik) C + CTF_ik S_{s_ik(t)} P(Aproj_ik R)(mu + W z_i) + noise,

with ``s_ik = [Aproj_ik t_i]_{1,2}`` the image shift of the particle's core offset ``t_i`` (3D, pixels, in
the stream's translation convention) and ``c_i`` an optional per-particle core scale (1 by default). The
PPCA then reads ``d_ik = y_ik - c_i CTF_ik S_{s_ik} P(Aproj_ik) C`` instead of ``y_ik``
(:func:`relax.ppca_initial_model.tomo.subtract_fixed_core`), and its empty hypothesis
(:mod:`relax.ppca_refinement.occupancy`) means "core only".

The prediction uses the stream's own projector and conventions, so a core the stream would predict as
``mu`` is removed exactly: RECOVAR's ``batch_slice_volume`` on half volumes and half images with the
stream's ``disc_type`` and no RELION texture lookup, clipped at the full-box radius ``N/2 - 1`` (the
stream's projection radius without a Fourier window), float32 frame matrices, and the image shift as the
conjugate of the phase the stream puts on an image for that shift
(:func:`relax.fourier.preprocessing.translation_phases` on
:func:`relax.fourier.preprocessing.relion_half_translation_lattice`). The stream
compares ``CTF P(mu)`` with ``y e^{-2 pi i s.k}``, so its prediction of ``y`` is ``CTF P(mu) e^{+2 pi i s.k}``.

The volume is in the stream's model units (centered half-Fourier, RECOVAR's internal frame), with no
gridding pre-correction: the stream applies none to ``mu`` either.
"""

from __future__ import annotations

import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from recovar import core

from relax.fourier.preprocessing import relion_half_translation_lattice, translation_phases
from relax.ppca_refinement.dense_blocks import coerce_augmented_half_volumes


@dataclasses.dataclass(frozen=True)
class FixedCore:
    """A core volume in the particle frame: centered half-Fourier ``volume_half`` ``(n_half_volume,)``
    complex64 on a cube ``volume_shape`` of ``voxel_size`` Angstrom, projected with ``disc_type``."""

    volume_half: jax.Array
    volume_shape: tuple
    voxel_size: float
    disc_type: str = "linear_interp"

    @classmethod
    def from_volume(cls, volume, *, volume_shape, voxel_size, volume_domain="auto", disc_type="linear_interp"):
        """A real-space volume (RECOVAR's internal frame) or a centered half-Fourier one, converted as the stream
        converts ``mu`` (:func:`relax.ppca_refinement.dense_blocks.coerce_augmented_half_volumes`)."""
        volume_shape = tuple(int(n) for n in volume_shape)
        stacked, _ = coerce_augmented_half_volumes(volume, None, volume_shape=volume_shape, volume_domain=volume_domain)
        return cls(stacked[0], volume_shape, float(voxel_size), str(disc_type))

    @classmethod
    def from_map(cls, path, *, disc_type="linear_interp"):
        """A RELION-convention MRC map (a RELION or relax map: ``recovar.utils.helpers.load_relion_volume``)."""
        from recovar.utils.helpers import load_relion_volume

        volume, voxel_size = load_relion_volume(str(path), return_voxel_size=True)
        volume = np.asarray(volume, np.float32)
        # The header holds one voxel size per axis (an mrcfile (x, y, z) record); a core map must be isotropic.
        sizes = np.asarray(voxel_size.tolist() if hasattr(voxel_size, "tolist") else voxel_size, np.float64).reshape(-1)
        if not np.allclose(sizes, sizes[0], rtol=1e-5):
            raise ValueError(f"{path} has anisotropic voxels {sizes.tolist()}")
        return cls.from_volume(
            volume, volume_shape=volume.shape, voxel_size=float(sizes[0]), volume_domain="real", disc_type=disc_type
        )


def core_prediction(fixed: FixedCore, image_shape, projections, ctf, shifts) -> jax.Array:
    """``CTF S_s P(Aproj) C`` of each image, ``(n, n_half)`` complex64 on the packed half grid.

    ``projections`` ``(n, 3, 3)`` are the images' full projection rotations of the particle frame
    (``Aproj``), ``ctf`` ``(n, n_half)`` their CTF rows and ``shifts`` ``(n, 2)`` their 2D image shifts in
    pixels (:func:`image_core_shifts`).
    """
    image_shape = tuple(int(n) for n in image_shape)
    return _prediction(
        fixed.volume_half,
        jnp.asarray(np.asarray(projections), jnp.float32),
        jnp.asarray(ctf, jnp.float32),
        jnp.asarray(np.asarray(shifts), jnp.float32),
        jnp.asarray(np.asarray(relion_half_translation_lattice(image_shape), np.float32)),
        image_shape=image_shape,
        volume_shape=tuple(fixed.volume_shape),
        disc_type=fixed.disc_type,
    )


@partial(jax.jit, static_argnames=("image_shape", "volume_shape", "disc_type"))
def _prediction(volume_half, projections, ctf, shifts, lattice, *, image_shape, volume_shape, disc_type):
    sliced = core.batch_slice_volume(
        volume_half[None],
        projections,
        image_shape,
        volume_shape,
        disc_type,
        half_volume=True,
        half_image=True,
        relion_texture_interp=False,
        # The stream's projection radius without a Fourier window (RECOVAR's default clip).
        max_r=image_shape[0] // 2 - 1,
    )[0]
    return (ctf * sliced * jnp.conj(translation_phases(shifts, lattice))).astype(jnp.complex64)


def image_core_shifts(projections, offsets) -> np.ndarray:
    """2D shift ``[Aproj_n t_n]_{1,2}`` of each image for its particle's 3D core offset ``t_n``, ``(n, 2)`` float32.

    RELION's ``getTranslationInTiltSeries`` (:func:`relax.sampling.tomo_particles.tilt_image_shifts`, float64),
    the shift :func:`relax.ppca_initial_model.tomo.tilt_shifts` gives a stream candidate ``t``.
    """
    from relax.sampling.tomo_particles import tilt_image_shifts

    projections = np.asarray(projections, np.float64)
    n = projections.shape[0]
    return tilt_image_shifts(np.zeros((1, 3)), offsets, projections, np.arange(n))[:, 0].astype(np.float32)


def core_scale_sums(prediction, half, precision, owner, n_particles: int):
    """Per-particle sums ``(a, b)`` of the core scale's normal equation, float64 ``(n_particles,)`` each.

    ``a = sum Re(conj(p) y) w / sigma^2`` and ``b = sum |p|^2 w / sigma^2`` over a particle's images, with
    ``p`` the core prediction, ``y`` the observed half spectrum and ``precision = w / sigma^2`` the stream's
    image metric (Hermitian half-image weights over the image's noise, as in ``y_norm``).
    """
    prediction, half, precision = (np.asarray(x) for x in (prediction, half, precision))
    a_image = np.sum((prediction.conj() * half).real * precision, axis=-1, dtype=np.float64)
    b_image = np.sum(np.abs(prediction) ** 2 * precision, axis=-1, dtype=np.float64)
    owner = np.asarray(owner, np.int64)
    return (
        np.bincount(owner, a_image, minlength=n_particles),
        np.bincount(owner, b_image, minlength=n_particles),
    )


def core_scale(a, b, *, prior_sd: float | None = None) -> np.ndarray:
    """Each particle's core scale ``c`` from :func:`core_scale_sums`, float64.

    Without ``prior_sd`` the least-squares scale ``a / b``; with it the posterior mode under ``c ~ N(1,
    prior_sd^2)``, ``(a + 1 / prior_sd^2) / (b + 1 / prior_sd^2)``, which shrinks a weakly determined scale
    toward 1. A particle with no core signal (``b = 0``) without a prior is refused.
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if prior_sd is None:
        if np.any(b <= 0):
            raise ValueError("A particle has no core signal to fit its scale; pass prior_sd")
        return a / b
    if not prior_sd > 0:
        raise ValueError(f"prior_sd must be positive, got {prior_sd}")
    precision = 1.0 / float(prior_sd) ** 2
    return (a + precision) / (b + precision)


def core_basis_volumes(volume) -> list[np.ndarray]:
    """The core and its first-order shape-error directions, real-space float32 volumes in the internal frame
    (axis 0 = RELION x, 1 = y, 2 = z = the filament axis; relax/io/map_io.py): ``[C, x dC/dx + y dC/dy, dC/dx,
    dC/dy]`` (voxel units, central differences, coordinates from the box centre ``N // 2``).

    A particle's core is then fitted as ``c_0 C + c_1 D + c_2 dC/dx + c_3 dC/dy``: its scale, a radial dilation
    of the tube about its axis (``c_0 C((1 + c_1/c_0) r)`` to first order, so a wider tube has ``c_1 < 0``) and a
    transverse shift (``-(c_2, c_3) / c_0`` voxels), all small errors of a fixed core that a scale alone leaves in
    the residual at the wall.
    """
    volume = np.asarray(volume, np.float64)
    n = volume.shape[0]
    x = (np.arange(n) - n // 2)[:, None, None]
    y = (np.arange(n) - n // 2)[None, :, None]
    dx, dy = np.gradient(volume, axis=0), np.gradient(volume, axis=1)
    return [v.astype(np.float32) for v in (volume, x * dx + y * dy, dx, dy)]


def core_basis_sums(predictions, half, precision, owner, n_particles: int):
    """Per-particle normal equations of a core basis fit, float64: ``G`` ``(n_particles, K, K)`` with
    ``G_kl = sum Re(conj(p_k) p_l) w / sigma^2`` and ``h`` ``(n_particles, K)`` with ``h_k = sum Re(conj(p_k) y) w /
    sigma^2`` over a particle's images; ``predictions`` ``(K, n, F)``, the metric as :func:`core_scale_sums`."""
    predictions, half, precision = (np.asarray(x) for x in (predictions, half, precision))
    owner = np.asarray(owner, np.int64)
    K = predictions.shape[0]
    g_image = np.einsum("knf,lnf,nf->nkl", predictions.conj(), predictions, precision).real
    h_image = np.einsum("knf,nf,nf->nk", predictions.conj(), half, precision).real
    G = np.zeros((n_particles, K, K))
    h = np.zeros((n_particles, K))
    np.add.at(G, owner, g_image)
    np.add.at(h, owner, h_image)
    return G, h


def core_basis_coefficients(G, h, *, relative_ridge: float = 1e-6) -> np.ndarray:
    """Least-squares coefficients ``(n_particles, K)`` from :func:`core_basis_sums`, with a ridge of
    ``relative_ridge`` times each diagonal entry (the derivative directions are nearly collinear for a thin wall)."""
    G = np.asarray(G, np.float64)
    h = np.asarray(h, np.float64)
    diagonal = np.einsum("pkk->pk", G)
    if np.any(diagonal[:, 0] <= 0):
        raise ValueError("A particle sees no core signal: its basis fit is undetermined")
    regularized = G + relative_ridge * np.einsum("pk,kl->pkl", diagonal, np.eye(G.shape[1]))
    return np.linalg.solve(regularized, h[..., None])[..., 0]
