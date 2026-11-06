"""The expected rotation accuracy of a PPCA model, for the HEALPix order rule (docs/math/vdam_ppca_algorithm.md 4.1).

RELION's ``calculateExpectedAngularErrors`` rotates the reference projection at an image's best pose until the
squared difference, in units of the noise, passes a threshold. Here the two poses are compared as image laws: the
Kullback-Leibler divergence between the marginal Gaussians of the two poses, whose covariance carries the loadings
projected at each pose. With no loadings it is RELION's quantity.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from recovar import core
from recovar.core.configs import ForwardModelConfig
from recovar.reconstruction import noise as noise_utils
from recovar.reconstruction.noise import make_radial_noise

from relax.ppca_refinement.dense_blocks import prepare_dense_ppca_dataset_inputs
from relax.sampling.healpix import euler_angles_to_matrix
from relax.sampling.relion_expected_accuracy import PVALUE

# RELION's trial particles (ml_optimiser.cpp calculateExpectedAngularErrors) and the largest rotation it tries.
N_TRIALS = 100
MAX_ERROR_DEG = 30.0
# Projections of one chunk of trials are kept under this many bytes.
CHUNK_BYTES = 1 << 28


class KLTerms(NamedTuple):
    """The three parts of twice the divergence (section 4.1), each of the shape of the batch."""

    mean: jax.Array  # delta* V1^-1 delta: the difference of the mean images against the second pose's covariance
    log_det: jax.Array  # log det M1 - log det M0: the change of the posterior precision of the latents
    trace: jax.Array  # tr(V1^-1 V0) - n: the change of the covariance

    @property
    def kl(self) -> jax.Array:
        """The divergence K(pose 0 || pose 1)."""
        return 0.5 * (self.mean + self.log_det + self.trace)


def _gram(left: jax.Array, right: jax.Array) -> jax.Array:
    """``Re(left^* right)`` over the pixels: the real-image inner products of the loading columns."""
    return jnp.real(jnp.conj(left).T @ right)


def _kl_terms_one(difference: jax.Array, load0: jax.Array, load1: jax.Array) -> KLTerms:
    """:func:`kl_terms` for one pair of poses: ``difference`` ``(F,)``, ``load0`` and ``load1`` ``(F, q)``."""
    quad = jnp.real(jnp.vdot(difference, difference))
    q = load0.shape[-1]
    if q == 0:
        zero = jnp.zeros_like(quad)
        return KLTerms(quad, zero, zero)
    eye = jnp.eye(q, dtype=quad.dtype)
    m0, m1, g01 = eye + _gram(load0, load0), eye + _gram(load1, load1), _gram(load0, load1)
    chol0, chol1 = jnp.linalg.cholesky(m0), jnp.linalg.cholesky(m1)
    m1_inv = jax.scipy.linalg.cho_solve((chol1, True), eye)
    b = jnp.real(jnp.conj(load1).T @ difference)

    def log_det(chol: jax.Array) -> jax.Array:
        return 2 * jnp.sum(jnp.log(jnp.diagonal(chol)))

    return KLTerms(
        mean=quad - b @ m1_inv @ b,
        log_det=log_det(chol1) - log_det(chol0),
        trace=jnp.trace(m0) + jnp.trace(m1_inv) - 2 * q - jnp.trace(m1_inv @ g01.T @ g01),
    )


def kl_terms(difference: jax.Array, load0: jax.Array, load1: jax.Array) -> KLTerms:
    """The parts of ``2 K(pose 0 || pose 1)`` between two image laws with the same noise (section 4.1).

    All inputs are scaled by ``sqrt(CTF^2 w / noise)``, ``w`` the Hermitian weights times the Fourier window, so
    that ``Re(a^* b)`` is the engine's inner product: ``difference`` ``(..., F)`` complex is the mean image of pose 0
    minus that of pose 1, ``load0`` and ``load1`` ``(..., F, q)`` the projected loadings of each pose. The batch
    dimensions are free; ``q = 0`` gives half the squared difference as RELION has it.
    """
    batch = difference.shape[:-1]
    size = int(np.prod(batch))
    terms = jax.vmap(_kl_terms_one)(
        difference.reshape((size,) + difference.shape[-1:]),
        load0.reshape((size,) + load0.shape[-2:]),
        load1.reshape((size,) + load1.shape[-2:]),
    )
    return KLTerms(*(part.reshape(batch) for part in terms))


def error_angles() -> tuple[np.ndarray, float]:
    """RELION's rotation steps in degrees and the error it keeps when none passes the threshold.

    The step is 0.05 below 0.2 degrees, then 0.1 below 1, 0.2 below 2, 0.5 below 5, 1 below 10, 2 below 20 and 5
    beyond; the list ends at the first angle above ``MAX_ERROR_DEG``, which is the error of a trial no step passes.
    """
    angle, angles = 0.0, []
    while True:
        step = (
            0.05
            if angle < 0.2
            else 0.1
            if angle < 1.0
            else 0.2
            if angle < 2.0
            else 0.5
            if angle < 5.0
            else (1.0 if angle < 10.0 else 2.0 if angle < 20.0 else 5.0)
        )
        angle += step
        angles.append(angle)
        if angle > MAX_ERROR_DEG:
            return np.asarray(angles[:-1]), angles[-1]


class PoseAccuracy(NamedTuple):
    """The trials' expected rotation error under the divergence and under the mean image alone."""

    acc_rot: float  # degrees: mean over the trials of the first angle with K above the threshold
    acc_rot_mean_only: float  # the same with the loadings left out of the covariance (RELION's quantity)
    terms: dict  # the trials' mean of (mean, log_det, trace) over two, at each trial's error
    n_trials: int


def first_passing(
    values: np.ndarray, angles: np.ndarray, cap: float, threshold: float = PVALUE
) -> tuple[np.ndarray, np.ndarray]:
    """Each trial's first angle whose value passes ``threshold`` and its index (``cap`` and the last index if none).

    A value that is not finite raises: a diverged model has no accuracy, and RELION's loop would read NaN as a pass.
    """
    if not np.all(np.isfinite(values)):
        raise ValueError("The pose accuracy is not finite: the model's loadings or noise have diverged")
    passed = values > threshold
    index = np.where(passed.any(axis=1), passed.argmax(axis=1), len(angles) - 1)
    return np.where(passed.any(axis=1), angles[index], cap), index


class Divergences(NamedTuple):
    """The trials' divergence at every step of the rotation schedule."""

    angles: np.ndarray  # (steps,) degrees
    cap: float  # the error of a trial no step passes
    kl: np.ndarray  # (trials, steps): K(best pose || pose turned by the angle)
    mean_only: np.ndarray  # (trials, steps): half the squared difference of the mean images, RELION's quantity
    parts: np.ndarray  # (trials, steps, 3): twice K as (mean, log_det, trace)


def trial_divergences(
    dataset, theta, noise_row, *, ids: np.ndarray, eulers_deg: np.ndarray, radius: int, active_pcs: int
) -> Divergences:
    """The divergence of each trial image's best pose from the poses turned by :func:`error_angles`.

    ``theta`` ``(half volume, 1 + active_pcs)`` is the model the E-step used (mean, then loadings) and ``noise_row``
    its noise spectrum; ``ids`` are the trial images and ``eulers_deg`` ``(n, 3)`` their best poses (RELION's
    rot, tilt, psi). Trial ``i`` turns the Euler angle ``i mod 3``. The projections, window, weights and noise are
    the E-step's.
    """
    ids = np.asarray(ids)
    eulers = np.asarray(eulers_deg, np.float64).reshape(-1, 3)
    n = len(ids)
    resolved = prepare_dense_ppca_dataset_inputs(
        dataset,
        theta[:, 0],
        theta[:, 1 : 1 + active_pcs],
        q=active_pcs,
        volume_domain="fourier_half",
        current_size=2 * radius,
        full_real_observation=True,
    )
    forward = ForwardModelConfig.from_dataset(dataset, disc_type="linear_interp", process_fn=dataset.process_images)
    noise = noise_utils.to_batched_half_pixel_noise(
        make_radial_noise(noise_row, dataset.image_shape), resolved.image_shape
    ).squeeze()
    _, _, _, ctf_params, _, _, indices = next(iter(dataset.iter_batches(n, indices=ids, by_image=False)))
    if not np.array_equal(np.asarray(indices), ids):
        raise ValueError("The dataset returned the trial images in another order")
    scale = jnp.sqrt(forward.compute_ctf_half(ctf_params) ** 2 / noise * resolved.score_mask)

    angles, cap = error_angles()
    steps = len(angles)
    turned = np.zeros((n, steps, 3))
    turned[np.arange(n)[:, None], np.arange(steps)[None, :], (np.arange(n) % 3)[:, None]] = angles
    perturbed = eulers[:, None, :] + turned
    q = resolved.q
    chunk = max(1, CHUNK_BYTES // ((steps + 1) * (q + 1) * scale.shape[-1] * 8))
    kl, mean_only, parts = [], [], []
    for first in range(0, n, chunk):
        rows = slice(first, min(first + chunk, n))
        count = rows.stop - rows.start
        matrices = euler_angles_to_matrix(np.concatenate([eulers[rows], perturbed[rows].reshape(-1, 3)])).astype(
            np.float32
        )
        projections = jnp.swapaxes(
            core.batch_slice_volume(
                resolved.augmented_half_volumes,
                jnp.asarray(matrices),
                resolved.image_shape,
                resolved.volume_shape,
                "linear_interp",
                half_volume=True,
                half_image=True,
                relion_texture_interp=False,
                max_r=resolved.projection_max_r,
            ),
            0,
            1,
        )
        best, moved = projections[:count], projections[count:].reshape(count, steps, q + 1, -1)
        weight = scale[rows][:, None, :]
        difference = weight * (best[:, None, 0] - moved[:, :, 0])
        load0 = jnp.broadcast_to(
            jnp.swapaxes(best[:, None, 1:], -1, -2) * weight[..., None], (count, steps, scale.shape[-1], q)
        )
        load1 = jnp.swapaxes(moved[:, :, 1:], -1, -2) * weight[..., None]
        terms = kl_terms(difference, load0, load1)
        kl.append(np.asarray(terms.kl))
        parts.append(np.stack([np.asarray(part) for part in terms], axis=-1))
        mean_only.append(0.5 * np.asarray(jnp.sum(jnp.abs(difference) ** 2, axis=-1)))
    return Divergences(angles, cap, np.concatenate(kl), np.concatenate(mean_only), np.concatenate(parts))


def expected_rotation_accuracy(
    dataset,
    theta,
    noise_row,
    *,
    ids: np.ndarray,
    eulers_deg: np.ndarray,
    radius: int,
    active_pcs: int,
    threshold: float = PVALUE,
) -> PoseAccuracy:
    """The expected rotation accuracy of the model at the trial images' best poses (arguments of
    :func:`trial_divergences`): the mean over the trials of the first angle whose divergence passes ``threshold``
    (RELION's ``PVALUE`` by default)."""
    d = trial_divergences(
        dataset, theta, noise_row, ids=ids, eulers_deg=eulers_deg, radius=radius, active_pcs=active_pcs
    )
    error, index = first_passing(d.kl, d.angles, d.cap, threshold)
    error_mean_only, _ = first_passing(d.mean_only, d.angles, d.cap, threshold)
    at_error = d.parts[np.arange(len(error)), index]
    return PoseAccuracy(
        acc_rot=float(np.mean(error)),
        acc_rot_mean_only=float(np.mean(error_mean_only)),
        terms=dict(zip(("mean", "log_det", "trace"), (0.5 * at_error.mean(axis=0)).tolist(), strict=True)),
        n_trials=len(error),
    )
