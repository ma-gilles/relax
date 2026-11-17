"""A per-image rotation prior centred on each image's input pose (section 19 of ``docs/math/vdam_ppca_algorithm.md``).

Image ``i`` with input rotation ``R0_i`` (RELION's ``Euler_angles2matrix`` of its ``rlnAngleRot/Tilt/Psi``, the
matrix the stream's rotation grid holds: :func:`relax.sampling.grids.get_relion_hidden_rotation_grid`) gives each
candidate rotation ``R`` the log-prior

    log p(R | R0_i) = -theta(R, R0_i)^2 / (2 sigma^2) - log_norm,
    theta(R, R0) = min_G arccos((tr(R0^T R G) - 1) / 2),

the geodesic angle to the nearest symmetry copy ``R G`` (``G`` the point group's operators; ``R`` and ``R G``
project a ``G``-symmetric map alike). ``log_norm`` is the log mean of ``exp(-theta^2 / (2 sigma^2))`` over the
stream's rotation grid at ``R0 = I``: the HEALPix grid is uniform to within its quadrature, so the prior sums to
about one over the grid for every ``R0``. The stream adds it to the grid's own (direction) log-prior.

The scorers add it per (rotation, image) where they form the pose log-prior
(:mod:`relax.ppca_refinement.full_row_kernels`, :mod:`relax.ppca_refinement.oversampled_stream`); the CUDA
epilogue reads only a shared prior table, so on GPU streams the term enters through the score GEMM's mean channel
(:func:`add_to_mean_inner`): the score is ``-0.5 (|A_0|^2 - 2 Re<A_0, Y>) + ...``, so adding ``c`` to
``Re<A_0, Y>`` adds exactly ``c`` to the score and leaves the latent moments unchanged.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

HIGHEST = jax.lax.Precision.HIGHEST


@dataclasses.dataclass(frozen=True, eq=False)
class PosePrior:
    """Host description of the prior: each image's input rotation ``centres`` ``(N, 3, 3)`` indexed by original
    image id, ``sigma_deg`` and the symmetry ``operators`` ``(S, 3, 3)`` (identity first)."""

    centres: np.ndarray
    sigma_deg: float
    operators: np.ndarray

    def __post_init__(self):
        centres = np.asarray(self.centres)
        operators = np.asarray(self.operators)
        if centres.ndim != 3 or centres.shape[1:] != (3, 3) or operators.ndim != 3 or operators.shape[1:] != (3, 3):
            raise ValueError("Pose prior centres and operators are (n, 3, 3) rotation matrices")
        if not (np.isfinite(self.sigma_deg) and self.sigma_deg > 0):
            raise ValueError(f"The pose prior sigma must be finite and positive, got {self.sigma_deg}")
        if not np.all(np.isfinite(centres)):
            raise ValueError("Pose prior centres must be finite")

    @property
    def inverse_two_sigma2(self) -> float:
        """``1 / (2 sigma^2)`` in rad^-2."""
        return 0.5 / np.deg2rad(float(self.sigma_deg)) ** 2


class ImagePosePrior(NamedTuple):
    """Device operands of the prior, fixed for one stream (its rotation grid sets ``log_norm``)."""

    operators: jax.Array  # (S, 3, 3) float32
    inverse_two_sigma2: jax.Array  # () float32, rad^-2
    log_norm: jax.Array  # () float32


def centres_sha256(centres) -> str:
    """Identity of a set of input rotations (the multi-GPU ranks compare it)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(centres, np.float32)).tobytes()).hexdigest()


def symmetric_traces_np(rotations, centres, operators) -> np.ndarray:
    """``max_G tr(R0^T R G)`` ``(R, B)`` in float64 (the host reference)."""
    rotations = np.asarray(rotations, np.float64)
    centres = np.asarray(centres, np.float64)
    copies = np.einsum("rij,sjk->srik", rotations, np.asarray(operators, np.float64))
    return np.einsum("srik,bik->srb", copies, centres).max(axis=0)


def angle_from_trace(trace, xp=np):
    """The rotation angle of a rotation with trace ``trace``: ``2 asin(sqrt((3 - tr) / 4))``, which is
    ``arccos((tr - 1) / 2)`` and better conditioned near zero."""
    return 2.0 * xp.arcsin(xp.sqrt(xp.clip((3.0 - trace) * 0.25, 0.0, 1.0)))


def geodesic_angles_np(rotations, centres, operators) -> np.ndarray:
    """Symmetry-minimized geodesic angles ``(R, B)`` in radians, float64."""
    return angle_from_trace(symmetric_traces_np(rotations, centres, operators))


def log_normalizer(prior: PosePrior, rotations) -> float:
    """``log mean_r exp(-theta(R_r, I)^2 / (2 sigma^2))`` over the grid ``rotations``, float64 (cached per grid,
    sigma and operators: every update builds its streams again)."""
    rotations = np.ascontiguousarray(rotations, np.float32)
    operators = np.ascontiguousarray(prior.operators, np.float64)
    key = (
        hashlib.sha256(rotations.tobytes()).hexdigest(),
        rotations.shape,
        float(prior.sigma_deg),
        operators.tobytes(),
    )
    if key not in _NORMALIZERS:
        # At R0 = I: tr(R G) = sum_ij R_ij G_ji, maximized over the operators.
        trace = np.einsum("rij,sji->sr", rotations.astype(np.float64), operators).max(axis=0)
        log_weights = -(angle_from_trace(trace) ** 2) * prior.inverse_two_sigma2
        top = log_weights.max()
        if len(_NORMALIZERS) >= 16:
            _NORMALIZERS.pop(next(iter(_NORMALIZERS)))
        _NORMALIZERS[key] = float(top + np.log(np.mean(np.exp(log_weights - top))))
    return _NORMALIZERS[key]


_NORMALIZERS: dict = {}


def device_prior(prior: PosePrior, rotations) -> ImagePosePrior:
    """:class:`ImagePosePrior` of ``prior`` for a stream over the grid ``rotations`` (sentinel row excluded)."""
    return ImagePosePrior(
        operators=jnp.asarray(np.asarray(prior.operators, np.float32)),
        inverse_two_sigma2=jnp.asarray(np.float32(prior.inverse_two_sigma2)),
        log_norm=jnp.asarray(np.float32(log_normalizer(prior, rotations))),
    )


def tile_centres(prior_centres, original_ids, n_slots: int) -> np.ndarray:
    """The tile's input rotations ``(n_slots, 3, 3)`` float32 in tile order: its real images' (``original_ids``)
    first, identity for the padding slots after them (they carry no posterior mass)."""
    original_ids = np.asarray(original_ids, np.int64)
    if original_ids.size > n_slots:
        raise ValueError("A tile has more images than slots")
    centres = np.tile(np.eye(3, dtype=np.float32), (n_slots, 1, 1))
    centres[: original_ids.size] = np.asarray(prior_centres, np.float32)[original_ids]
    return centres


def _log_prior_from_trace(prior: ImagePosePrior, trace):
    theta = angle_from_trace(trace, jnp)
    return -(theta * theta) * prior.inverse_two_sigma2 - prior.log_norm


def outer_rotation_log_prior(prior: ImagePosePrior, rotations, centres):
    """``log p(R_r | R0_b)`` ``(R, B)`` of ``rotations`` ``(R, 3, 3)`` at every image's ``centres`` ``(B, 3, 3)``."""
    S, R, B = prior.operators.shape[0], rotations.shape[0], centres.shape[0]
    copies = jnp.einsum("rij,sjk->srik", rotations, prior.operators, precision=HIGHEST).reshape(S * R, 9)
    trace = jnp.dot(copies, centres.reshape(B, 9).T, precision=HIGHEST).reshape(S, R, B)
    return _log_prior_from_trace(prior, jnp.max(trace, axis=0))


def paired_rotation_log_prior(prior: ImagePosePrior, rotations, centres):
    """``log p(R_jr | R0_j)`` ``(J, R)`` of rotations ``(J, R, 3, 3)`` each at its own ``centres[j]`` ``(J, 3, 3)``."""
    copies = jnp.einsum("jrik,skl->sjril", rotations, prior.operators, precision=HIGHEST)
    trace = jnp.einsum("sjril,jil->sjr", copies, centres, precision=HIGHEST)
    return _log_prior_from_trace(prior, jnp.max(trace, axis=0))


def add_to_pose_log_prior(pose_log_prior, image_rotation_prior):
    """``pose_log_prior`` ``(B, R, T)`` plus the per-image term ``(R, B)`` (None: unchanged)."""
    if image_rotation_prior is None:
        return pose_log_prior
    return pose_log_prior + jnp.transpose(image_rotation_prior)[:, :, None]


def add_to_mean_inner(inner, image_rotation_prior, n_translations: int):
    """The score GEMM output ``inner`` ``(P, R, ld)`` (column ``b T + t``, ``ld >= B T``) with the per-image term
    ``(R, B)`` added to its mean channel ``Re<A_0, Y_bt>``, which enters the pose score with weight one
    (None: unchanged). The padding columns past ``B T`` are left as they are."""
    if image_rotation_prior is None:
        return inner
    B = image_rotation_prior.shape[1]
    return inner.at[0, :, : B * n_translations].add(jnp.repeat(image_rotation_prior, n_translations, axis=1))
