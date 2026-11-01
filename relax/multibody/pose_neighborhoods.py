"""Per-body pose comparison with the unobserved depth left free, for fixed-pose subset selection.

For one body, with mask voxel coordinates ``x_l`` relative to its native pivot and normalised mask weights ``w_l``,
take ``m = sum w_l x_l`` and ``C = sum w_l (x_l - m)(x_l - m)^T`` (angstroms). A particle's rotation ``Q_i`` is known
in the common molecular frame, and its translation only as a line ``t_i = u_i + lambda_i n_i`` along the viewing
direction ``n_i``. The score of two particles is the smallest weighted RMS displacement of corresponding body points
over the unobserved depths::

    d(i, j)^2 = tr[(Q_i - Q_j) C (Q_i - Q_j)^T]
                + min_{|l_i|, |l_j| <= L} ||u_i - u_j + (Q_i - Q_j) m + l_i n_i - l_j n_j||^2

Finite rotations are used exactly. ``d`` is a compatibility score, not a metric: different physical states can have
zero score, so it has no triangle inequality and does not recover true 3D translation distances. Viewing rotations
only interpret the observation; they are not coordinates. Each body's score reads that body's poses alone.

Particle ``i`` of a multi-body run (``relax.multibody.geometry``): with consensus matrix ``A_i``, composed body matrix
``B_i``, native pivot ``c0``, consensus origin ``o_i``, fitted body origin ``f_i`` (pixels) and pixel size ``a``::

    Q_i = A_i^T B_i              p_i = (A_i c0)_xy        r_i = ROUND(o_i - p_i)
    e_i = o_i - p_i - r_i        s_i = a (e_i - f_i)
    u_i = A_i^T (s_ix, s_iy, 0)  n_i = A_i^T (0, 0, 1)

so the body's pivot lies at ``-r_i - f_i`` in the raw image, a displacement ``s_i`` from where the consensus puts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from relax.multibody import geometry
from relax.multibody.bodies import Body


@dataclass(frozen=True)
class BodyMoments:
    """Mask moments of one body about its native pivot, angstroms: ``centroid`` is ``m``, ``covariance`` is ``C``."""

    centroid: np.ndarray  # [3]
    covariance: np.ndarray  # [3, 3]


@dataclass(frozen=True)
class PoseFeatures:
    """One body's precomputed comparison terms, ``[N, ...]``: ``F_i = Q_i C^(1/2)``, ``a_i = u_i + Q_i m`` and the
    unit depth direction ``n_i``, float64."""

    shape_factors: np.ndarray  # [N, 3, 3]
    points: np.ndarray  # [N, 3]
    normals: np.ndarray  # [N, 3]


def mask_moments(mask: np.ndarray, pivot_px, pixel_size: float) -> BodyMoments:
    """The moments of a soft mask (RELION frame, ``bodies``) about ``pivot_px`` (x, y, z), every nonzero voxel weighted
    by its value."""

    mask = np.asarray(mask, dtype=np.float64)
    if mask.min() < 0.0:
        raise ValueError("a body mask must be nonnegative")
    k, i, j = np.nonzero(mask)
    weights = mask[k, i, j] / mask[k, i, j].sum()
    half = mask.shape[0] // 2
    x = (np.stack([j, i, k], axis=1) - half - np.asarray(pivot_px, dtype=np.float64)) * float(pixel_size)
    centroid = weights @ x
    centred = x - centroid
    return BodyMoments(centroid, (centred * weights[:, None]).T @ centred)


def body_pose_lines(consensus: np.ndarray, composed: np.ndarray, pivot_px, consensus_offsets_px, body_offsets_px,
                    pixel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``Q``, ``u`` (angstroms) and ``n`` of every particle (module docstring). ``consensus`` and ``composed`` are the
    ``[N, 3, 3]`` camera matrices ``A`` and ``B``; offsets are RELION origins in pixels."""

    consensus = np.asarray(consensus, dtype=np.float64)
    transposed = np.swapaxes(consensus, 1, 2)
    offsets = np.asarray(consensus_offsets_px, dtype=np.float64)
    projected = (consensus @ np.asarray(pivot_px, dtype=np.float64))[:, :2]
    fractional = offsets - projected - geometry.integer_image_shifts(offsets, projected)
    shift = float(pixel_size) * (fractional - np.asarray(body_offsets_px, dtype=np.float64))
    lift = transposed[:, :, :2]
    return transposed @ np.asarray(composed, dtype=np.float64), np.einsum("nij,nj->ni", lift, shift), transposed[:, :, 2]


def body_features(consensus: np.ndarray, composed: np.ndarray, body: Body, consensus_offsets_px, body_offsets_px,
                  pixel_size: float) -> PoseFeatures:
    """:class:`PoseFeatures` of a multi-body run's body, about its native pivot."""

    rotations, lifts, normals = body_pose_lines(
        consensus, composed, body.com, consensus_offsets_px, body_offsets_px, pixel_size
    )
    return pose_features(rotations, lifts, normals, mask_moments(body.mask, body.com, pixel_size))


def pose_features(rotations: np.ndarray, lifts: np.ndarray, normals: np.ndarray, moments: BodyMoments) -> PoseFeatures:
    """:class:`PoseFeatures` from the rotations ``Q``, zero-depth lifts ``u`` and unit depth directions ``n``."""

    values, vectors = np.linalg.eigh(moments.covariance)
    root = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    rotations = np.asarray(rotations, dtype=np.float64)
    return PoseFeatures(rotations @ root, np.asarray(lifts, dtype=np.float64) + rotations @ moments.centroid,
                        np.asarray(normals, dtype=np.float64))


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1)


def _box_minimum(delta: np.ndarray, normal_i: np.ndarray, normal_j: np.ndarray, bound: float) -> np.ndarray:
    """``min ||delta + alpha n_i - beta n_j||^2`` over ``|alpha|, |beta| <= bound`` of every row: the feasible interior
    minimum and the four clipped edge minima, compared by their residuals."""

    candidates = []
    for beta in (-bound, bound):
        shifted = delta - beta * normal_j
        candidates.append((np.clip(-_dot(shifted, normal_i), -bound, bound), np.full(len(delta), beta)))
    for alpha in (-bound, bound):
        shifted = delta + alpha * normal_i
        candidates.append((np.full(len(delta), alpha), np.clip(_dot(shifted, normal_j), -bound, bound)))
    cross = np.cross(normal_i, normal_j)
    length2 = _dot(cross, cross)
    safe = np.where(length2 > 0.0, length2, 1.0)
    alpha = -_dot(np.cross(delta, normal_j), cross) / safe
    beta = -_dot(np.cross(delta, normal_i), cross) / safe
    feasible = (length2 > 0.0) & (np.abs(alpha) <= bound) & (np.abs(beta) <= bound)

    def residual2(alpha, beta):
        rest = delta + alpha[:, None] * normal_i - beta[:, None] * normal_j
        return _dot(rest, rest)

    edges = np.min([residual2(a, b) for a, b in candidates], axis=0)
    interior = residual2(np.where(feasible, alpha, 0.0), np.where(feasible, beta, 0.0))
    return np.where(feasible, np.minimum(interior, edges), edges)


def _line_minimum(delta: np.ndarray, normal_i: np.ndarray, normal_j: np.ndarray) -> np.ndarray:
    """``min ||delta + alpha n_i - beta n_j||^2`` with free depths: the distance between the two lines."""

    cross = np.cross(normal_i, normal_j)
    length2 = _dot(cross, cross)
    along_i = delta - _dot(delta, normal_i)[:, None] * normal_i
    along_j = delta - _dot(delta, normal_j)[:, None] * normal_j
    parallel = np.minimum(_dot(along_i, along_i), _dot(along_j, along_j))
    safe = np.where(length2 > 0.0, length2, 1.0)
    return np.where(length2 > 0.0, np.minimum(_dot(delta, cross) ** 2 / safe, parallel), parallel)


def _target_terms(features: PoseFeatures, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = features.shape_factors - features.shape_factors[target]
    shape = np.sum(difference * difference, axis=(1, 2))
    return shape, features.points - features.points[target], np.broadcast_to(features.normals[target], features.normals.shape)


def compatibility_distance(features: PoseFeatures, target: int, depth_bound: float | None) -> np.ndarray:
    """``d(i, target)`` of every particle in angstroms, ``[N]``; ``depth_bound=None`` leaves the depths free (a
    diagnostic)."""

    return compatibility_distance_and_bound_hits(features, target, depth_bound)[0]


def compatibility_distance_and_bound_hits(features: PoseFeatures, target: int, depth_bound: float | None) -> tuple[np.ndarray, np.ndarray]:
    """:func:`compatibility_distance` and which particles' score the depth bound raises above the free-depth score, by
    more than 1e-12 of the squared centroid displacement (none when the depth is free)."""

    shape, delta, target_normals = _target_terms(features, target)
    free = _line_minimum(delta, features.normals, target_normals)
    if depth_bound is None:
        return np.sqrt(shape + free), np.zeros(len(shape), dtype=bool)
    translation = _box_minimum(delta, features.normals, target_normals, float(depth_bound))
    return np.sqrt(shape + translation), translation > free + 1e-12 * _dot(delta, delta)


def zero_depth_distance(features: PoseFeatures, target: int) -> np.ndarray:
    """The Euclidean comparison of zero-depth lifts, a negative control: it treats a camera-dependent projection of
    the translation as a 3D position."""

    shape, delta, _ = _target_terms(features, target)
    return np.sqrt(shape + _dot(delta, delta))
