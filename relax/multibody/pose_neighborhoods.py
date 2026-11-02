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

The pairwise score sees two particles only, so a depth left free can stay ambiguous for every pair. The population
resolves it: a body's motion is fitted from all particles as ``y_i = mu + W l_i`` with ``k`` orthonormal components,
where ``y_i = (F_i, a_i)`` and each particle's depth is projected out, ``P_i = diag(I, I - n_i n_i^T)``, minimising
``sum ||P_i (y_i - mu - W l_i)||^2``. Particles are then compared at ``|l_i - l_j|``, in angstroms. This is a motion
model, so different states that the model does not separate get the same ``l``.
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


def snap_round_off(distances: np.ndarray, length: float) -> np.ndarray:
    """``distances`` with values at most ``64 eps * length`` set to zero: exact ties that differ only by round-off of
    a geometry of size ``length`` (angstroms) become true ties for ``d <= h``."""

    return np.where(distances <= 64.0 * np.finfo(np.float64).eps * float(length), 0.0, distances)


def zero_depth_distance(features: PoseFeatures, target: int) -> np.ndarray:
    """The Euclidean comparison of zero-depth lifts, a negative control: it treats a camera-dependent projection of
    the translation as a 3D position."""

    shape, delta, _ = _target_terms(features, target)
    return np.sqrt(shape + _dot(delta, delta))


@dataclass(frozen=True)
class LowRankMotion:
    """One body's fitted motion over ``N`` particles: ``y_i = mean + components l_i`` (module docstring)."""

    mean: np.ndarray  # [12]
    components: np.ndarray  # [12, k], orthonormal, by falling variance of the scores
    scores: np.ndarray  # [N, k], angstroms
    objective: np.ndarray  # [iterations + 1], the mean squared residual after each iteration


def _depth_projected(values: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """``P_i values_i`` of ``[N, 12]`` values: the last three coordinates lose their part along ``n_i``."""

    out = values.copy()
    out[:, 9:] -= _dot(values[:, 9:], normals)[:, None] * normals
    return out


def _motion_scores(residual: np.ndarray, components: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Each particle's least-squares scores given the components."""

    along = normals @ components[9:]  # [N, k]
    gram = components.T @ components - along[:, :, None] * along[:, None, :]
    right = residual @ components - along * _dot(residual[:, 9:], normals)[:, None]
    # A view along the whole motion leaves no signal: the tiny ridge sends that score to the mean.
    ridge = 1e-9 * np.eye(components.shape[1])
    return np.linalg.solve(gram + ridge, right[:, :, None])[:, :, 0]


def _motion_update(coordinates: np.ndarray, scores: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The mean and components given the scores: ordinary regression for the shape coordinates, and the normal
    equations of the depth-projected regression for the translation."""

    design = np.hstack([scores, np.ones((len(scores), 1))])  # [N, k + 1]
    size = design.shape[1]
    solved = np.zeros((12, size))
    solved[:9] = np.linalg.lstsq(design, coordinates[:, :9], rcond=None)[0].T
    projection = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    outer = design[:, :, None] * design[:, None, :]
    normal_matrix = np.einsum("nab,nij->aibj", outer, projection, optimize=True).reshape(3 * size, 3 * size)
    right = np.einsum("nij,nj,na->ai", projection, coordinates[:, 9:], design, optimize=True).reshape(-1)
    solved[9:] = np.linalg.lstsq(normal_matrix, right, rcond=None)[0].reshape(size, 3).T
    return solved[:, -1], solved[:, :-1]


def fit_low_rank_motion(features: PoseFeatures, rank: int, iterations: int = 30) -> LowRankMotion:
    """The rank ``rank`` motion of one body, by alternating scores and components from the depth-projected PCA."""

    coordinates = np.hstack([features.shape_factors.reshape(len(features.points), -1), features.points])
    normals = features.normals
    mean = coordinates.mean(axis=0)
    projected = _depth_projected(coordinates - mean, normals)
    components = np.linalg.svd(projected, full_matrices=False)[2][:rank].T
    objective = []

    def error(mean, components, scores):
        return float(np.mean(np.sum(_depth_projected(coordinates - mean - scores @ components.T, normals) ** 2, axis=1)))

    scores = _motion_scores(coordinates - mean, components, normals)
    objective.append(error(mean, components, scores))
    for _ in range(iterations):
        mean, components = _motion_update(coordinates, scores, normals)
        scores = _motion_scores(coordinates - mean, components, normals)
        objective.append(error(mean, components, scores))
    # The gauge: orthonormal components, centred scores, by falling variance. It leaves the fit unchanged.
    mean = mean + components @ scores.mean(axis=0)
    centred = scores - scores.mean(axis=0)
    left, values, right = np.linalg.svd(components @ centred.T @ centred @ components.T, hermitian=True)
    basis = left[:, :rank]
    return LowRankMotion(mean, basis, (centred @ components.T) @ basis, np.asarray(objective))


def low_rank_distance(motion: LowRankMotion, target: int) -> np.ndarray:
    """``|l_i - l_target|`` of every particle, angstroms, ``[N]``."""

    difference = motion.scores - motion.scores[target]
    return np.sqrt(np.sum(difference * difference, axis=1))


def motion_summary(features: PoseFeatures, motion: LowRankMotion, covariance: np.ndarray) -> dict:
    """How much of the fitted motion is rotation and how much translation, and the pose noise the fit leaves, without
    truth: per rotation axis in degrees, and per observed image direction in angstroms. Each particle's score absorbs
    part of its own noise, so the residuals are divided by the freedom they keep (their leverage is taken away). A
    rotation error ``theta`` costs ``theta^2 (2/3) tr C`` on average over axes. Unmodelled real motion counts as noise."""

    coordinates = np.hstack([features.shape_factors.reshape(len(features.points), -1), features.points])
    fitted = motion.scores @ motion.components.T
    rotation, translation = float(np.mean(np.sum(fitted[:, :9] ** 2, axis=1))), float(np.mean(np.sum(fitted[:, 9:] ** 2, axis=1)))
    residual = _depth_projected(coordinates - motion.mean - fitted, features.normals)
    components = motion.components
    along = features.normals @ components[9:]
    inverse = np.linalg.inv(components.T @ components - along[:, :, None] * along[:, None, :] + 1e-9 * np.eye(components.shape[1]))
    rotation_block = components[:9].T @ components[:9]
    translation_block = components[9:].T @ components[9:] - along[:, :, None] * along[:, None, :]
    rotation_leverage = np.mean(np.einsum("nab,ba->n", inverse, rotation_block))
    translation_leverage = np.mean(np.einsum("nab,nba->n", inverse, translation_block))
    shape = float(np.mean(np.sum(residual[:, :9] ** 2, axis=1))) / max(1.0 - rotation_leverage / 3.0, 1e-9)
    return {
        "rotation_share_of_motion": rotation / (rotation + translation),
        "motion_rms_angstrom": float(np.sqrt(rotation + translation)),
        "rotation_noise_deg_per_axis": float(np.rad2deg(np.sqrt(shape / (2.0 / 3.0 * np.trace(covariance)) / 3.0))),
        "translation_noise_angstrom_per_direction": float(np.sqrt(np.mean(np.sum(residual[:, 9:] ** 2, axis=1)) / max(2.0 - translation_leverage, 1e-9))),
    }
