"""RELION's local angular searches for the mixture: each particle's candidate coarse poses around its stored pose.

relion_refine with ``orientational_prior_mode = PRIOR_ROTTILT_PSI`` keeps, for each particle, the grid
directions within ``3 sigma`` of the direction of its previous (rot, tilt) and the psi samples within
``3 sigma`` of its previous psi (``HealpixSampling::selectOrientationsWithNonZeroPriorProbability``,
healpix_sampling.cpp; ``sigma_cutoff`` 3). Auto-sampling sets ``sigma`` to twice the oversampled angular step
(``sigma2_rot = sigma2_tilt = sigma2_psi = (2 step)^2``, ml_optimiser.cpp ``updateAngularSampling``); the
mixture's order-1 children are that oversampling. Deviations from RELION, both documented in the README:
the kept poses carry the stage's flat angular prior instead of RELION's Gaussian prior weights (the streamed
engine's pose prior is per row, not per particle), and the translation grid stays centered on zero instead
of on the particle's previous shift (the reader shifts a tile's images by one grid), so every grid shift
within ``offset_range_px`` of the stored shift is a candidate.

The candidates are the ``significant_rows`` of :func:`relax.ppca_refinement.full_row_stream.tile_support`:
packed ``rotation * n_translations + translation`` ids of the coarse grid; ``None`` keeps every pose.
"""

from dataclasses import dataclass

import numpy as np

from relax.helpers.convergence import healpix_angular_step

SIGMA_CUTOFF = 3.0  # RELION's sigma_cutoff: +/- 3 standard deviations


def relion_local_sigma_deg(coarse_order: int) -> float:
    """RELION's local-search width for a coarse HEALPix order whose children are the oversampling: twice the
    children's angular step."""
    return 2.0 * healpix_angular_step(int(coarse_order) + 1)


def _directions(eulers_deg):
    """``Euler_angles2direction`` (euler.cpp): the unit vector of (rot, tilt)."""
    eulers = np.deg2rad(np.asarray(eulers_deg, np.float64).reshape(-1, 3))
    rot, tilt = eulers[:, 0], eulers[:, 1]
    return np.stack([np.sin(tilt) * np.cos(rot), np.sin(tilt) * np.sin(rot), np.cos(tilt)], axis=1)


@dataclass(frozen=True)
class LocalGrid:
    """A coarse grid prepared for candidate selection: its directions, psi angles and shift vectors."""

    directions: np.ndarray  # (R, 3)
    psi_deg: np.ndarray  # (R,)
    translations_px: np.ndarray  # (T, D)

    @classmethod
    def of(cls, grid):
        eulers = np.asarray(grid.eulers, np.float64)
        return cls(_directions(eulers), eulers[:, 2].copy(), np.asarray(grid.translations, np.float64))

    @property
    def n_translations(self) -> int:
        return int(self.translations_px.shape[0])


def candidate_rows(local: LocalGrid, euler_deg, translation_px, *, sigma_deg: float, offset_range_px=None):
    """Packed coarse rows of one stored pose: rotations whose direction lies within ``3 sigma`` of the stored
    direction and whose psi lies within ``3 sigma`` of the stored psi, times the grid shifts within
    ``offset_range_px`` (None: every shift) of the stored shift. An empty angular neighbourhood keeps the
    nearest direction's psi-nearest rotation, as RELION keeps the nearest sample."""
    cutoff = SIGMA_CUTOFF * float(sigma_deg)
    direction = _directions(np.asarray(euler_deg, np.float64)[None])[0]
    cosine = np.clip(local.directions @ direction, -1.0, 1.0)
    angle = np.rad2deg(np.arccos(cosine))
    psi = np.abs((local.psi_deg - float(euler_deg[2]) + 180.0) % 360.0 - 180.0)
    rotations = np.flatnonzero((angle <= cutoff) & (psi <= cutoff))
    if rotations.size == 0:
        nearest = np.flatnonzero(angle <= angle.min() + 1e-9)
        rotations = nearest[np.argmin(psi[nearest])][None]
    if offset_range_px is None:
        shifts = np.arange(local.n_translations)
    else:
        stored = np.asarray(translation_px, np.float64)[: local.translations_px.shape[1]]
        distance = np.linalg.norm(local.translations_px - stored, axis=1)
        shifts = np.flatnonzero(distance <= float(offset_range_px) + 1e-6)
        if shifts.size == 0:
            shifts = np.argmin(distance)[None]
    rows = rotations[:, None] * local.n_translations + shifts[None, :]
    return rows.reshape(-1).astype(np.int32)


def tile_candidates(local: LocalGrid, membership, particle_ids, *, sigma_deg: float, offset_range_px=None):
    """The ``significant_rows`` of a tile: one candidate row array per particle with a stored pose, ``None``
    (the full grid) for a particle evaluated never or without a pose."""
    rows, known = membership.known_poses(particle_ids)
    return [
        candidate_rows(local, membership.euler_deg[row], membership.translation_px[row],
                       sigma_deg=sigma_deg, offset_range_px=offset_range_px) if is_known else None
        for row, is_known in zip(rows, known, strict=True)
    ]
