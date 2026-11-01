"""A body's local hypothesis layout: RELION's local search over residual angles, its body prior and its rows.

A body is searched like an auto-refine local search (``relax.local_search.layout``) centred on the particle's
residual Euler angles with the run's ``sigma2_rot`` and ``sigma2_psi``. RELION multiplies that local prior by the
body's own prior on the deviation from the unmoved pose (``selectOrientationsWithNonZeroPriorProbability`` with
``sigma_tilt_from_ninety`` and ``sigma_psi_from_zero``, healpix_sampling.cpp at f2c1a38, called from
acc_ml_optimiser_impl.h:108-128): a Gaussian of the angle between a sampled direction and the ``(rot, tilt) =
(0, 90)`` direction, and of the sampled psi's distance from 0, each cut at 3 sigma. Both read the canonical
(unperturbed) HEALPix angles of the coarse direction and psi, as the local prior does, and an oversampled child
inherits its parent's prior.

The scored and backprojected rows are the composed body matrices ``MBL R MBR`` of each image
(:mod:`relax.multibody.geometry`); the layout keeps the residual angles as its canonical metadata, so the best
pose it reports is the residual pose RELION writes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from relax.local_search.layout import LocalHypothesisLayout, _rotation_eulers_from_grid_metadata
from relax.multibody.bodies import Body
from relax.sampling.grids import build_local_search_grid_metadata

SIGMA_CUTOFF = 3.0  # selectOrientationsWithNonZeroPriorProbability's sigma_cutoff in the multi-body call


def _log_gaussian1d(x, sigma):
    """log of RELION's ``gaussian1D(x, sigma, 0)``."""

    return -0.5 * (x / sigma) ** 2 - np.log(np.sqrt(2.0 * np.pi) * sigma)


def body_prior_log(eulers_deg: np.ndarray, body: Body) -> np.ndarray:
    """The body prior's log weight of each canonical ``(rot, tilt, psi)`` row; ``-inf`` beyond 3 sigma."""

    rot, tilt, psi = (np.deg2rad(eulers_deg[:, k]) if k < 2 else eulers_deg[:, k] for k in range(3))
    # Euler_angles2direction(rot, tilt) . (1, 0, 0), the direction of (0, 90).
    dot = np.cos(rot) * np.sin(tilt)
    diffang = np.abs(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
    diff_psi = np.where(psi > 180.0, psi - 360.0, np.where(psi < -180.0, psi + 360.0, psi))
    diff_psi = np.abs(diff_psi)
    log_prior = np.zeros(eulers_deg.shape[0])
    if body.sigma_tilt_deg > 0.0:
        log_prior += np.where(
            diffang > SIGMA_CUTOFF * body.sigma_tilt_deg, -np.inf, _log_gaussian1d(diffang, body.sigma_tilt_deg)
        )
    if body.sigma_psi_deg > 0.0:
        log_prior += np.where(
            diff_psi > SIGMA_CUTOFF * body.sigma_psi_deg, -np.inf, _log_gaussian1d(diff_psi, body.sigma_psi_deg)
        )
    return log_prior


def _take_rows(layout: LocalHypothesisLayout, keep: np.ndarray, **updates) -> LocalHypothesisLayout:
    offsets = np.asarray(layout.rotation_offsets, dtype=np.int64)
    counts = np.diff(offsets)
    row_image = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    new_counts = np.bincount(row_image[keep], minlength=counts.shape[0])
    if np.any(new_counts == 0):
        # RELION then keeps the direction nearest the local prior alone; a body prior that excludes a particle's
        # whole local search is outside the supported configuration.
        raise NotImplementedError(
            f"the body prior excludes every local orientation of image {int(np.flatnonzero(new_counts == 0)[0])}"
        )

    def take(values):
        return None if values is None else np.asarray(values)[keep]

    new_offsets = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    new_offsets[1:] = np.cumsum(new_counts, dtype=np.int64)
    fields = dict(
        rotation_offsets=new_offsets,
        rotation_ids_flat=take(layout.rotation_ids_flat),
        rotations_flat=take(layout.rotations_flat),
        rotation_log_priors_flat=take(layout.rotation_log_priors_flat),
        rotation_counts=new_counts.astype(np.asarray(layout.rotation_counts).dtype),
        rotation_posterior_ids_flat=take(layout.rotation_posterior_ids_flat),
        sample_mask_bits=take(layout.sample_mask_bits),
        mstep_rotations_flat=take(layout.mstep_rotations_flat),
        source_eulers_flat=take(layout.source_eulers_flat),
        row_class_flat=take(layout.row_class_flat),
    )
    fields.update({name: take(value) for name, value in updates.items()})
    return replace(layout, **fields)


def with_body_prior(layout: LocalHypothesisLayout, body: Body) -> LocalHypothesisLayout:
    """A local id-row layout (pass 1, or the single pass without oversampling) with the body prior applied:
    rows beyond the prior's cut removed and the prior's log weight added to the others."""

    if layout.oversampled_rows or layout.id_rows_source is None:
        raise ValueError("the body prior is applied to a layout of canonical grid rows, before oversampling")
    healpix_order = int(layout.id_rows_source[0])
    eulers = _rotation_eulers_from_grid_metadata(
        layout.rotation_ids_flat, build_local_search_grid_metadata(healpix_order, symmetry=layout.symmetry),
        dtype=np.float64,
    )
    log_prior = body_prior_log(eulers, body)
    keep = np.isfinite(log_prior)
    priors = np.asarray(layout.rotation_log_priors_flat)
    return _take_rows(layout, keep, rotation_log_priors_flat=(priors + log_prior.astype(priors.dtype)))


def with_body_rows(layout: LocalHypothesisLayout, left: np.ndarray, right: np.ndarray) -> LocalHypothesisLayout:
    """The layout's scoring and M-step rows composed as ``left[i] R right`` for the rows of image ``i``
    (``left`` ``[n_images, 3, 3]``, ``right`` ``[3, 3]``), in float64 and cast back."""

    counts = np.diff(np.asarray(layout.rotation_offsets, dtype=np.int64))
    row_left = np.repeat(np.asarray(left, dtype=np.float64), counts, axis=0)

    def compose(rows):
        if rows is None:
            return None
        rows = np.asarray(rows)
        return (row_left @ rows.astype(np.float64) @ np.asarray(right, dtype=np.float64)).astype(rows.dtype)

    return replace(
        layout, rotations_flat=compose(layout.rotations_flat), mstep_rotations_flat=compose(layout.mstep_rotations_flat)
    )


@dataclass(frozen=True)
class BodySearch:
    """One body's local search over one half's images (``LocalPriorSpec.body_search``): the body prior on the pass-1
    parents and the composition ``left[i] R right`` of every scored row, ``left`` in the half's image order."""

    body: Body
    left: np.ndarray  # [n_images, 3, 3] MBL
    right: np.ndarray  # [3, 3] MBR

    def parent_layout(self, layout: LocalHypothesisLayout) -> LocalHypothesisLayout:
        return with_body_prior(layout, self.body)

    def scored_rows(self, layout: LocalHypothesisLayout) -> LocalHypothesisLayout:
        return with_body_rows(layout, self.left, self.right)
