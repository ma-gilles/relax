"""RELION's prior on the tilt angle alone (``--sigma_tilt``): which orientations a particle keeps and their prior.

The reference is a direct transcription of ``HealpixSampling::selectOrientationsWithNonZeroPriorProbability``
(RELION 5.0.1 healpix_sampling.cpp:695-1068) for ``sigma_rot == sigma_psi == 0 < sigma_tilt``, one direction
and one in-plane angle at a time in plain NumPy scalars: it shares no code with the selector under test
(relax.local_search.layout). The sampling grid and RELION's symmetry operators are its inputs.
"""

import math

import numpy as np
import pytest

from relax.local_search.layout import _build_factorized_local_entries, build_local_hypothesis_layout
from relax.sampling.grids import build_local_search_grid_metadata
from relax.sampling.symmetry import relion_symmetry_operators


def _gaussian1d(x, sigma):
    return 1.0 / math.sqrt(2.0 * math.pi * sigma * sigma) * math.exp(-0.5 * (x / sigma) ** 2)


def relion_tilt_prior_selection(prior_tilt, sigma_tilt, rot_angles, tilt_angles, psi_angles, operators, sigma_cutoff=3.0):
    """``(pointer_dir_nonzeroprior, directions_prior, pointer_psi_nonzeroprior, psi_prior)`` as RELION fills them."""

    pointer_dir, directions_prior = [], []
    sumprior, best_ang, best_idir = 0.0, 9999.0, -999
    for idir in range(len(rot_angles)):
        rot, tilt = math.radians(rot_angles[idir]), math.radians(tilt_angles[idir])
        # Euler_angles2direction
        my_direction = np.array([math.sin(tilt) * math.cos(rot), math.sin(tilt) * math.sin(rot), math.cos(tilt)])
        diffang = abs(tilt_angles[idir] - prior_tilt)
        if diffang > 180.0:
            diffang = abs(diffang - 360.0)
        best_diffang = diffang
        for left, right in operators:
            sym_direction = left @ (my_direction @ right)  # L * (d^T R)^T
            sym_direction = sym_direction / np.linalg.norm(sym_direction)
            sym_tilt = math.degrees(math.acos(min(1.0, max(-1.0, sym_direction[2]))))  # Euler_direction2angles
            diffang = abs(sym_tilt - prior_tilt)
            if diffang > 180.0:
                diffang = abs(diffang - 360.0)
            if diffang < best_diffang:
                best_diffang = diffang
        if best_diffang < sigma_cutoff * sigma_tilt:
            prior = _gaussian1d(best_diffang, sigma_tilt)
            pointer_dir.append(idir)
            directions_prior.append(prior)
            sumprior += prior
        if best_diffang < best_ang:
            best_idir = idir
            best_ang = diffang  # RELION stores the last mate's distance here, not the smallest
    directions_prior = [prior / sumprior for prior in directions_prior]
    if not directions_prior:
        pointer_dir, directions_prior = [best_idir], [1.0]
    pointer_psi = list(range(len(psi_angles)))
    psi_prior = [1.0 / len(psi_angles)] * len(psi_angles)
    return pointer_dir, directions_prior, pointer_psi, psi_prior


def _reference_entries(prior_tilts, sigma_tilt_deg, metadata, symmetry):
    left, right = relion_symmetry_operators(symmetry)
    operators = list(zip(left[1:], right[1:]))  # RELION's R_repository has no identity
    n_pixels = int(metadata["n_pixels"])
    ids, log_priors, counts = [], [], []
    for prior_tilt in prior_tilts:
        dirs, dir_prior, psis, psi_prior = relion_tilt_prior_selection(
            float(prior_tilt), sigma_tilt_deg, metadata["rot_deg"], metadata["tilt_deg"], metadata["psi_deg"], operators
        )
        # relax numbers an orientation psi * n_pixels + direction, psi-major.
        ids += [ipsi * n_pixels + idir for ipsi in psis for idir in dirs]
        log_priors += [math.log(p_psi) + math.log(p_dir) for p_psi in psi_prior for p_dir in dir_prior]
        counts.append(len(dirs) * len(psis))
    return np.asarray(ids), np.asarray(log_priors), np.asarray(counts)


PRIOR_TILTS = np.array([90.0, 0.0, 180.0, 37.3, 100.0, 171.9, -20.0, 200.0, 999.0 - 900.0])


@pytest.mark.unit
@pytest.mark.parametrize("symmetry", ["C1", "C6", "D2", "O"])
@pytest.mark.parametrize("healpix_order, sigma_tilt_deg", [(1, 10.0), (2, 10.0), (2, 3.0), (2, 25.0), (3, 5.0)])
def test_tilt_prior_selection_is_relions(symmetry, healpix_order, sigma_tilt_deg):
    metadata = build_local_search_grid_metadata(healpix_order, symmetry=symmetry)
    prior_eulers = np.column_stack([np.linspace(-170.0, 170.0, PRIOR_TILTS.size), PRIOR_TILTS, np.zeros(PRIOR_TILTS.size)])
    offsets, counts, ids, log_priors = _build_factorized_local_entries(
        prior_eulers, healpix_order, 0.0, 0.0, metadata, dtype=np.float64,
        sigma_tilt_deg=sigma_tilt_deg, prior_tilt_deg=PRIOR_TILTS,
    )  # fmt: skip
    ref_ids, ref_log_priors, ref_counts = _reference_entries(PRIOR_TILTS, sigma_tilt_deg, metadata, symmetry)
    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_array_equal(np.diff(offsets), ref_counts)
    np.testing.assert_array_equal(ids, ref_ids)
    # The selector returns float32 log priors (one rounding of the float64 value).
    np.testing.assert_allclose(log_priors, ref_log_priors, rtol=0.0, atol=2e-6)
    n_psi = int(metadata["n_psi"])
    assert np.all(counts % n_psi == 0) and np.all(counts > 0)
    for start, stop in zip(offsets[:-1], offsets[1:]):  # each particle's prior sums to 1 over its samples
        assert abs(np.exp(log_priors[start:stop].astype(np.float64)).sum() - 1.0) < 1e-5


@pytest.mark.unit
@pytest.mark.parametrize("symmetry", ["C1", "C6", "D2"])
def test_width_below_the_grid_spacing_keeps_relions_single_direction(symmetry):
    """No direction within three widths: RELION keeps the direction its scan ends on, with prior 1."""

    metadata = build_local_search_grid_metadata(1, symmetry=symmetry)
    prior_tilts = np.array([33.0, 90.0, 12.0, 147.0, 65.0])
    eulers = np.column_stack([np.zeros(5), prior_tilts, np.zeros(5)])
    offsets, counts, ids, log_priors = _build_factorized_local_entries(
        eulers, 1, 0.0, 0.0, metadata, dtype=np.float64, sigma_tilt_deg=0.01, prior_tilt_deg=prior_tilts
    )
    ref_ids, ref_log_priors, ref_counts = _reference_entries(prior_tilts, 0.01, metadata, symmetry)
    np.testing.assert_array_equal(counts, ref_counts)
    np.testing.assert_array_equal(ids, ref_ids)
    np.testing.assert_allclose(log_priors, ref_log_priors, rtol=0.0, atol=2e-6)


@pytest.mark.unit
def test_tilt_prior_does_not_depend_on_the_particles_rot_and_psi():
    metadata = build_local_search_grid_metadata(2)
    tilts = np.array([90.0, 60.0])
    entries = [
        _build_factorized_local_entries(
            np.column_stack([rot, tilts, psi]), 2, 0.0, 0.0, metadata, sigma_tilt_deg=10.0, prior_tilt_deg=tilts
        )
        for rot, psi in ((np.zeros(2), np.zeros(2)), (np.array([77.0, -140.0]), np.array([33.0, 310.0])))
    ]
    for a, b in zip(*entries):
        np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_layout_children_inherit_the_parent_tilt_prior():
    """With adaptive oversampling the parents are selected on the coarse grid and each child carries its
    parent's prior (RELION's getOrientations on the selected parents)."""

    tilts = np.array([90.0, 45.0])
    eulers = np.column_stack([np.array([10.0, 20.0]), tilts, np.array([5.0, 6.0])])
    common = dict(grid_metadata=build_local_search_grid_metadata(2), sigma_tilt_deg=10.0, prior_tilt_deg=tilts)
    flat = (np.zeros((1, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), 1.0, None, 4.0)
    parents = build_local_hypothesis_layout(eulers, None, 0.0, 0.0, 2, *flat, **common)
    children = build_local_hypothesis_layout(eulers, None, 0.0, 0.0, 3, *flat, local_parent_oversampling_order=1, **common)
    np.testing.assert_array_equal(children.rotation_counts, 8 * parents.rotation_counts)
    for image in range(2):
        parent = np.asarray(parents.rotation_log_priors_flat)[parents.rotation_offsets[image] : parents.rotation_offsets[image + 1]]
        child = np.asarray(children.rotation_log_priors_flat)[children.rotation_offsets[image] : children.rotation_offsets[image + 1]]
        np.testing.assert_array_equal(np.unique(child), np.unique(parent))


@pytest.mark.unit
def test_tilt_prior_needs_its_inputs():
    metadata = build_local_search_grid_metadata(1)
    eulers = np.zeros((2, 3))
    with pytest.raises(ValueError, match="one prior tilt angle per image"):
        _build_factorized_local_entries(eulers, 1, 0.0, 0.0, metadata, sigma_tilt_deg=6.0)
    with pytest.raises(NotImplementedError, match="separate rot prior"):
        _build_factorized_local_entries(eulers, 1, 0.2, 0.2, metadata, sigma_tilt_deg=6.0, prior_tilt_deg=np.zeros(2))
