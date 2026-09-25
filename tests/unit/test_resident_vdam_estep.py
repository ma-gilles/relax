"""The resident pass 2 in VDAM's (``--grad``) configuration, against the compact engine.

VDAM's E-step is auto-refine's with RELION's SGD backprojection of the residual
(``cuda_kernel_backproject3D_SGD``, BP.cuh:406-560) and without scale-correction
groups (the InitialModel command has no ``--scale``). Both are checked on the
production-shaped 8x8 fixture of ``test_resident_pass2_driver``.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_resident_zero_oversampling import _rel_l2, _resident_production_env, requires_resident_gpu  # noqa: F401

import relax.sparse_pass2.resident_pass2 as rp

pytestmark = pytest.mark.unit


def _vdam_args(*, residual: bool, groups: bool, seed=20260925):
    from test_resident_pass2_driver import _driver_fixture_args

    args = _driver_fixture_args(seed=seed)
    args["mstep_subtract_ctf_projection"] = bool(residual)
    if not groups:
        args.update(group_ids=None, scale_correction_group_count=None, scale_correction_data_vs_prior=None)
    return args


def test_block_residual_is_the_exact_local_statement():
    """``summed - (sum_t w) * ctf^2/sigma2 * proj``, zero where the row carries no mass."""

    rng = np.random.default_rng(0)
    rows, pixels, images = 5, 7, 3
    summed = rng.normal(size=(rows, pixels)) + 1j * rng.normal(size=(rows, pixels))
    proj = rng.normal(size=(rows, pixels)) + 1j * rng.normal(size=(rows, pixels))
    ctf2 = rng.random((images, pixels))
    row_image = np.array([0, 2, 1, 1, 0], dtype=np.int32)
    mass = np.array([0.5, 0.0, 1.0, 0.25, 2.0])
    out = np.asarray(
        rp._resident_block_residual(
            summed.astype(np.complex64),
            mass.astype(np.float32),
            proj.astype(np.complex64),
            ctf2.astype(np.float32),
            row_image,
        )
    )
    expected = summed - mass[:, None] * (proj * ctf2[row_image])
    # A float32 statement against a float64 reference: 4.7e-8 of the largest value measured.
    assert_matches(out, expected, rtol=1e-6)
    # A row without mass keeps its image sum untouched (no float computation reaches it).
    np.testing.assert_array_equal(out[1], summed[1].astype(np.complex64))


@requires_resident_gpu
def test_resident_residual_backprojection_matches_the_compact_engine(_resident_production_env):
    from relax.sparse_pass2.sparse_pass2_bucketed import compute_pass2_stats_sparse_bucketed

    args = _vdam_args(residual=True, groups=True)
    compact = compute_pass2_stats_sparse_bucketed(**args)
    resident = rp.compute_pass2_stats_resident(**args)
    plain = rp.compute_pass2_stats_resident(**_vdam_args(residual=False, groups=True))

    # The residual changes the numerator only; the weight is RELION's Fweight either way.
    assert _rel_l2(plain.Ft_y, resident.Ft_y) > 1e-3
    assert _rel_l2(plain.Ft_ctf, resident.Ft_ctf) < 1e-6
    np.testing.assert_array_equal(compact.hard_assignment, resident.hard_assignment)
    assert _rel_l2(compact.Ft_y, resident.Ft_y) < 1e-6
    assert _rel_l2(compact.Ft_ctf, resident.Ft_ctf) < 1e-6
    for field in ("wsum_sigma2_noise", "wsum_img_power"):
        assert _rel_l2(getattr(compact.noise_stats, field), getattr(resident.noise_stats, field)) < 1e-4, field


@requires_resident_gpu
def test_resident_without_scale_groups_keeps_the_scale_one_wavg(_resident_production_env):
    """No ``--scale``: RELION's Wavg still runs at scale 1 and only the XA/AA sums are skipped.

    The same pass with one scale group of scale 1 is therefore the reference for every
    output except the scale sums. The compact engine is not: without groups it takes its
    non-atomic noise arithmetic (19% apart on this fixture's algebraic Wavg branch, job
    14421464), where the production VDAM path, with RELION's exact BPref operands, agrees
    with the exact-local route (the k1-25 and k1-11 VDAM runs of job 14421464).
    """

    grouped = rp.compute_pass2_stats_resident(**_vdam_args(residual=True, groups=True))
    ungrouped = rp.compute_pass2_stats_resident(**_vdam_args(residual=True, groups=False))
    np.testing.assert_array_equal(grouped.hard_assignment, ungrouped.hard_assignment)
    assert _rel_l2(grouped.Ft_y, ungrouped.Ft_y) < 1e-6
    assert _rel_l2(grouped.Ft_ctf, ungrouped.Ft_ctf) < 1e-6
    for field in ("wsum_sigma2_noise", "wsum_img_power", "wsum_sigma2_offset"):
        assert _rel_l2(getattr(grouped.noise_stats, field), getattr(ungrouped.noise_stats, field)) < 1e-6, field
