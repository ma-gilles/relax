"""The resident pass 2 in VDAM's (``--grad``) configuration, against the compact engine.

VDAM's E-step is auto-refine's with RELION's SGD backprojection of the residual
(``cuda_kernel_backproject3D_SGD``, BP.cuh:406-560) and without scale-correction
groups (the InitialModel command has no ``--scale``). Both are checked on the
production-shaped 8x8 fixture of ``test_resident_pass2_driver``.
"""

import jax.numpy as jnp
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
@pytest.mark.usefixtures("_resident_production_env")
def test_resident_residual_backprojection_matches_the_compact_engine():
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
@pytest.mark.usefixtures("_resident_production_env")
def test_resident_without_scale_groups_keeps_the_scale_one_wavg():
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


@pytest.mark.parametrize("shape", [(64, 37), (4096, 1537)])
def test_native_fine_units_in_place_is_the_eager_conversion(shape):
    """On the default backend (CPU, or the Slurm GPU): a cache-sized block and a small one."""

    from relax.sparse_pass2.sparse_pass2_scoring import _relion_native_fine_units

    rng = np.random.default_rng(3)
    values = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64) * np.float32(1e3)
    eager = np.asarray(_relion_native_fine_units(values, 136 * 136))
    fused = np.asarray(rp._relion_native_fine_units_in_place(jnp.asarray(values), 136 * 136))
    assert fused.dtype == np.complex64
    # Division in binary64 and one rounding to float32 in both forms (measured: identical).
    assert_matches(fused, eager)


# ---------------------------------------------------------------------------
# CPU: VDAM's pseudo-halfset accumulator slots (docs/development/resident_segments.md)
# ---------------------------------------------------------------------------


def _two_class_tables(**fields):
    """Three images, two classes, rows image-major then class-major, every image full-mask."""

    from relax.sparse_pass2.resident_candidates import ResidentCandidateTables

    row_unit = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    row_class = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.int32)
    n_rows = row_unit.size
    return ResidentCandidateTables(
        n_images=3,
        n_rows=n_rows,
        n_fine_trans=4,
        n_coarse_trans=2,
        row_offsets=np.array([0, 3, 5, 9], dtype=np.int32),
        row_unit=row_unit,
        row_fine_rot=np.arange(n_rows, dtype=np.int32),
        row_parent_local=np.zeros(n_rows, dtype=np.int32),
        row_log_prior=np.zeros(n_rows, dtype=np.float32),
        mask_mode=np.zeros(3, dtype=np.int8),
        parent_offsets=np.zeros(4, dtype=np.int32),
        parent_trans_bits=np.zeros((0, 1), dtype=np.uint32),
        row_class=row_class,
        n_classes=2,
        **fields,
    )


def test_tables_write_each_row_its_slot():
    from relax.sparse_pass2.resident_candidates import CapacityChunk, materialize_chunk

    tables = _two_class_tables(unit_slot_offset=np.array([1, 0, 1], dtype=np.int32), n_slot_groups=2)
    assert tables.n_slots == 4
    chunk = CapacityChunk(image_start=0, image_stop=3, row_start=0, row_stop=9, row_capacity=12, image_capacity=4)
    host = materialize_chunk(tables, chunk)
    expected = tables.row_class + 2 * np.array([1, 0, 1], dtype=np.int32)[tables.row_unit]
    np.testing.assert_array_equal(host["row_slot"][:9], expected)
    np.testing.assert_array_equal(host["row_slot"][9:], 0)
    plain = materialize_chunk(_two_class_tables(), chunk)
    np.testing.assert_array_equal(plain["row_slot"][:9], _two_class_tables().row_class)


def test_tables_refuse_inconsistent_slot_groups():
    with pytest.raises(ValueError, match="unit_slot_offset values"):
        _two_class_tables(unit_slot_offset=np.array([0, 2, 1], dtype=np.int32), n_slot_groups=2)
    with pytest.raises(ValueError, match="need unit_slot_offset"):
        _two_class_tables(n_slot_groups=2)
    with pytest.raises(ValueError, match="shape"):
        _two_class_tables(unit_slot_offset=np.zeros(2, dtype=np.int32), n_slot_groups=2)


def test_reconstruction_groups_become_unit_slot_offsets():
    tables = rp._with_reconstruction_groups(_two_class_tables(), [1, 0, 1], 2, n_images=3)
    np.testing.assert_array_equal(tables.unit_slot_offset, [1, 0, 1])
    assert tables.n_slots == 4
    assert rp._with_reconstruction_groups(_two_class_tables(), None, None, n_images=3).n_slots == 2
    with pytest.raises(ValueError, match="go together"):
        rp._with_reconstruction_groups(_two_class_tables(), [0, 1, 0], None, n_images=3)
    with pytest.raises(ValueError, match="one entry per image"):
        rp._with_reconstruction_groups(_two_class_tables(), [0, 1], 2, n_images=3)


def test_class_accumulators_stack_the_pseudo_halfsets():
    """Slot ``k + K * h`` becomes group ``h`` of class ``k``: the exact-local engine's grouped layout."""

    volumes = tuple(np.full(3, float(a)) for a in range(4))
    result = rp._ResidentPass2Result(
        Ft_y=volumes,
        Ft_ctf=tuple(v + 10.0 for v in volumes),
        n_classes=2,
        n_slot_groups=2,
        finalized=None,
        noise_stats=None,
        fine_translations=None,
        score_real_dtype=np.float32,
    )
    y1, ctf1 = rp._class_accumulators(result, 1)
    np.testing.assert_array_equal(np.asarray(y1)[:, 0], [1.0, 3.0])
    np.testing.assert_array_equal(np.asarray(ctf1)[:, 0], [11.0, 13.0])
    single = result._replace(Ft_y=volumes[:2], Ft_ctf=volumes[:2], n_slot_groups=1)
    assert np.asarray(rp._class_accumulators(single, 1)[0])[0] == 1.0


def test_compact_pass_refuses_reconstruction_groups(monkeypatch):
    """The compact engine has one BPref pair per class; groups must not merge silently."""

    from test_resident_pass2_driver import _driver_fixture_args

    from relax.sparse_pass2.dispatch import compute_pass2_stats_sparse

    monkeypatch.setenv("RELAX_SPARSE_PASS2_RESIDENT", "0")
    args = _driver_fixture_args()
    n_images = len(args["significant_sample_indices"])
    args["mean_variance"] = None
    with pytest.raises(NotImplementedError, match="device-resident pass 2"):
        compute_pass2_stats_sparse(
            **args,
            reconstruction_group_ids=np.arange(n_images) % 2,
            reconstruction_group_count=2,
        )
