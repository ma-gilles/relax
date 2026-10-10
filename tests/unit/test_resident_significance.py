"""Tests for the device-compacted coarse significance path (T13, CPU only).

Ticket: `em_parity_tickets_20260918/T13_device_significance_compaction.md`.

The host path is the oracle in every test here: the compacted CSR must equal
`compact_significant_sample_indices_from_mask` per image bitwise, and the
candidate tables built from that CSR must equal, field by field, the tables
built through `_prepare_per_image_pass2_inputs` for the same support.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.scoring.significant_samples import (
    ComplementSignificantSampleIndices,
    compact_significant_sample_indices_from_mask,
)
from relax.scoring.sparse_bucket_arrays import (
    _prepare_per_image_pass2_inputs,
    relion_parent_execution_key,
)
from relax.sparse_pass2.resident_candidates import build_resident_candidate_tables
from relax.sparse_pass2.resident_significance import (
    CoarseSignificanceCSR,
    DeviceCompactedSignificantSamples,
    PackedCoarseSignificanceCSR,
    PackedSignificanceBatch,
    build_coarse_significance_csr,
    build_resident_candidate_tables_from_csr,
    compact_batch_significance,
    compact_batch_significance_classes,
    csr_candidate_rows_per_image,
    csr_capacity_for_total,
    csr_restricted_to_images,
    fine_rotation_children,
    host_support_rows,
    significant_coarse_parents,
)

pytestmark = pytest.mark.unit

# Override fixture: a fine rotation grid given explicitly, as the adaptive
# K-class route supplies it (k_class.py passes fine_rotations_override and
# fine_rotation_parent_override into pass 2). RELION's parent execution order
# decomposes a coarse id into (direction, psi) with n_psi = 6 at healpix
# level 0, so the coarse grid is whole psi rows: 3 directions x 6 psi.
N_COARSE_ROT = 18
CHILDREN = 4
N_COARSE_TRANS = 5
N_FINE_TRANS = 15
FINE_TRANS_PARENT = np.repeat(np.arange(N_COARSE_TRANS, dtype=np.int32), 3)

# Standard fixture: no override, so both paths call the sampling generator.
# healpix level 0 has 12 pixels and 6 in-plane angles.
STD_NSIDE_LEVEL = 0
STD_N_COARSE_ROT = 72
STD_OVERSAMPLING = 1


def _z_rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _fine_rotation_override():
    """A fine grid whose parents are deliberately not in ascending order."""

    parent = np.concatenate(
        [
            np.repeat(np.arange(N_COARSE_ROT // 2, N_COARSE_ROT, dtype=np.int64), CHILDREN),
            np.repeat(np.arange(0, N_COARSE_ROT // 2, dtype=np.int64), CHILDREN),
        ],
    )
    rotations = np.stack([_z_rotation(0.01 * k) for k in range(parent.size)]).astype(np.float32)
    return rotations, parent


def _supports(n_images: int, n_samples: int, seed: int = 13) -> list[np.ndarray]:
    """Per-image significant coarse cell ids, one image of every regime.

    Image 1 has no support, image 2 is dense (the host encoder stores its
    complement), image 3 is fully significant (the host encoder stores
    ``None``), image 4 sits exactly on the threshold where the rule keeps the
    included ids, and the rest are ordinary sparse supports.
    """

    rng = np.random.default_rng(seed)
    supports = []
    for image in range(n_images):
        if image == 1:
            supports.append(np.zeros(0, dtype=np.int32))
            continue
        if image == 2 and n_images > 2:
            count = n_samples - max(1, n_samples // 8)
        elif image == 3 and n_images > 3:
            count = n_samples
        elif image == 4 and n_images > 4:
            count = n_samples // 2  # included == excluded: keep the included ids
        else:
            count = int(rng.integers(1, max(2, n_samples // 4)))
        ids = rng.choice(n_samples, size=count, replace=False)
        supports.append(np.sort(ids).astype(np.int32))
    return supports


def _mask_from_supports(supports, n_samples: int) -> np.ndarray:
    mask = np.zeros((len(supports), n_samples), dtype=bool)
    for row, ids in enumerate(supports):
        mask[row, np.asarray(ids, dtype=np.int64)] = True
    return mask


def _encoded_supports(supports, n_samples: int) -> list:
    """The host encoder's output for these supports, as production produces it.

    ``_prepare_per_image_pass2_inputs`` never sees raw id lists in production:
    it sees whatever
    :func:`compact_significant_sample_indices_from_mask` encoded, which is
    ``None`` for a full support and a sparse complement for a dense one.
    """

    mask = _mask_from_supports(supports, n_samples)
    return [compact_significant_sample_indices_from_mask(mask[i]) for i in range(mask.shape[0])]


def _csr_from_supports(supports, *, n_coarse_rot, n_coarse_trans) -> CoarseSignificanceCSR:
    """Build the CSR the device path would build for these supports."""

    n_samples = n_coarse_rot * n_coarse_trans
    n_significant = np.asarray([np.asarray(s).size for s in supports], dtype=np.int32)
    store_excluded = (n_significant.astype(np.int64) * 2) > n_samples
    stored = []
    for image, ids in enumerate(supports):
        ids = np.asarray(ids, dtype=np.int64)
        if store_excluded[image]:
            keep = np.ones(n_samples, dtype=bool)
            keep[ids] = False
            stored.append(np.flatnonzero(keep).astype(np.int32))
        else:
            stored.append(ids.astype(np.int32))
    return build_coarse_significance_csr(
        n_images=len(supports),
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_coarse_trans,
        n_significant_per_batch=[n_significant],
        store_excluded_per_batch=[store_excluded],
        ids_per_batch=[np.concatenate(stored) if stored else np.zeros(0, np.int32)],
    )


def _batch_ids(result, *, n_coarse_rot, n_coarse_trans) -> np.ndarray:
    """One batch's stored ids, image-major, whether the compaction returned them as ids or packed its dense rows."""

    n_significant, store_excluded, ids, _rot_any = result
    csr = build_coarse_significance_csr(
        n_images=n_significant.size, n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans,
        n_significant_per_batch=[n_significant], store_excluded_per_batch=[store_excluded], ids_per_batch=[ids],
    )
    return csr.ids_of_images(0, csr.n_images)


# --- The device compaction reproduces the host encoding --------------------


def test_compaction_matches_flatnonzero_per_image_bitwise():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(9, n_samples)
    mask = _mask_from_supports(supports, n_samples)
    counts_host = mask.sum(axis=1).astype(np.int32)

    n_significant, store_excluded, ids, rot_any = compact_batch_significance(
        mask,
        actual_batch_size=mask.shape[0],
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        batch_n_sig=counts_host,
    )
    assert_matches(n_significant, counts_host)
    csr = build_coarse_significance_csr(
        n_images=mask.shape[0],
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        n_significant_per_batch=[n_significant],
        store_excluded_per_batch=[store_excluded],
        ids_per_batch=[ids],
    )
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    for image in range(mask.shape[0]):
        expected = (
            np.flatnonzero(~mask[image]) if store_excluded[image] else np.flatnonzero(mask[image])
        )
        assert_matches(csr.image_ids(image), expected.astype(np.int32))
        # The stored set is always the smaller one, exactly as the host rule picks it.
        assert bool(store_excluded[image]) == (int(counts_host[image]) * 2 > n_samples)
    assert_matches(
        rot_any,
        mask.reshape(mask.shape[0], N_COARSE_ROT, N_COARSE_TRANS).any(axis=(0, 2)),
    )


def test_class_compaction_matches_each_class_alone(monkeypatch):
    """A class-major K-class mask compacts to each class's single-class result.

    A one-class read-back group exercises the grouping as well as one group.
    """

    import relax.sparse_pass2.resident_significance as resident_significance

    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    n_classes, actual = 3, 5
    masks = [_mask_from_supports(_supports(7, n_samples), n_samples) for _ in range(n_classes)]
    masks[1][2] = True  # a dense image stores its excluded cells
    class_major = np.concatenate(masks, axis=1)
    expected = [
        compact_batch_significance(
            mask,
            actual_batch_size=actual,
            n_coarse_rot=N_COARSE_ROT,
            n_coarse_trans=N_COARSE_TRANS,
            batch_n_sig=mask.sum(axis=1).astype(np.int32),
        )
        for mask in masks
    ]
    for pending_bytes in (resident_significance._PENDING_CLASS_ID_BYTES, 1):
        monkeypatch.setattr(resident_significance, "_PENDING_CLASS_ID_BYTES", pending_bytes)
        got = compact_batch_significance_classes(
            class_major,
            n_classes=n_classes,
            actual_batch_size=actual,
            n_coarse_rot=N_COARSE_ROT,
            n_coarse_trans=N_COARSE_TRANS,
        )
        assert len(got) == n_classes
        grid = dict(n_coarse_rot=N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS)
        for class_got, class_expected in zip(got, expected):
            for index in (0, 1, 3):
                assert_matches(class_got[index], class_expected[index])
            assert type(class_got[2]) is type(class_expected[2])
            assert_matches(_batch_ids(class_got, **grid), _batch_ids(class_expected, **grid))


def test_compaction_ignores_padded_image_rows():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(6, n_samples)
    mask = _mask_from_supports(supports, n_samples)
    actual = 4
    counts_host = mask[:actual].sum(axis=1).astype(np.int32)

    n_significant, store_excluded, ids, _rot_any = compact_batch_significance(
        mask,
        actual_batch_size=actual,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        batch_n_sig=mask.sum(axis=1).astype(np.int32),
    )
    assert_matches(n_significant, counts_host)
    csr = build_coarse_significance_csr(
        n_images=actual,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        n_significant_per_batch=[n_significant],
        store_excluded_per_batch=[store_excluded],
        ids_per_batch=[ids],
    )
    for image in range(actual):
        expected = (
            np.flatnonzero(~mask[image]) if store_excluded[image] else np.flatnonzero(mask[image])
        )
        assert_matches(csr.image_ids(image), expected.astype(np.int32))


def test_host_support_rows_match_the_host_encoder():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(7, n_samples)
    mask = _mask_from_supports(supports, n_samples)
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    rows = host_support_rows(csr)
    for image in range(mask.shape[0]):
        expected = compact_significant_sample_indices_from_mask(mask[image])
        got = rows[image]
        assert type(got) is type(expected), image
        if expected is None:
            continue
        if isinstance(expected, ComplementSignificantSampleIndices):
            assert_matches(got.excluded_indices, expected.excluded_indices)
            assert int(got.total_size) == int(expected.total_size)
            assert np.asarray(got.excluded_indices).dtype == np.int32
            continue
        assert_matches(np.asarray(got), np.asarray(expected))
        assert np.asarray(got).dtype == np.int32


@pytest.mark.parametrize("actual", [7, 5])
def test_per_class_compaction_of_a_joint_mask_matches_the_host_encoder(actual):
    """K>1: each class-major slice of the joint support compacts as the host encodes it.

    Mirrors the coarse pass: the joint ``[batch, K * n_rot * n_trans]`` device mask
    is split per class, each class counts its own support, and the padded tail of
    a short batch is ignored.
    """

    import jax.numpy as jnp

    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    n_classes = 3
    class_masks = [
        _mask_from_supports(_supports(7, n_samples, seed=41 + k), n_samples)[np.roll(np.arange(7), k)]
        for k in range(n_classes)
    ]
    joint = jnp.asarray(np.concatenate(class_masks, axis=1))
    per_class = joint.reshape(joint.shape[0], n_classes, n_samples)
    for k in range(n_classes):
        class_mask = per_class[:, k, :]
        n_significant, store_excluded, ids, _rot_any = compact_batch_significance(
            class_mask,
            actual_batch_size=actual,
            n_coarse_rot=N_COARSE_ROT,
            n_coarse_trans=N_COARSE_TRANS,
            batch_n_sig=jnp.sum(class_mask, axis=1, dtype=jnp.int32),
        )
        rows = host_support_rows(
            build_coarse_significance_csr(
                n_images=actual,
                n_coarse_rot=N_COARSE_ROT,
                n_coarse_trans=N_COARSE_TRANS,
                n_significant_per_batch=[n_significant],
                store_excluded_per_batch=[store_excluded],
                ids_per_batch=[ids],
            )
        )
        for image in range(actual):
            expected = compact_significant_sample_indices_from_mask(class_masks[k][image])
            got = rows[image]
            assert type(got) is type(expected), (k, image)
            if expected is None:
                continue
            if isinstance(expected, ComplementSignificantSampleIndices):
                assert_matches(got.excluded_indices, expected.excluded_indices, strict=True)
                continue
            assert_matches(np.asarray(got), np.asarray(expected), strict=True)


def test_compaction_stores_the_complement_of_a_dense_support():
    """A dense support is stored as its complement, as the host encoder does."""

    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    mask = np.zeros((2, n_samples), dtype=bool)
    mask[0, : n_samples // 2 + 1] = True  # just over the threshold
    mask[1, : n_samples // 2] = True  # exactly on it: keep the included ids
    result = compact_batch_significance(
        mask,
        actual_batch_size=2,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        batch_n_sig=mask.sum(axis=1).astype(np.int32),
    )
    n_significant, store_excluded, _ids, _rot_any = result
    ids = _batch_ids(result, n_coarse_rot=N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS)
    assert list(store_excluded) == [True, False]
    assert_matches(n_significant, mask.sum(axis=1).astype(np.int32))
    split = n_samples - int(n_significant[0])
    assert_matches(ids[:split], np.flatnonzero(~mask[0]).astype(np.int32))
    assert_matches(ids[split:], np.flatnonzero(mask[1]).astype(np.int32))


def test_capacity_ladder_is_power_of_two_and_covers_the_total():
    for total in (0, 1, 4095, 4096, 4097, 100000):
        capacity = csr_capacity_for_total(total)
        assert capacity >= max(total, 4096)
        assert capacity & (capacity - 1) == 0


def test_support_list_carries_its_csr_and_stays_a_list():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(5, n_samples)
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    rows = DeviceCompactedSignificantSamples(host_support_rows(csr), csr=csr)
    assert isinstance(rows, list)
    assert len(rows) == len(supports)
    expected = _encoded_supports(supports, n_samples)
    for image, got in enumerate(rows):
        want = expected[image]
        if want is None:
            assert got is None
        elif isinstance(want, ComplementSignificantSampleIndices):
            assert_matches(got.excluded_indices, want.excluded_indices)
        else:
            assert_matches(np.asarray(got), np.asarray(want))
    assert rows.csr is csr
    with pytest.raises(ValueError, match="CSR covers"):
        DeviceCompactedSignificantSamples(rows[:-1], csr=csr)


# --- The candidate tables equal the host path's, field by field ------------


def _assert_tables_equal(got, expected):
    assert got.n_images == expected.n_images
    assert got.n_rows == expected.n_rows
    assert got.n_fine_trans == expected.n_fine_trans
    assert got.n_coarse_trans == expected.n_coarse_trans
    for name in (
        "row_offsets",
        "row_unit",
        "row_fine_rot",
        "row_parent_local",
        "mask_mode",
        "parent_offsets",
        "parent_trans_bits",
    ):
        got_value = getattr(got, name)
        expected_value = getattr(expected, name)
        assert got_value.dtype == expected_value.dtype, name
        assert_matches(got_value, expected_value, err_msg=name)
    assert_matches(got.row_log_prior, expected.row_log_prior)


@pytest.mark.parametrize("execution_order", [False, True])
def test_tables_from_csr_match_the_host_path_with_a_fine_grid_override(execution_order):
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(11, n_samples)
    fine_rotations, fine_parent = _fine_rotation_override()
    rotation_log_prior = np.linspace(-2.0, 2.0, N_COARSE_ROT, dtype=np.float32)

    per_image_inputs = _prepare_per_image_pass2_inputs(
        _encoded_supports(supports, n_samples),
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
    )
    expected = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    got = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
    )
    _assert_tables_equal(got, expected)


@pytest.mark.parametrize("execution_order", [False, True])
def test_tables_from_csr_match_the_host_path_on_the_generated_grid(execution_order):
    n_samples = STD_N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(9, n_samples, seed=7)
    rotation_log_prior = np.linspace(-1.0, 1.0, STD_N_COARSE_ROT, dtype=np.float32)

    per_image_inputs = _prepare_per_image_pass2_inputs(
        _encoded_supports(supports, n_samples),
        n_coarse_rot=STD_N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=STD_OVERSAMPLING,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
    )
    expected = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=STD_N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    got = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=STD_OVERSAMPLING,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=rotation_log_prior,
        random_perturbation=0.0,
        relion_parent_execution_order=execution_order,
        dtype=np.float32,
    )
    _assert_tables_equal(got, expected)


def test_tables_from_csr_match_the_host_path_without_a_rotation_prior():
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(6, n_samples, seed=3)
    fine_rotations, fine_parent = _fine_rotation_override()

    per_image_inputs = _prepare_per_image_pass2_inputs(
        _encoded_supports(supports, n_samples),
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=fine_rotations,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=True,
        dtype=np.float32,
    )
    expected = build_resident_candidate_tables(
        per_image_inputs,
        n_coarse_trans=N_COARSE_TRANS,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
    )
    csr = _csr_from_supports(
        supports,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
    )
    got = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=True,
        dtype=np.float32,
    )
    _assert_tables_equal(got, expected)


def test_compaction_fills_an_exactly_full_capacity():
    """A total support equal to the id capacity must not be corrupted.

    Cells outside the support scatter with an out-of-bounds index and are
    dropped.  When the total exactly fills the buffer there is no slack left,
    so this is the case that would expose a dropped index being wrapped to the
    last slot instead.
    """

    n_coarse_rot, n_coarse_trans = 2048, 8
    n_samples = n_coarse_rot * n_coarse_trans
    capacity = csr_capacity_for_total(0)
    rng = np.random.default_rng(5)
    ids = np.sort(rng.choice(n_samples, size=capacity, replace=False)).astype(np.int32)
    mask = np.zeros((2, n_samples), dtype=bool)
    mask[0, ids.astype(np.int64)] = True

    from relax.sparse_pass2.resident_significance import _compact_program

    # The program itself: the batch entry point keeps a row this dense as its bit mask and drops the ids.
    compacted, counts, _rot_any, packed = _compact_program()(
        mask, 0, np.ones(2, dtype=bool), np.zeros(2, dtype=bool),
        capacity=capacity, n_coarse_trans=n_coarse_trans, n_classes=1,
    )
    compacted = np.asarray(compacted)
    assert compacted.size == capacity and list(np.asarray(counts)) == [capacity, 0]
    assert_matches(compacted, ids)
    # The packed rows are the same stored set, bit ``id`` of the row little-endian within a byte.
    np.testing.assert_array_equal(np.unpackbits(np.asarray(packed), axis=1, bitorder="little"), mask)


def test_relion_parent_execution_key_uses_the_grid_direction_count():
    """One owner for the host rows and the resident CSR tables.

    A full C1 grid keeps the HEALPix pixel count; a symmetry-reduced grid (the
    same psi count, fewer directions) must use its own direction count, and a
    grid that is not whole psi rows cannot be decomposed at all.
    """

    full_ids = np.arange(72, dtype=np.int64)  # level 0: 12 directions x 6 psi
    assert_matches(
        relion_parent_execution_key(full_ids, n_coarse_rot=72, nside_level=0),
        (full_ids % 12) * 6 + full_ids // 12,
    )
    reduced_ids = np.arange(18, dtype=np.int64)  # 3 directions x 6 psi
    reduced = relion_parent_execution_key(reduced_ids, n_coarse_rot=18, nside_level=0)
    assert_matches(reduced, (reduced_ids % 3) * 6 + reduced_ids // 3)
    assert sorted(reduced.tolist()) == list(range(18))
    with pytest.raises(ValueError, match="whole psi rows"):
        relion_parent_execution_key(np.arange(16), n_coarse_rot=16, nside_level=0)
    with pytest.raises(ValueError, match="outside the coarse grid"):
        relion_parent_execution_key(np.asarray([18]), n_coarse_rot=18, nside_level=0)


def test_fixture_covers_every_support_regime():
    """Guard the fixture: it must exercise all four host encodings.

    The device path has to reproduce every one of them, so a fixture that
    quietly stopped covering one would hide a gap like the dense-support case
    that failed the first end-to-end gate.
    """

    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(11, n_samples)
    encoded = _encoded_supports(supports, n_samples)
    assert any(e is None for e in encoded), "no full-support image"
    assert any(isinstance(e, ComplementSignificantSampleIndices) for e in encoded), "no dense image"
    assert any(isinstance(e, np.ndarray) and e.size == 0 for e in encoded), "no empty image"
    assert any(isinstance(e, np.ndarray) and e.size > 0 for e in encoded), "no sparse image"

    _rotations, fine_parent = _fine_rotation_override()
    per_image_inputs = _prepare_per_image_pass2_inputs(
        encoded,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=STD_NSIDE_LEVEL,
        oversampling_order=0,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotations_override=_fine_rotation_override()[0],
        fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=True,
        dtype=np.float32,
    )
    modes = {mask.mode for mask in per_image_inputs["candidate_mask"]}
    assert modes == {"coarse", "empty", "full", "coarse_exclude"}, (
        f"fixture built modes {modes}; update the fixture, not the assertion"
    )


def test_batches_of_one_shape_share_one_compaction_program():
    """The id capacity depends on the mask shape only, not on the batch's total.

    Totals on either side of a power of two used to key two programs for the
    same mask shape; now both batches reuse one and still compact exactly.
    """

    from relax.sparse_pass2.resident_significance import _compact_program

    n_coarse_rot, n_coarse_trans = 512, 16
    n_samples = n_coarse_rot * n_coarse_trans
    rng = np.random.default_rng(11)
    before = None
    for density in (0.01, 0.3):
        mask = rng.random((3, n_samples)) < density
        result = compact_batch_significance(
            mask,
            actual_batch_size=3,
            n_coarse_rot=n_coarse_rot,
            n_coarse_trans=n_coarse_trans,
            batch_n_sig=mask.sum(axis=1).astype(np.int32),
        )
        expected = np.concatenate([np.flatnonzero(row).astype(np.int32) for row in mask])
        assert not result[1].any()
        # The sparse batch comes back as ids, the dense one with its rows packed on the device.
        assert isinstance(result[2], PackedSignificanceBatch) == (density == 0.3)
        np.testing.assert_array_equal(
            _batch_ids(result, n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans), expected
        )
        size = _compact_program()._cache_size()
        if before is None:
            before = size
    assert _compact_program()._cache_size() == before


def test_seed_iteration_supports_keep_a_csr_whose_tables_match_the_host_path():
    # A seed iteration gives each image's first-class support to its random class only. With a device-compacted
    # first-class support every class keeps a CSR, so the resident pass builds its tables block by block; each
    # class's tables must be the host path's for that class's rows, empty images included.
    from relax.classification.k_class_inputs import seed_iteration_supports

    n_samples = STD_N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(9, n_samples, seed=11)
    csr = _csr_from_supports(supports, n_coarse_rot=STD_N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS)
    first_class = DeviceCompactedSignificantSamples(host_support_rows(csr), csr=csr)
    seeds = np.array([0, 2, 1, 2, 0, 1, 1, 2, 0])
    rotation_log_prior = np.linspace(-1.0, 1.0, STD_N_COARSE_ROT, dtype=np.float32)

    by_class = seed_iteration_supports(first_class, seeds, 3)

    assert [len(rows) for rows in by_class] == [9, 9, 9]
    for k, rows in enumerate(by_class):
        assert np.array_equal(rows.csr.n_significant > 0, (seeds == k) & (csr.n_significant > 0))
        expected = build_resident_candidate_tables(
            _prepare_per_image_pass2_inputs(
                list(rows),
                n_coarse_rot=STD_N_COARSE_ROT,
                n_coarse_trans=N_COARSE_TRANS,
                nside_level=STD_NSIDE_LEVEL,
                oversampling_order=STD_OVERSAMPLING,
                n_fine_trans=N_FINE_TRANS,
                fine_translation_parent=FINE_TRANS_PARENT,
                rotation_log_prior=rotation_log_prior,
                random_perturbation=0.0,
                relion_parent_execution_order=True,
                dtype=np.float32,
            ),
            n_coarse_trans=N_COARSE_TRANS,
            n_fine_trans=N_FINE_TRANS,
            fine_translation_parent=FINE_TRANS_PARENT,
        )
        got = build_resident_candidate_tables_from_csr(
            rows.csr,
            nside_level=STD_NSIDE_LEVEL,
            oversampling_order=STD_OVERSAMPLING,
            n_fine_trans=N_FINE_TRANS,
            fine_translation_parent=FINE_TRANS_PARENT,
            rotation_log_prior=rotation_log_prior,
            random_perturbation=0.0,
            relion_parent_execution_order=True,
            dtype=np.float32,
        )
        _assert_tables_equal(got, expected)


def test_seed_iteration_supports_without_a_csr_stay_plain_lists():
    from relax.classification.k_class_inputs import seed_iteration_supports

    supports = [np.array([1, 2], np.int32), np.array([3], np.int32)]
    by_class = seed_iteration_supports(supports, [1, 0], 2)
    assert all(type(rows) is list for rows in by_class)
    assert by_class[0][1] is supports[1] and by_class[1][0] is supports[0] and by_class[0][0].size == 0


def _parents_by_unique(csr, n_coarse_trans):
    """The previous whole-buffer route: np.unique of an int64 copy, with parent 0 for an empty support."""

    parents = np.unique(np.asarray(csr.ids_of_images(0, csr.n_images), dtype=np.int64) // int(n_coarse_trans))
    if bool(np.any(np.asarray(csr.n_significant) == 0)):
        parents = np.union1d(parents, [0])
    return parents


@pytest.mark.parametrize("with_empty_image", [False, True])
def test_coarse_parents_slice_scan_matches_unique_across_id_slices(with_empty_image):
    """Ids longer than one 2**20 slice give the parents np.unique gave; an empty image adds parent 0."""

    rng = np.random.default_rng(30)
    n_coarse_rot, n_coarse_trans = 200_000, 7
    # Distinct sorted cells per image; parent 0 appears only through the empty image (the ids start at parent 1).
    lengths = np.array([700_000, 0 if with_empty_image else 300_000, 600_000], dtype=np.int32)
    ids = [
        np.sort(rng.choice(np.arange(n_coarse_trans, n_coarse_rot * n_coarse_trans), size=length, replace=False))
        .astype(np.int32)
        for length in lengths
    ]
    assert sum(int(length) for length in lengths) > (1 << 20)
    csr = build_coarse_significance_csr(
        n_images=3, n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans,
        n_significant_per_batch=[lengths], store_excluded_per_batch=[np.zeros(3, dtype=bool)],
        ids_per_batch=[np.concatenate(ids)],
    )
    support = DeviceCompactedSignificantSamples(host_support_rows(csr), csr=csr)

    parents = significant_coarse_parents(
        support, n_images=3, n_coarse_rot=n_coarse_rot, n_coarse_trans=n_coarse_trans,
    )

    expected = _parents_by_unique(csr, n_coarse_trans)
    assert parents.dtype == expected.dtype
    np.testing.assert_array_equal(parents, expected)
    assert (0 in parents) == with_empty_image


def test_coarse_parents_of_only_empty_images_are_parent_zero():
    csr = build_coarse_significance_csr(
        n_images=2, n_coarse_rot=8, n_coarse_trans=3,
        n_significant_per_batch=[np.zeros(2, dtype=np.int32)],
        store_excluded_per_batch=[np.zeros(2, dtype=bool)], ids_per_batch=[np.zeros(0, dtype=np.int32)],
    )
    support = DeviceCompactedSignificantSamples(host_support_rows(csr), csr=csr)

    parents = significant_coarse_parents(support, n_images=2, n_coarse_rot=8, n_coarse_trans=3)

    np.testing.assert_array_equal(parents, _parents_by_unique(csr, 3))
    np.testing.assert_array_equal(parents, [0])


def test_assembled_class_releases_its_per_batch_ids():
    """A class compacted over the whole pass hands its per-batch lists to its CSR and they are emptied, so the
    pass does not hold every class's ids twice (relax#34); the CSR is the concatenation of the batches."""

    from types import SimpleNamespace

    from relax.scoring.pass1_assembly import significant_samples_after_loop

    n_rot, n_trans = 4, 2
    batches = [
        (np.array([1, 2], dtype=np.int32), np.zeros(2, dtype=bool), np.array([0, 3, 5], dtype=np.int32), 0),
        (np.array([0, 1], dtype=np.int32), np.zeros(2, dtype=bool), np.array([7], dtype=np.int32), 2),
    ]
    outputs = SimpleNamespace(
        significant_sample_indices=[[None] * 4, [None] * 4],
        device_significance_counts=[[b[0] for b in batches], [b[0] for b in batches]],
        device_significance_polarity=[[b[1] for b in batches], [b[1] for b in batches]],
        device_significance_ids=[[b[2] for b in batches], [b[2] for b in batches]],
        device_significance_starts=[[b[3] for b in batches], [b[3] for b in batches]],
    )
    plan = SimpleNamespace(n_classes=2, n_images=4, n_rot=n_rot, n_trans=n_trans)

    samples = significant_samples_after_loop(outputs, plan)

    for class_index in range(2):
        np.testing.assert_array_equal(samples[class_index].csr.ids_of_images(0, 4), [0, 3, 5, 7])
        np.testing.assert_array_equal(samples[class_index].csr.offsets, [0, 1, 3, 3, 4])
        assert outputs.device_significance_ids[class_index] == []
        assert outputs.device_significance_counts[class_index] == []


def test_csr_consumes_its_id_blocks_and_holds_their_concatenation(monkeypatch):
    """The CSR's ids are the per-batch blocks in order; the blocks' list slots are emptied as they are copied, and
    the freed heap is returned along the way, so the pass never holds a class's ids twice (relax#34)."""

    from relax.sparse_pass2 import resident_significance

    rng = np.random.default_rng(3)
    counts = [rng.integers(0, 6, size=n).astype(np.int32) for n in (4, 7, 1, 5)]
    blocks = [rng.integers(0, 12, size=int(c.sum())).astype(np.int32) for c in counts]
    # 400 x 3 samples: no row is smaller as a bit mask, so the CSR holds the blocks as they are.
    expected = np.concatenate(blocks)
    trims = []
    monkeypatch.setattr(resident_significance, "_CSR_TRIM_BYTES", 16)
    monkeypatch.setattr(resident_significance, "return_freed_heap", lambda where, **kwargs: trims.append(where) or 0)
    ids_per_batch = list(blocks)
    csr = build_coarse_significance_csr(
        n_images=17,
        n_coarse_rot=400,
        n_coarse_trans=3,
        n_significant_per_batch=counts,
        store_excluded_per_batch=[np.zeros(c.size, dtype=bool) for c in counts],
        ids_per_batch=ids_per_batch,
    )
    assert ids_per_batch == [None, None, None, None]
    assert csr.ids.dtype == np.int32 and np.array_equal(csr.ids, expected)
    assert np.array_equal(csr.offsets, np.r_[0, np.cumsum(np.concatenate(counts))])
    assert trims


# --- Rows kept as bit masks (relax#34) ---------------------------------------


def _mixed_encoding_case(seed=5, n_images=23):
    """Per-batch compaction results whose rows span every regime, and the plain CSR of the same ids."""

    rng = np.random.default_rng(seed)
    n_samples = STD_N_COARSE_ROT * N_COARSE_TRANS  # 360 samples: a 45-byte mask, so rows above 11 ids are masks
    sizes = rng.choice([0, 1, 5, 11, 12, 40, 170, 181, 300, n_samples], size=n_images)
    n_significant = sizes.astype(np.int32)
    store_excluded = 2 * sizes > n_samples
    stored = []
    for size, excluded in zip(sizes, store_excluded):
        ids = np.sort(rng.choice(n_samples, size=int(size), replace=False))
        stored.append((np.setdiff1d(np.arange(n_samples), ids) if excluded else ids).astype(np.int32))
    edges = [0, 4, 4, 15, n_images]
    batches = dict(
        n_significant_per_batch=[n_significant[a:b] for a, b in zip(edges[:-1], edges[1:])],
        store_excluded_per_batch=[store_excluded[a:b] for a, b in zip(edges[:-1], edges[1:])],
        ids_per_batch=[
            np.concatenate(stored[a:b]) if b > a else np.zeros(0, np.int32) for a, b in zip(edges[:-1], edges[1:])
        ],
    )
    offsets = np.zeros(n_images + 1, dtype=np.int32)
    offsets[1:] = np.cumsum([row.size for row in stored])
    plain = CoarseSignificanceCSR(
        n_images=n_images, n_coarse_rot=STD_N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS, offsets=offsets,
        ids=np.concatenate(stored), store_excluded=store_excluded, n_significant=n_significant,
    )
    return batches, plain


def _assert_same_csr(got: CoarseSignificanceCSR, expected: CoarseSignificanceCSR):
    assert type(got) is CoarseSignificanceCSR
    assert (got.n_images, got.n_coarse_rot, got.n_coarse_trans) == (
        expected.n_images, expected.n_coarse_rot, expected.n_coarse_trans,
    )
    for name in ("offsets", "ids", "store_excluded", "n_significant"):
        assert getattr(got, name).dtype == getattr(expected, name).dtype, name
        np.testing.assert_array_equal(getattr(got, name), getattr(expected, name), err_msg=name)


def _packed_and_plain():
    batches, plain = _mixed_encoding_case()
    packed = build_coarse_significance_csr(
        n_images=plain.n_images, n_coarse_rot=plain.n_coarse_rot, n_coarse_trans=plain.n_coarse_trans, **batches
    )
    assert isinstance(packed, PackedCoarseSignificanceCSR)
    return packed, plain


def test_a_row_is_stored_in_the_smaller_of_ids_and_bit_mask_and_decodes_to_the_same_ids():
    packed, plain = _packed_and_plain()
    counts = plain.counts().astype(np.int64)

    np.testing.assert_array_equal(packed.as_mask, 4 * counts > 45)
    assert packed.as_mask.any() and not packed.as_mask.all()
    assert packed.mask_rows.shape == (int(packed.as_mask.sum()), 45)
    assert packed.nbytes == int(np.minimum(4 * counts, 45).sum()) < plain.nbytes
    assert packed.n_ids == plain.n_ids
    np.testing.assert_array_equal(packed.counts(), plain.counts())
    for start, stop in [(0, plain.n_images), (0, 0), (3, 4), (2, 17), (9, plain.n_images)]:
        got = packed.ids_of_images(start, stop)
        assert got.dtype == np.int32
        np.testing.assert_array_equal(got, plain.ids_of_images(start, stop))
        _assert_same_csr(packed.image_block(start, stop), plain.image_block(start, stop))
    for image in range(plain.n_images):
        np.testing.assert_array_equal(packed.image_ids(image), plain.image_ids(image))


def test_a_support_without_a_dense_row_stays_a_plain_csr():
    counts = np.array([0, 3, 11], dtype=np.int32)
    csr = build_coarse_significance_csr(
        n_images=3, n_coarse_rot=STD_N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS,
        n_significant_per_batch=[counts], store_excluded_per_batch=[np.zeros(3, dtype=bool)],
        ids_per_batch=[np.arange(14, dtype=np.int32)],
    )
    assert type(csr) is CoarseSignificanceCSR


def test_a_mask_row_must_be_strictly_ascending():
    ids = np.arange(20, dtype=np.int32)
    ids[[4, 5]] = ids[[5, 4]]
    with pytest.raises(ValueError, match="strictly ascending"):
        build_coarse_significance_csr(
            n_images=1, n_coarse_rot=STD_N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS,
            n_significant_per_batch=[np.array([20], dtype=np.int32)],
            store_excluded_per_batch=[np.zeros(1, dtype=bool)], ids_per_batch=[ids],
        )


def test_packed_csr_consumers_give_the_plain_csr_results():
    packed, plain = _packed_and_plain()
    children = fine_rotation_children(
        n_coarse_rot=STD_N_COARSE_ROT, nside_level=STD_NSIDE_LEVEL, oversampling_order=STD_OVERSAMPLING,
        random_perturbation=0.0, fine_rotation_parent_override=None,
    )
    # Steps of a few images, so a step mixes id rows and mask rows.
    np.testing.assert_array_equal(
        csr_candidate_rows_per_image(packed, children[0], cells_per_step=200),
        csr_candidate_rows_per_image(plain, children[0], cells_per_step=200),
    )
    table_kwargs = dict(
        nside_level=STD_NSIDE_LEVEL, oversampling_order=STD_OVERSAMPLING, n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=np.linspace(-1.0, 1.0, STD_N_COARSE_ROT, dtype=np.float32), random_perturbation=0.0,
        relion_parent_execution_order=True, dtype=np.float32, children=children,
    )
    _assert_tables_equal(
        build_resident_candidate_tables_from_csr(packed, **table_kwargs),
        build_resident_candidate_tables_from_csr(plain, **table_kwargs),
    )
    keep = np.arange(plain.n_images) % 3 != 1
    restricted, expected = csr_restricted_to_images(packed, keep), csr_restricted_to_images(plain, keep)
    assert isinstance(restricted, PackedCoarseSignificanceCSR)
    _assert_same_csr(restricted.image_block(0, plain.n_images), expected)
    sparse_only = ~plain.store_excluded & (plain.n_significant != plain.n_samples)
    for images in (sparse_only, sparse_only & (plain.counts() <= 11)):
        parents = [
            significant_coarse_parents(
                DeviceCompactedSignificantSamples(csr=csr_restricted_to_images(csr, images)),
                n_images=plain.n_images, n_coarse_rot=STD_N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS,
            )
            for csr in (packed, plain)
        ]
        np.testing.assert_array_equal(parents[0], parents[1])
    assert type(csr_restricted_to_images(packed, sparse_only & (plain.counts() <= 11))) is CoarseSignificanceCSR


def _same_host_row(row, reference) -> bool:
    if reference is None or row is None:
        return row is reference
    if isinstance(reference, ComplementSignificantSampleIndices):
        return (
            isinstance(row, ComplementSignificantSampleIndices)
            and row.total_size == reference.total_size
            and np.array_equal(row.excluded_indices, reference.excluded_indices)
        )
    return (
        not isinstance(row, ComplementSignificantSampleIndices)
        and row.dtype == reference.dtype
        and np.array_equal(row, reference)
    )


@pytest.mark.parametrize("packed_table", [False, True])
def test_seed_iteration_supports_from_the_csr_are_the_redistributed_host_rows(packed_table):
    """A seed iteration's per-class supports, taken from the restricted CSR, are the rows the host form gives:
    each image's first-class row in its seed class and an empty int32 row in the others, in either encoding."""

    from relax.classification.k_class_inputs import seed_iteration_supports

    packed, plain = _packed_and_plain()
    first_rows = host_support_rows(plain)
    seeds = np.random.default_rng(8).integers(0, 3, size=plain.n_images)
    host_form = seed_iteration_supports(first_rows, seeds, 3)
    assert all(type(rows) is list for rows in host_form)

    by_class = seed_iteration_supports(
        DeviceCompactedSignificantSamples(csr=packed if packed_table else plain), seeds, 3
    )

    for k in range(3):
        assert len(by_class[k]) == plain.n_images
        for image, (row, reference) in enumerate(zip(by_class[k], host_form[k], strict=True)):
            assert _same_host_row(row, reference), (k, image)
        np.testing.assert_array_equal(by_class[k].csr.counts(), np.where(seeds == k, plain.counts(), 0))


def test_support_rows_decode_on_access_and_are_not_kept():
    packed, plain = _packed_and_plain()
    expected = host_support_rows(plain)
    support = DeviceCompactedSignificantSamples(csr=packed)
    same = _same_host_row

    assert len(support) == plain.n_images
    assert all(same(row, reference) for row, reference in zip(support, expected, strict=True))
    assert all(same(support[image], expected[image]) for image in range(plain.n_images))
    assert same(support[-1], expected[-1])
    assert all(same(row, reference) for row, reference in zip(support[3:9], expected[3:9], strict=True))
    # Nothing decoded is stored: the list's own slots stay placeholders, which refuse to act as a row.
    with pytest.raises(TypeError, match="device-compacted support row"):
        np.asarray(list.__getitem__(support, 0))


def test_device_packed_batches_build_the_table_plain_id_batches_build():
    """Batches whose dense rows the device packed give the table the same supports give as id arrays, and a
    padded final batch packs only its real images."""

    n_samples = STD_N_COARSE_ROT * N_COARSE_TRANS
    rng = np.random.default_rng(21)
    grid = dict(n_coarse_rot=STD_N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS)
    device, plain = dict(n=[], s=[], ids=[]), dict(n=[], s=[], ids=[])
    for actual, densities in ((4, (0.0, 0.01, 0.2, 0.9)), (3, (0.4, 0.03, 1.0, 0.7))):
        mask = np.stack([rng.random(n_samples) < density for density in densities])
        n_significant, store_excluded, ids, _rot_any = compact_batch_significance(
            mask, actual_batch_size=actual, batch_n_sig=mask.sum(axis=1).astype(np.int32), **grid
        )
        assert isinstance(ids, PackedSignificanceBatch) and ids.mask_rows.shape[1] == 45
        stored = [np.flatnonzero(~row if excluded else row) for row, excluded in zip(mask[:actual], store_excluded)]
        np.testing.assert_array_equal(ids.as_mask, [4 * row.size > 45 for row in stored])
        device["n"].append(n_significant), device["s"].append(store_excluded), device["ids"].append(ids)
        plain["n"].append(n_significant), plain["s"].append(store_excluded)
        plain["ids"].append(np.concatenate(stored).astype(np.int32))
    tables = [
        build_coarse_significance_csr(
            n_images=7, n_significant_per_batch=b["n"], store_excluded_per_batch=b["s"], ids_per_batch=b["ids"], **grid
        )
        for b in (device, plain)
    ]
    assert all(isinstance(table, PackedCoarseSignificanceCSR) for table in tables)
    for name in ("offsets", "as_mask", "id_values", "mask_rows", "store_excluded", "n_significant"):
        np.testing.assert_array_equal(getattr(tables[0], name), getattr(tables[1], name), err_msg=name)
    assert device["ids"] == [None, None]
