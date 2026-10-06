"""Seed-class partitioning keeps compact supports and the host candidate ordering."""

import numpy as np
import pytest

from helpers.float_compare import assert_matches
from relax.classification.k_class_inputs import seed_iteration_supports
from relax.scoring.significant_samples import compact_significant_sample_indices_from_mask
from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs
from relax.sparse_pass2.resident_candidates import build_resident_candidate_tables
from relax.sparse_pass2.resident_significance import (
    DeviceCompactedSignificantSamples,
    build_coarse_significance_csr,
    build_resident_candidate_tables_from_csr,
    host_support_rows,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("execution_order", [False, True])
def test_seed_classes_keep_csr_and_match_host_candidate_tables(execution_order):
    n_rot, n_trans, n_classes = 18, 5, 4
    masks = np.zeros((6, n_rot * n_trans), dtype=bool)
    masks[0, [0, 7, 9]] = True
    masks[2:4] = True
    masks[3, [2, 11]] = False
    masks[4, [80, 87]] = True
    masks[5] = True
    counts = masks.sum(axis=1).astype(np.int32)
    excluded = counts * 2 > masks.shape[1]
    ids = np.concatenate([
        np.flatnonzero(~row if is_excluded else row).astype(np.int32)
        for row, is_excluded in zip(masks, excluded)
    ])
    csr = build_coarse_significance_csr(
        n_images=len(masks), n_coarse_rot=n_rot, n_coarse_trans=n_trans,
        n_significant_per_batch=[counts], store_excluded_per_batch=[excluded],
        ids_per_batch=[ids],
    )
    supports = DeviceCompactedSignificantSamples(host_support_rows(csr), csr=csr)
    seeds = np.array([0, 2, 2, 0, 1, 2])
    classes = seed_iteration_supports(supports, seeds, n_classes)
    assert all(isinstance(support, DeviceCompactedSignificantSamples) for support in classes)
    assert sum(support.csr.ids.size for support in classes) == csr.ids.size
    assert all(not np.shares_memory(support.csr.ids, csr.ids) for support in classes)
    fine_parent = np.repeat(np.arange(n_rot, dtype=np.int32), 2)
    fine_rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (len(fine_parent), 3, 3))
    trans_parent = np.repeat(np.arange(n_trans, dtype=np.int32), 3)
    prior = np.linspace(-2, 2, n_rot, dtype=np.float32)
    common = dict(
        nside_level=0, oversampling_order=0, n_fine_trans=len(trans_parent),
        fine_translation_parent=trans_parent, rotation_log_prior=prior,
        random_perturbation=0.0, fine_rotation_parent_override=fine_parent,
        relion_parent_execution_order=execution_order, dtype=np.float32,
    )
    for k, support in enumerate(classes):
        expected_mask = masks & (seeds == k)[:, None]
        expected_rows = [compact_significant_sample_indices_from_mask(row) for row in expected_mask]
        np.testing.assert_array_equal(support.csr.n_significant, expected_mask.sum(axis=1))
        # Decode every selected/excluded/full/empty row, including the empty class.
        for i in range(len(masks)):
            decoded = np.full(masks.shape[1], support.csr.store_excluded[i], dtype=bool)
            decoded[support.csr.image_ids(i)] = not support.csr.store_excluded[i]
            np.testing.assert_array_equal(decoded, expected_mask[i])
        host = _prepare_per_image_pass2_inputs(
            expected_rows, n_coarse_rot=n_rot, n_coarse_trans=n_trans,
            fine_rotations_override=fine_rotations, **common,
        )
        expected = build_resident_candidate_tables(
            host, n_coarse_trans=n_trans, n_fine_trans=len(trans_parent),
            fine_translation_parent=trans_parent,
        )
        got = build_resident_candidate_tables_from_csr(support.csr, **common)
        for name in (
            "row_offsets", "row_unit", "row_fine_rot", "row_parent_local",
            "mask_mode", "parent_offsets", "parent_trans_bits", "row_log_prior",
        ):
            assert_matches(getattr(got, name), getattr(expected, name), err_msg=name)
