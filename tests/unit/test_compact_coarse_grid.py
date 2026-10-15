"""A compact coarse grid with per-image rotation priors builds the full grid's pass-2 hypotheses (S4.2 local search).

A subtomogram local search scores only the union of its particles' local rotations and gives each
particle its own orientation prior. The host pass-2 preparation then works on that compact grid
(``coarse_rotation_ids``) with ``per_image_rotation_log_prior``; every image's fine rows, their RELION
execution order, priors and candidate masks must be the ones the full HEALPix grid gives.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytestmark = pytest.mark.unit


def _inputs(supports, *, n_coarse_rot, fine_ids, prior, **extra):
    from relax.fine_pass.bucket_arrays import prepare_per_image_pass2_inputs
    from relax.sampling import oversampled_grids

    order, n_trans = 1, 3
    fine_rot, parent, _mstep, eulers = oversampled_grids.get_oversampled_rotation_grid_from_samples(
        fine_ids,
        order,
        oversampling_order=1,
        random_perturbation=0.13,
        return_mstep_rotations=True,
        return_source_eulers=True,
        dtype=np.float32,
    )
    return prepare_per_image_pass2_inputs(
        supports,
        n_coarse_rot=n_coarse_rot,
        n_coarse_trans=n_trans,
        nside_level=order,
        oversampling_order=1,
        n_fine_trans=2 * n_trans,
        fine_translation_parent=np.repeat(np.arange(n_trans), 2),
        rotation_log_prior=prior,
        random_perturbation=0.13,
        fine_source_eulers_override=eulers,
        fine_rotations_override=fine_rot,
        fine_rotation_parent_override=parent,
        relion_parent_execution_order=True,
        **extra,
    )


def test_compact_grid_with_image_priors_equals_the_full_grid():
    from relax.sampling import grids

    rng = np.random.default_rng(7)
    n_rot, n_trans = int(grids.rotation_grid_size(1)), 3
    prior = rng.normal(size=n_rot).astype(np.float32)
    per_image_prior = [rng.normal(size=n_rot).astype(np.float32) for _ in range(4)]
    global_supports = [
        np.sort(rng.choice(n_rot * n_trans, size=int(rng.integers(3, 40)), replace=False)).astype(np.int32)
        for _ in range(4)
    ]
    used = np.unique(np.concatenate([s // n_trans for s in global_supports]))
    compact_of = {int(g): c for c, g in enumerate(used)}
    compact_supports = [
        np.asarray([compact_of[int(x // n_trans)] * n_trans + int(x % n_trans) for x in s], dtype=np.int32)
        for s in global_supports
    ]

    # The full grid with each image's own prior is built one image at a time (a shared prior is all it takes).
    full = [
        _inputs([s], n_coarse_rot=n_rot, fine_ids=np.arange(n_rot), prior=p)
        for s, p in zip(global_supports, per_image_prior)
    ]
    compact = _inputs(
        compact_supports,
        n_coarse_rot=used.size,
        fine_ids=used,
        prior=None,
        coarse_rotation_ids=used,
        per_image_rotation_log_prior=[p[np.unique(s // n_trans)] for s, p in zip(global_supports, per_image_prior)],
    )
    for image, reference in enumerate(full):
        assert_matches(compact["oversampled_rots"][image], reference["oversampled_rots"][0])
        assert_matches(compact["source_eulers"][image], reference["source_eulers"][0])
        assert_matches(compact["log_prior"][image], reference["log_prior"][0])
        np.testing.assert_array_equal(
            np.asarray(compact["candidate_mask"][image]), np.asarray(reference["candidate_mask"][0])
        )


def test_k_class_tables_on_a_compact_grid_carry_each_class_supports_own_priors():
    """Class3D under an orientational prior: one candidate table per class on the compact grid, each row with
    the particle's own prior of its parent rotation, and no candidate for a class the particle is not scored
    against (the seed iteration, or a class without a significant sample)."""

    from relax.fine_pass import resident_table_blocks
    from relax.sampling import grids, oversampled_grids

    rng = np.random.default_rng(11)
    order, n_trans, n_units, n_classes = 1, 3, 4, 2
    n_rot = int(grids.rotation_grid_size(order))
    unit_prior = [rng.normal(size=n_rot).astype(np.float32) for _ in range(n_units)]  # over the grid's rotations
    grid_supports = [
        [
            np.sort(rng.choice(n_rot * n_trans, size=int(rng.integers(3, 30)), replace=False)).astype(np.int32)
            for _ in range(n_units)
        ]
        for _ in range(n_classes)
    ]
    grid_supports[1][2] = np.zeros(0, dtype=np.int32)  # unit 2 is not scored against class 1
    used = np.unique(np.concatenate([s // n_trans for class_supports in grid_supports for s in class_supports]))
    compact_of = np.full(n_rot, -1)
    compact_of[used] = np.arange(used.size)
    supports = [[compact_of[s // n_trans] * n_trans + s % n_trans for s in class_supports] for class_supports in grid_supports]
    priors = tuple(
        [unit_prior[u][np.unique(s // n_trans)] for u, s in enumerate(class_supports)] for class_supports in grid_supports
    )
    fine_rot, parent, _mstep, eulers = oversampled_grids.get_oversampled_rotation_grid_from_samples(
        used, order, oversampling_order=1, random_perturbation=0.13, return_mstep_rotations=True,
        return_source_eulers=True, dtype=np.float32,
    )  # fmt: skip
    blocks, _, _ = resident_table_blocks.candidate_table_blocks(
        supports,
        (None, None),
        whole=False,
        reconstruction_group_ids=None,
        reconstruction_group_count=None,
        n_images=n_units,
        n_coarse_rot=used.size,
        n_coarse_trans=n_trans,
        nside_level=order,
        oversampling_order=1,
        n_fine_trans=2 * n_trans,
        fine_translation_parent=np.repeat(np.arange(n_trans), 2),
        random_perturbation=0.13,
        fine_source_eulers_override=eulers,
        fine_rotations_override=fine_rot,
        fine_mstep_rotations_override=None,
        fine_rotation_parent_override=parent,
        use_relion_f32_fine_posterior=True,
        dtype=np.float32,
        symmetry_label="C1",
        coarse_rotation_ids=used,
        per_image_rotation_log_prior=priors,
    )
    tables = blocks.block_tables(0)
    assert (tables.n_images, tables.n_classes) == (n_units, n_classes)
    row_unit, row_class = np.asarray(tables.row_unit), np.asarray(tables.row_class)
    row_parent = used[np.asarray(parent)[np.asarray(tables.row_fine_rot)]]  # the grid rotation of the row's parent
    bits = np.asarray(tables.parent_trans_bits)
    parent_offsets = np.asarray(tables.parent_offsets)
    for unit in range(n_units):
        for k in range(n_classes):
            rows = (row_unit == unit) & (row_class == k)
            if grid_supports[k][unit].size == 0:
                continue
            assert_matches(np.asarray(tables.row_log_prior)[rows], unit_prior[unit][row_parent[rows]])
            np.testing.assert_array_equal(np.unique(row_parent[rows]), np.unique(grid_supports[k][unit] // n_trans))
        # One bit per significant coarse sample of the unit, over its classes: a class without a sample sets none.
        unit_bits = bits[parent_offsets[unit] : parent_offsets[unit + 1]]
        n_set = int(sum(bin(int(word)).count("1") for word in unit_bits.reshape(-1)))
        assert n_set == sum(grid_supports[k][unit].size for k in range(n_classes))

    with pytest.raises(ValueError, match="each class its particles' priors"):
        resident_table_blocks.candidate_table_blocks(
            supports, (None, None), whole=False, reconstruction_group_ids=None, reconstruction_group_count=None,
            n_images=n_units, n_coarse_rot=used.size, n_coarse_trans=n_trans, coarse_rotation_ids=used,
            per_image_rotation_log_prior=priors[:1],
        )  # fmt: skip
