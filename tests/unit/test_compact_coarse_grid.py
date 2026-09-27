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
    from relax import sampling
    from relax.scoring.sparse_bucket_arrays import _prepare_per_image_pass2_inputs

    order, n_trans = 1, 3
    fine_rot, parent, _mstep, eulers = sampling.get_oversampled_rotation_grid_from_samples(
        fine_ids,
        order,
        oversampling_order=1,
        random_perturbation=0.13,
        return_mstep_rotations=True,
        return_source_eulers=True,
        dtype=np.float32,
    )
    return _prepare_per_image_pass2_inputs(
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
    from relax import sampling

    rng = np.random.default_rng(7)
    n_rot, n_trans = int(sampling.rotation_grid_size(1)), 3
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
