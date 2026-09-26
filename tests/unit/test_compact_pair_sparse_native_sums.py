"""Pair-sparse native weighted sums for compact K-class buckets.

``dual_weighted_sums_f32`` loops over every (row, translation) slot of a dense
``(B, R, T)`` probability table and skips zeros; compact pairs populate ~1 % of
that table and most rows are padding. ``dual_weighted_sums_pairs_f32`` iterates
each row's own sorted pair range with the same fma order, so it must be
bit-identical. The source-ordered CSR offsets are pinned on CPU; the kernel pin needs a GPU.
"""

import numpy as np



def _random_unique_pairs(rng, batch, n_rows, n_trans, n_pairs, *, fill=0.6):
    """Source-ordered pairs as build_compact_pair_index_arrays emits them.

    Rotation-major, translation-minor valid prefix (C order of the (R, T) mask),
    then -1 padding. The CSR helper relies on this order instead of sorting.
    """
    rows = np.full((batch, n_pairs), -1, dtype=np.int32)
    trans = np.full((batch, n_pairs), -1, dtype=np.int32)
    for b in range(batch):
        n_valid = int(rng.integers(1, max(2, int(fill * n_pairs))))
        keys = np.sort(rng.choice(n_rows * n_trans, size=min(n_valid, n_rows * n_trans), replace=False))
        rows[b, : keys.size] = keys // n_trans
        trans[b, : keys.size] = keys % n_trans
    mask = rows >= 0
    probs = np.where(mask, rng.random((batch, n_pairs)), 0.0).astype(np.float32)
    return rows, trans, mask, probs


def test_builder_emits_source_order_the_csr_helper_relies_on():
    from relax.scoring.compact_candidates import (
        SparseCandidateMask,
        build_compact_pair_index_arrays,
    )

    rng = np.random.default_rng(3)
    n_rows, n_trans, c_rot, c_trans = 15, 8, 5, 4
    ftp = np.repeat(np.arange(c_trans), 2).astype(np.int32)
    masks = [
        SparseCandidateMask(
            mode="coarse",
            n_rows=n_rows,
            n_fine_trans=n_trans,
            parent_map=rng.integers(0, c_rot, n_rows),
            coarse_valid=rng.random((c_rot, c_trans)) < 0.5,
            fine_translation_parent=ftp,
        )
        for _ in range(3)
    ] + [SparseCandidateMask(mode="full", n_rows=n_rows, n_fine_trans=n_trans)]
    arrays = build_compact_pair_index_arrays(masks, pair_bucket_size=n_rows * n_trans)
    for b in range(len(masks)):
        count = int(arrays["pair_counts"][b])
        keys = arrays["local_rotation_row"][b, :count].astype(np.int64) * n_trans + arrays["translation_idx"][b, :count]
        assert np.all(np.diff(keys) > 0), "valid prefix must be rotation-major, translation-minor"
        assert np.all(arrays["local_rotation_row"][b, count:] == -1) and not arrays["pair_mask"][b, count:].any()


