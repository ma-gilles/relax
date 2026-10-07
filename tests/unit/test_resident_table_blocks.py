"""Candidate tables built one image block at a time are the whole-pass tables.

A pass with the device-compacted CSR plans its chunks from per-image row counts
(``csr_candidate_rows_per_image``) and builds each image block's tables when the
chunk loop reaches it (``_candidate_table_blocks``). Every chunk's host rows must
be the rows the whole-pass table gives the same chunk, for one class and for the
merged K-class table, with and without RELION's parent execution order and
VDAM's reconstruction groups.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches
from test_resident_significance import _csr_from_supports, _supports

import relax.sparse_pass2.resident_pass2 as rp
from relax.sparse_pass2.resident_candidates import CandidateTableBlocks, plan_capacity_chunks, table_block_starts
from relax.sparse_pass2.resident_significance import (
    DeviceCompactedSignificantSamples,
    build_resident_candidate_tables_from_csr,
    csr_candidate_rows_per_image,
    fine_rotation_children,
    host_support_rows,
)

pytestmark = pytest.mark.unit

N_COARSE_ROT = 18  # 3 directions x 6 psi at HEALPix level 0
N_COARSE_TRANS = 5
N_FINE_TRANS = 10
FINE_TRANS_PARENT = np.repeat(np.arange(N_COARSE_TRANS, dtype=np.int32), 2)
FINE_ROT_PARENT = np.repeat(np.arange(N_COARSE_ROT, dtype=np.int64), 2)
N_IMAGES = 37


def _class_csr(seed):
    n_samples = N_COARSE_ROT * N_COARSE_TRANS
    supports = _supports(N_IMAGES, n_samples, seed=seed)
    order = np.random.default_rng(seed).permutation(N_IMAGES)
    return _csr_from_supports([supports[i] for i in order], n_coarse_rot=N_COARSE_ROT, n_coarse_trans=N_COARSE_TRANS)


def _children():
    return fine_rotation_children(
        n_coarse_rot=N_COARSE_ROT,
        nside_level=0,
        oversampling_order=1,
        random_perturbation=0.0,
        fine_rotation_parent_override=FINE_ROT_PARENT,
    )


@pytest.mark.parametrize("cells_per_step", [1, 4, 1000])
def test_row_counts_are_the_built_tables_rows(cells_per_step):
    csr = _class_csr(seed=5)
    children = _children()
    tables = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=0,
        oversampling_order=1,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotation_parent_override=FINE_ROT_PARENT,
        children=children,
    )
    counts = csr_candidate_rows_per_image(csr, children[0], cells_per_step=cells_per_step)
    np.testing.assert_array_equal(counts, np.diff(np.asarray(tables.row_offsets, dtype=np.int64)))


def test_block_starts_hold_at_most_the_budget_unless_one_image_exceeds_it():
    rows = np.array([3, 5, 2, 9, 1, 1, 4, 0, 6])
    offsets = np.concatenate([[0], np.cumsum(rows)])
    starts = table_block_starts(offsets, 7)
    assert starts[0] == 0 and starts[-1] == rows.size
    for start, stop in zip(starts[:-1], starts[1:]):
        assert stop > start
        assert offsets[stop] - offsets[start] <= 7 or stop == start + 1


def _source(n_classes, *, whole, execution_order, groups, monkeypatch):
    monkeypatch.setattr(rp, "_BLOCK_UNITS", 40)
    csrs = [_class_csr(seed=11 + 7 * k) for k in range(n_classes)]
    supports = [DeviceCompactedSignificantSamples(host_support_rows(csr), csr=csr) for csr in csrs]
    priors = [np.linspace(-1.0, 1.0, N_COARSE_ROT, dtype=np.float32) + k for k in range(n_classes)]
    group_ids = (np.arange(N_IMAGES) % 2).astype(np.int32) if groups else None
    blocks, _, _ = rp._candidate_table_blocks(
        supports,
        priors,
        whole=whole,
        reconstruction_group_ids=group_ids,
        reconstruction_group_count=2 if groups else None,
        n_images=N_IMAGES,
        n_coarse_rot=N_COARSE_ROT,
        n_coarse_trans=N_COARSE_TRANS,
        nside_level=0,
        oversampling_order=1,
        n_fine_trans=N_FINE_TRANS,
        fine_translation_parent=FINE_TRANS_PARENT,
        random_perturbation=0.0,
        fine_source_eulers_override=None,
        fine_rotations_override=np.stack([np.eye(3, dtype=np.float32)] * FINE_ROT_PARENT.size),
        fine_mstep_rotations_override=None,
        fine_rotation_parent_override=FINE_ROT_PARENT,
        use_relion_f32_fine_posterior=execution_order,
        dtype=np.float32,
        symmetry_label="C1",
    )
    return blocks


@pytest.mark.parametrize("n_classes", [1, 3])
@pytest.mark.parametrize("execution_order", [False, True])
@pytest.mark.parametrize("groups", [False, True])
def test_blocked_chunk_rows_are_the_whole_table_rows(n_classes, execution_order, groups, monkeypatch):
    kwargs = dict(execution_order=execution_order, groups=groups, monkeypatch=monkeypatch)
    blocked = _source(n_classes, whole=False, **kwargs)
    whole = _source(n_classes, whole=True, **kwargs)
    assert isinstance(blocked, CandidateTableBlocks) and blocked.n_blocks > 2 and whole.n_blocks == 1
    np.testing.assert_array_equal(blocked.row_offsets, whole.row_offsets)
    assert (blocked.n_classes, blocked.n_slots) == (whole.n_classes, whole.n_slots)

    chunks = [
        chunk
        for block in blocked.blocks()
        for chunk in plan_capacity_chunks(
            blocked, row_capacity_ladder=(8, 32), image_capacity_ladder=(2, 4), image_range=block
        )
    ]
    assert [c.image_start for c in chunks[1:]] == [c.image_stop for c in chunks[:-1]]
    assert chunks[-1].image_stop == N_IMAGES
    for chunk in chunks:
        _, got, _ = rp._chunk_host_rows(blocked, chunk)
        _, want, _ = rp._chunk_host_rows(whole, chunk)
        assert got.keys() == want.keys()
        for key in got:
            if key == "row_log_prior":
                assert_matches(got[key], want[key])
            else:
                np.testing.assert_array_equal(got[key], want[key], err_msg=key)
    # The loop keeps only the blocks it still needs.
    assert len(blocked._built) <= blocked.blocks_kept
