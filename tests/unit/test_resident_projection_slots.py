"""A significant-parent projection cache holds every candidate row's fine rotation, and nothing else."""

from types import SimpleNamespace

import numpy as np
import pytest

from relax.sparse_pass2 import resident_pass2 as rp
from relax.sparse_pass2.resident_significance import (
    CoarseSignificanceCSR,
    build_resident_candidate_tables_from_csr,
    fine_rotation_children,
)

pytestmark = pytest.mark.unit

N_COARSE, N_TRANS = 18, 3  # whole psi rows at HEALPix level 0
PARENT = np.repeat(np.arange(N_COARSE), 8)  # oversampling 1: 8 children per parent


def _csr(per_image_cells, *, full_images=()):
    ids, offsets, n_significant, excluded = [], [0], [], []
    for image, cells in enumerate(per_image_cells):
        if image in full_images:
            n_significant.append(N_COARSE * N_TRANS)
            excluded.append(True)
            offsets.append(offsets[-1])
            continue
        cells = np.unique(np.asarray(cells, dtype=np.int32))
        ids.extend(cells.tolist())
        offsets.append(offsets[-1] + cells.size)
        n_significant.append(cells.size)
        excluded.append(False)
    return CoarseSignificanceCSR(
        n_images=len(per_image_cells),
        n_coarse_rot=N_COARSE,
        n_coarse_trans=N_TRANS,
        offsets=np.asarray(offsets, np.int64),
        ids=np.asarray(ids, np.int32),
        store_excluded=np.asarray(excluded, bool),
        n_significant=np.asarray(n_significant, np.int32),
    )


def _slots(supports, n_images):
    return rp._significant_projection_slots(
        supports,
        n_images=n_images,
        n_coarse_rot=N_COARSE,
        n_coarse_trans=N_TRANS,
        n_fine_rot=PARENT.size,
        children=fine_rotation_children(
            n_coarse_rot=N_COARSE,
            nside_level=0,
            oversampling_order=1,
            random_perturbation=0.0,
            fine_rotation_parent_override=PARENT,
        ),
    )


def _table_rotations(csr):
    table = build_resident_candidate_tables_from_csr(
        csr,
        nside_level=0,
        oversampling_order=1,
        n_fine_trans=2 * N_TRANS,
        fine_translation_parent=np.repeat(np.arange(N_TRANS), 2),
        rotation_log_prior=None,
        random_perturbation=0.0,
        fine_rotation_parent_override=PARENT,
        relion_parent_execution_order=False,
        dtype=np.float32,
    )
    return np.asarray(table.row_fine_rot, dtype=np.int64)


@pytest.fixture
def small_slots(monkeypatch):
    monkeypatch.setattr(rp, "_MIN_PROJECTION_SLOTS", 8)


def test_the_cache_rows_are_exactly_the_candidate_rows_rotations(small_slots):
    # Image 2 keeps nothing: RELION's empty support takes parent 0.
    classes = [_csr([[0, 4, 5], [21, 22, 50], []]), _csr([[7], [52, 53], [9]])]
    slots = _slots([SimpleNamespace(csr=csr) for csr in classes], 3)
    assert slots is not None and slots.capacity == 32  # 24 rotations of class 0 -> the next power of two
    n_fine = PARENT.size
    for class_index, csr in enumerate(classes):
        needed = np.unique(_table_rotations(csr))
        cached = slots.slot_projection[class_index * slots.capacity : (class_index + 1) * slots.capacity]
        np.testing.assert_array_equal(np.unique(cached - class_index * n_fine), needed)
        # Each needed projection id maps to a row holding it; the others are absent.
        ids = class_index * n_fine + needed
        np.testing.assert_array_equal(slots.slot_projection[slots.slot_of_projection[ids]], ids)
        absent = np.setdiff1d(np.arange(n_fine), needed) + class_index * n_fine
        assert np.all(slots.slot_of_projection[absent] == -1)


def test_rows_map_to_their_cache_rows_and_padded_rows_to_row_zero(small_slots):
    csr = _csr([[3, 4], [40]])
    slots = _slots([SimpleNamespace(csr=csr)], 2)
    rotations = _table_rotations(csr)
    host_chunk = {"row_fine_rot": np.concatenate([rotations, [5, 5]]), "n_valid_rows": rotations.size}
    mapped = rp._row_projection_ids(host_chunk, None, slots.slot_of_projection)
    np.testing.assert_array_equal(slots.slot_projection[mapped[: rotations.size]], rotations)
    np.testing.assert_array_equal(mapped[rotations.size :], 0)
    with pytest.raises(RuntimeError, match="not in the significant-parent projection cache"):
        rp._row_projection_ids({"row_fine_rot": np.array([0]), "n_valid_rows": 1}, None, slots.slot_of_projection)


def test_supports_that_take_every_parent_cache_the_whole_grid(small_slots):
    assert _slots([SimpleNamespace(csr=_csr([[3], []], full_images=(1,)))], 2) is None
    assert _slots([SimpleNamespace(csr=None)], 1) is None
    # The children of most parents need a capacity over half the grid.
    every_cell = list(range(N_COARSE * N_TRANS - 1))
    assert _slots([SimpleNamespace(csr=_csr([every_cell]))], 1) is None


def test_a_refinement_reuses_a_larger_capacity_it_already_compiled(small_slots):
    with rp.stable_window_class_history():
        assert rp._projection_slot_capacity(40, 144) == 64
        assert rp._projection_slot_capacity(20, 144) == 64  # 32 is new, 64 within twice it was used
        assert rp._projection_slot_capacity(10, 144) == 16
        assert rp._projection_slot_capacity(100, 144) is None  # 128 rows: more than half the grid's 144



def test_the_cache_holds_each_classs_capacity_not_its_whole_grid(small_slots):
    """554de97 allocated every class's whole fine grid (32 GiB at current size 170, refused twice
    before the pass split it into row blocks, ab_lazy 14914828); the cache holds the slots only."""
    classes = [_csr([[0, 4, 5], [21, 22, 50], []]), _csr([[7], [52, 53], [9]])]
    slots = _slots([SimpleNamespace(csr=csr) for csr in classes], 3)
    assert rp.projection_cache_rows(2, PARENT.size, slots) == 2 * slots.capacity == slots.slot_projection.size
    assert rp.projection_cache_rows(2, PARENT.size, None) == 2 * PARENT.size
