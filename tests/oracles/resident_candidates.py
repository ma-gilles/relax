"""One image's dense candidate mask decoded from the resident candidate table.

Test oracle: moved from relax/fine_pass/resident_candidates.py, where only tests called it; it is the independent
host decoding the table builders, ``materialize_chunk`` and ``expand_chunk_mask_jnp`` are checked against.
"""

import numpy as np

from relax.fine_pass.resident_candidates import (
    _MASK_MODE_BITSET,
    _MASK_MODE_EMPTY,
    _MASK_MODE_FULL,
    _MASK_WORD_BITS,
    ResidentCandidateTables,
)


def expand_mask_rows(tables: ResidentCandidateTables, image: int, fine_translation_parent) -> np.ndarray:
    """Reference (numpy) dense mask for one image; matches ``SparseCandidateMask``'s dense array exactly.

    Returns a ``bool[n_rows_i, len(fine_translation_parent)]`` array. This is
    the ground truth the device chunk-mask expansion must reproduce.
    """

    start, stop = int(tables.row_offsets[image]), int(tables.row_offsets[image + 1])
    n_rows_i = stop - start
    fine_translation_parent = np.asarray(fine_translation_parent)
    n_fine_trans = int(fine_translation_parent.shape[0])
    mode = int(tables.mask_mode[image])

    if mode == _MASK_MODE_FULL:
        return np.ones((n_rows_i, n_fine_trans), dtype=bool)
    if mode == _MASK_MODE_EMPTY:
        return np.zeros((n_rows_i, n_fine_trans), dtype=bool)
    if mode != _MASK_MODE_BITSET:
        raise ValueError(f"image {image}: unknown mask_mode {mode}")

    p0, p1 = int(tables.parent_offsets[image]), int(tables.parent_offsets[image + 1])
    bits = tables.parent_trans_bits[p0:p1]
    row_parent = tables.row_parent_local[start:stop]
    row_bits = bits[row_parent].astype(np.uint32)
    words, shifts = np.divmod(fine_translation_parent.astype(np.int64), _MASK_WORD_BITS)
    return (((row_bits[:, words] >> shifts.astype(np.uint32)[None, :]) & np.uint32(1)) != 0)
