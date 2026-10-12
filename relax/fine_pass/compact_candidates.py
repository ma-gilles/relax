"""Host candidate masks of pass 2 and their dense materialization.

Numerical execution stays in the local and sparse engines.
"""

from __future__ import annotations

import numpy as np


class SparseCandidateMask:
    """Compact host representation of one image's pass-2 candidate mask."""

    __slots__ = (
        "mode",
        "n_rows",
        "n_fine_trans",
        "parent_map",
        "coarse_valid",
        "coarse_excluded",
        "fine_translation_parent",
        "count",
    )

    def __init__(
        self,
        *,
        mode: str,
        n_rows: int,
        n_fine_trans: int,
        parent_map=None,
        coarse_valid=None,
        coarse_excluded=None,
        fine_translation_parent=None,
        count: int | None = None,
    ):
        self.mode = str(mode)
        self.n_rows = int(n_rows)
        self.n_fine_trans = int(n_fine_trans)
        self.parent_map = None if parent_map is None else np.asarray(parent_map, dtype=np.int32)
        self.coarse_valid = None if coarse_valid is None else np.asarray(coarse_valid, dtype=bool)
        self.coarse_excluded = None if coarse_excluded is None else np.asarray(coarse_excluded, dtype=np.int32)
        self.fine_translation_parent = (
            None if fine_translation_parent is None else np.asarray(fine_translation_parent, dtype=np.int32)
        )
        self.count = int(_dense_candidate_mask_from_spec(self).sum()) if count is None else int(count)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_rows, self.n_fine_trans)

    def __array__(self, dtype=None, copy=None):
        dense = _dense_candidate_mask_from_spec(self)
        if dtype is not None:
            return dense.astype(dtype, copy=False if copy is None else bool(copy))
        if copy:
            return dense.copy()
        return dense


def _dense_candidate_mask_from_spec(mask: SparseCandidateMask) -> np.ndarray:
    if mask.mode == "full":
        return np.ones(mask.shape, dtype=bool)
    if mask.mode == "empty":
        return np.zeros(mask.shape, dtype=bool)
    if mask.mode == "coarse":
        if mask.coarse_valid is None or mask.parent_map is None or mask.fine_translation_parent is None:
            raise ValueError("coarse candidate mask spec is missing parent/coarse arrays")
        return mask.coarse_valid[:, mask.fine_translation_parent][mask.parent_map]
    if mask.mode == "coarse_exclude":
        if mask.coarse_excluded is None or mask.parent_map is None or mask.fine_translation_parent is None:
            raise ValueError("coarse_exclude candidate mask spec is missing excluded/parent arrays")
        dense = np.ones(mask.shape, dtype=bool)
        excluded = np.asarray(mask.coarse_excluded, dtype=np.int64).reshape(-1)
        if excluded.size:
            n_coarse_trans = int(mask.fine_translation_parent.max(initial=-1) + 1)
            if n_coarse_trans <= 0:
                raise ValueError("coarse_exclude candidate mask has empty translation parent map")
            excluded_rot = excluded // n_coarse_trans
            excluded_trans = excluded % n_coarse_trans
            for coarse_rot, coarse_trans in zip(excluded_rot.tolist(), excluded_trans.tolist(), strict=False):
                rows = np.flatnonzero(mask.parent_map == int(coarse_rot))
                cols = np.flatnonzero(mask.fine_translation_parent == int(coarse_trans))
                if rows.size and cols.size:
                    dense[np.ix_(rows, cols)] = False
        return dense
    raise ValueError(f"Unknown sparse candidate mask mode {mask.mode!r}")


