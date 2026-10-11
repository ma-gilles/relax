"""Compact masks retain their stored identity through pickling."""

import pickle

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.scoring.compact_candidates import SparseCandidateMask

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("protocol", [4, 5])
@pytest.mark.parametrize("mode", ["full", "empty", "coarse", "coarse_exclude"])
def test_sparse_mask_roundtrip(mode, protocol):
    mask = SparseCandidateMask(
        mode=mode,
        n_rows=2,
        n_fine_trans=3,
        parent_map=np.array([1, 0]),
        fine_translation_parent=np.array([0, 1, 0]),
        coarse_valid=np.array([[True, False], [False, True]]),
        coarse_excluded=np.array([1]),
    )
    expected = {
        "full": [[True, True, True], [True, True, True]],
        "empty": [[False, False, False], [False, False, False]],
        "coarse": [[False, True, False], [True, False, True]],
        "coarse_exclude": [[True, True, True], [True, False, True]],
    }[mode]
    assert_matches(np.asarray(mask), expected)
    assert mask.count == np.count_nonzero(expected)
    packed = pickle.dumps(mask, protocol=protocol)
    restored = pickle.loads(packed)
    assert type(restored) is SparseCandidateMask
    assert_matches(np.asarray(restored), expected)
