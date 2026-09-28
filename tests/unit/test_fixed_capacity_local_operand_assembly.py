"""Host contracts for sealed cache-once fixed-capacity operands."""

from __future__ import annotations

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.local.local_caches import _build_local_raw_cache

pytestmark = pytest.mark.unit


class _IndexedDataset:
    def __init__(self, *, returned_indices=None, truncate_raw=False):
        self.images = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        self.ctf_params = np.arange(4 * 3, dtype=np.float32).reshape(4, 3) + 100
        self.returned_indices = returned_indices
        self.truncate_raw = truncate_raw
        self.calls = 0
        self.requested_indices = []

    def iter_batches(self, batch_size, *, indices, by_image):
        assert by_image is False
        requested = np.asarray(indices, dtype=np.int64)
        self.calls += 1
        self.requested_indices.append(requested.copy())
        returned = requested if self.returned_indices is None else np.asarray(self.returned_indices, dtype=np.int64)
        raw = self.images[returned]
        if self.truncate_raw:
            raw = raw[:-1]
        yield (
            raw,
            None,
            None,
            self.ctf_params[returned],
            None,
            None,
            returned,
        )


def test_existing_shared_raw_cache_still_indexes_rows_by_returned_image_id():
    dataset = _IndexedDataset(returned_indices=(2, 0, 3, 1))
    expected_raw = dataset.images.copy()
    expected_ctf = dataset.ctf_params.copy()

    raw_cache, ctf_cache = _build_local_raw_cache(dataset, 4)

    assert dataset.calls == 1
    assert_matches(dataset.requested_indices[0], [0, 1, 2, 3])
    assert_matches(raw_cache, expected_raw)
    assert_matches(ctf_cache, expected_ctf)


