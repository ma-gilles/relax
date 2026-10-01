"""Coarse support encodings retain exact IDs and stored NamedTuple identity."""

import pickle

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.scoring.significant_samples import (
    ComplementSignificantSampleIndices,
    significant_sample_ids,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("protocol", [4, 5])
def test_complement_support_roundtrip(protocol):
    samples = ComplementSignificantSampleIndices(np.array([1, 4], dtype=np.int32), 6)
    assert samples._fields == ("excluded_indices", "total_size")
    assert samples.__annotations__ == {"excluded_indices": np.ndarray, "total_size": int}
    packed = pickle.dumps(samples, protocol=protocol)
    restored = pickle.loads(packed)
    assert type(restored) is ComplementSignificantSampleIndices
    assert restored.size == 4
    assert restored.excluded_indices.dtype == np.int32
    assert_matches(significant_sample_ids(restored, 6), [0, 2, 3, 5])
