import numpy as np
import pytest
from helpers.array_padding import pad_axis
from helpers.float_compare import assert_matches

from relax.helpers.shape_buckets import (
    coarse_bucket,
    pow2_ceil,
    pow2_floor,
    power_bucket,
    power_of_two_bucket,
    round_up_to_multiple,
)


def test_round_up_to_multiple():
    assert round_up_to_multiple(0, 8) == 0
    assert round_up_to_multiple(1, 8) == 8
    assert round_up_to_multiple(16, 8) == 16
    assert round_up_to_multiple(17, 8) == 24


def test_pow2_ceil_and_floor_match_the_bit_length_rules_they_replace():
    for value in range(-3, 300):
        for minimum in (0, 1, 2, 64):
            assert pow2_ceil(value, minimum=minimum) == 1 << (max(value, minimum) - 1).bit_length()
        if value >= 1:
            assert pow2_floor(value) == 1 << (value.bit_length() - 1)
    assert [pow2_ceil(v) for v in (0, 1, 2, 3, 64, 65)] == [1, 1, 2, 4, 64, 128]
    assert [pow2_floor(v) for v in (1, 2, 3, 64, 65)] == [1, 2, 2, 64, 64]
    with pytest.raises(ValueError, match="positive"):
        pow2_floor(0)


def test_power_of_two_bucket():
    assert power_of_two_bucket(1, minimum=16, maximum=4096) == 16
    assert power_of_two_bucket(17, minimum=16, maximum=4096) == 32
    assert power_of_two_bucket(4097, minimum=16, maximum=4096) == 4097


def test_power_bucket_can_use_low_cardinality_radix_four_classes():
    assert power_bucket(17, base=4, minimum=16, maximum=4096) == 64
    assert power_bucket(65, base=4, minimum=16, maximum=4096) == 256
    assert power_bucket(257, base=4, minimum=16, maximum=4096) == 1024


def test_coarse_bucket():
    assert coarse_bucket(1, small_power2_max=64, large_multiple=16, minimum=16) == 16
    assert coarse_bucket(33, small_power2_max=64, large_multiple=16) == 64
    assert coarse_bucket(65, small_power2_max=64, large_multiple=16) == 80


def test_pad_axis_preserves_values_and_fills_constant():
    arr = np.arange(6).reshape(2, 3)
    padded = pad_axis(arr, 1, 5, value=-1)
    assert_matches(padded[:, :3], arr)
    assert_matches(padded[:, 3:], -np.ones((2, 2), dtype=arr.dtype))


def test_pad_axis_rejects_truncation():
    with pytest.raises(ValueError, match="cannot pad"):
        pad_axis(np.zeros((2, 3)), 1, 2)
