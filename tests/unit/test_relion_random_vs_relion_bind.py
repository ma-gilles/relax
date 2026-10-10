"""relax's glibc ``rand`` and ``std::shuffle(mt19937)`` against RELION's binding (the oracle)."""

from __future__ import annotations

import numpy as np
import pytest

from relax.numerics import relion_random

bind = pytest.importorskip("relax.relion_bind._relion_bind_core")


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 43, 1000, 123456789])
def test_rnd_unif_sequence_matches(seed):
    expected = np.asarray(bind.vdam_rnd_unif_sequence(seed, 700), dtype=np.float64)
    actual = relion_random.rnd_unif_sequence(seed, 700).astype(np.float64)
    # Both are float32 values; the draws are discrete, so they agree exactly.
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("low", "high"), [(0.5, 1.0), (-0.25, 0.75), (0.0, 360.0), (2.0, 2.0)])
def test_rnd_unif_range_sequence_matches(low, high):
    expected = np.asarray(bind.vdam_rnd_unif_range_sequence(7, 300, low, high), dtype=np.float64)
    actual = relion_random.rnd_unif_sequence(7, 300, low, high).astype(np.float64)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 10, 11, 1000, 1001, 65535, 65536, 70001, 200000])
@pytest.mark.parametrize("seed", [1, 43])
def test_single_shuffle_matches(n, seed):
    expected = np.asarray(bind.vdam_randomise_particles_order(n, seed), dtype=np.int64)
    actual = relion_random.shuffled_orders([n], seed)[0]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(("n1", "n2"), [(5000, 5001), (1, 0), (0, 7), (70000, 69999), (123, 65536)])
def test_paired_half_shuffles_match(n1, n2):
    expected = bind.auto_refine_randomise_half_orders_mt19937(n1, n2, 44)
    actual = relion_random.shuffled_orders([n1, n2], 44)
    for got, want in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got, np.asarray(want, dtype=np.int64))
