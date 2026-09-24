"""Pin RELION 5.0.1 (f2c1a384) AutoRefine and Class3D particle orders."""
from types import SimpleNamespace

import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.helpers.expected_accuracy import relion_auto_refine_half_orders, relion_class3d_trial_layout

pytestmark = pytest.mark.unit


def test_mt19937_paired_reference():
    from relax.relion_bind import _relion_bind_core as bind

    # Independent GCC 11 transcription of RELION f2c1a384 exp_model.cpp;
    # importantly, half 2 continues the generator consumed by half 1.
    first, second = bind.auto_refine_randomise_half_orders_mt19937(10, 7, 1712)
    assert_matches(first, [5, 6, 8, 4, 7, 0, 2, 1, 9, 3])
    assert_matches(second, [5, 0, 2, 3, 4, 1, 6])


def test_python_dispatch_and_stable_optics_sort(monkeypatch):
    from relax import relion_bind

    calls = []

    def shuffled(n1, n2, seed):
        calls.append((n1, n2, seed))
        return [2, 0, 1], [1, 0]

    monkeypatch.setattr(
        relion_bind, '_relion_bind_core', SimpleNamespace(auto_refine_randomise_half_orders_mt19937=shuffled)
    )
    first, second = relion_auto_refine_half_orders([1, 2, 1, 2, 1], 42, optics_group_ids=[2, 1, 1, 1, 2])
    assert calls == [(3, 2, 43)]
    assert_matches(first, [2, 4, 0])
    assert_matches(second, [3, 1])


def test_missing_binding_fails_closed(monkeypatch):
    from relax import relion_bind

    monkeypatch.setattr(relion_bind, '_relion_bind_core', SimpleNamespace())
    with pytest.raises(RuntimeError, match='mt19937'):
        relion_auto_refine_half_orders([1, 2], 42)


def test_half1_trial_order_is_the_paired_first_half():
    from relax.helpers.expected_accuracy import relion_half1_trial_order
    from relax.relion_bind import _relion_bind_core as bind

    # Half 1 is shuffled first from a fresh generator, so it does not depend on half 2.
    first, _second = bind.auto_refine_randomise_half_orders_mt19937(10, 7, 1712)
    assert_matches(relion_half1_trial_order(10, 1711), first)


def test_class3d_whole_vector_shuffle_is_the_first_half_generator():
    # Class3D shuffles the entire vector with the same fresh mt19937 draw that
    # AutoRefine spends on half 1 (exp_model.cpp:449-456).
    order, particle_ids = relion_class3d_trial_layout(np.arange(10), 1711)
    assert_matches(order, [5, 6, 8, 4, 7, 0, 2, 1, 9, 3])
    assert_matches(particle_ids, np.arange(10))


def test_class3d_layout_maps_part_ids_to_input_rows_then_sorts_optics(monkeypatch):
    from relax import relion_bind

    calls = []

    def shuffled(n1, n2, seed):
        calls.append((n1, n2, seed))
        return [3, 0, 2, 1], []

    monkeypatch.setattr(
        relion_bind, '_relion_bind_core', SimpleNamespace(auto_refine_randomise_half_orders_mt19937=shuffled)
    )
    # part_id j lives at input row sorted_rows[j].
    order, particle_ids = relion_class3d_trial_layout(
        [2, 0, 3, 1], 42, first_iteration=3, optics_group_ids=[1, 2, 1, 1]
    )
    assert calls == [(4, 0, 45)]
    assert_matches(order, [2, 3, 0, 1])
    assert_matches(particle_ids, [1, 3, 0, 2])


def test_class3d_layout_rejects_non_permutation():
    with pytest.raises(ValueError, match='permutation'):
        relion_class3d_trial_layout([0, 0, 1], 42)
