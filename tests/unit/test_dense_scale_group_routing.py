"""Dense scoring routes RELION scale groups through the engine that accumulates them.

RELION's ``storeWeightedSums`` accumulates the group-scale ``XA``/``AA`` sums
and the norm-correction residuals in every expectation pass regardless of
oversampling; only RECOVAR's adaptive/sparse engine does, so scale groups
select it even at oversampling 0 for K=1 and for K-class scoring.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from relax.refinement import half_scoring

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("oversampling", [0, 1, 2])
def test_scale_groups_always_select_the_adaptive_engine(oversampling):
    assert half_scoring._dense_uses_adaptive_engine(oversampling, np.zeros(4, dtype=np.int32)) is True


def test_without_scale_groups_only_positive_oversampling_selects_it():
    assert half_scoring._dense_uses_adaptive_engine(0, None) is False
    assert half_scoring._dense_uses_adaptive_engine(1, None) is True
    assert half_scoring._dense_uses_adaptive_engine(np.int64(0), None) is False


def _routed_block(source, start, end):
    block = source[source.index(start) :]
    return block[: block.index(end)]


def test_k1_dense_scorer_uses_the_routing_rule_and_no_longer_rejects_scale_groups():
    source = inspect.getsource(half_scoring._score_half_dense_one_shape)
    gate = (
        "if execution.preserve_bpref_particle_order or _dense_uses_adaptive_engine(\n"
        "        sampling.state.adaptive_oversampling, half.group_ids_k\n"
        "    ):"
    )
    assert source.count(gate) == 1
    assert "does not accumulate group XA/AA statistics" not in source
    assert "_score_adaptive_k1_dense(" in source[source.index(gate) :]
    # Oversampling 0 with scale groups is RELION's single pass on the current grid.
    adaptive = inspect.getsource(half_scoring._score_adaptive_k1_dense)
    routed = _routed_block(adaptive, "if adaptive_os <= 0:", "relion_x_half_mstep =")
    assert "coarse_current_size = sampling.cs_for_engine" in routed
    assert "fine_current_size = sampling.cs_for_engine" in routed


def test_k_class_dense_scorer_routes_scale_groups_at_oversampling_zero():
    source = inspect.getsource(half_scoring._score_half_dense_one_shape)
    gate = "elif _dense_uses_adaptive_engine(sampling.state.adaptive_oversampling, half.group_ids_k):"
    assert source.count(gate) == 1
    assert "_score_adaptive_kclass_dense(" in _routed_block(source, gate, "else:")
    adaptive = inspect.getsource(half_scoring._score_adaptive_kclass_dense)
    routed = _routed_block(adaptive, "if adaptive_os <= 0:", "pass2_grids = _adaptive_pass2_grids(")
    assert "coarse_current_size = sampling.cs_for_engine" in routed
    assert "fine_current_size = sampling.cs_for_engine" in routed
    assert "firstiter_coarse_current_size is not None and int(" not in source


def test_k_class_positive_oversampling_never_drops_to_the_direct_engine():
    """RELION keeps two passes under adaptive oversampling even when coarse_size == current_size."""
    source = inspect.getsource(half_scoring._score_half_dense_one_shape)
    gate = "elif _dense_uses_adaptive_engine(sampling.state.adaptive_oversampling, half.group_ids_k):"
    assert "firstiter_coarse_current_size is not None" not in _routed_block(source, "if variant.k_class_enabled:", gate)
    assert "or firstiter_coarse_current_size is not None" not in source
    assert half_scoring._dense_uses_adaptive_engine(1, None) is True
