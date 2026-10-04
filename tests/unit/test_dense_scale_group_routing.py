"""Global scoring runs one engine at every oversampling order, with or without scale groups.

RELION's ``storeWeightedSums`` accumulates the group-scale ``XA``/``AA`` sums
and the norm-correction residuals in every expectation pass regardless of
oversampling. The adaptive/sparse engine does, and since the direct dense engine
was removed (2026-10-03) it is the only route: oversampling 0 is its single pass
on the current grid.
"""

from __future__ import annotations

import inspect

import pytest

from relax.refinement import half_scoring

pytestmark = pytest.mark.unit


def _routed_block(source, start, end):
    block = source[source.index(start) :]
    return block[: block.index(end)]


def test_there_is_no_direct_dense_route_left():
    for name in ("_dense_uses_adaptive_engine", "_score_direct_k1_dense", "_score_direct_kclass_dense"):
        assert not hasattr(half_scoring, name)
    source = inspect.getsource(half_scoring._score_half_dense_one_shape)
    assert source.count("_score_adaptive_k1_dense(") == 1
    assert source.count("_score_adaptive_kclass_dense(") == 1
    assert "run_em(" not in inspect.getsource(half_scoring)
    assert "does not accumulate group XA/AA statistics" not in source


def test_k1_oversampling_zero_is_the_single_pass_on_the_current_grid():
    adaptive = inspect.getsource(half_scoring._score_adaptive_k1_dense)
    routed = _routed_block(adaptive, "if adaptive_os <= 0:", "relion_x_half_mstep =")
    assert "coarse_current_size = sampling.cs_for_engine" in routed
    assert "fine_current_size = sampling.cs_for_engine" in routed


def test_k_class_oversampling_zero_is_the_single_pass_on_the_current_grid():
    source = inspect.getsource(half_scoring._score_half_dense_one_shape)
    adaptive = inspect.getsource(half_scoring._score_adaptive_kclass_dense)
    routed = _routed_block(adaptive, "if adaptive_os <= 0:", "pass2_grids = prepare_adaptive_pass2_grids(")
    assert "coarse_current_size = sampling.cs_for_engine" in routed
    assert "fine_current_size = sampling.cs_for_engine" in routed
    # RELION keeps two passes under adaptive oversampling even when coarse_size == current_size.
    assert "firstiter_coarse_current_size is not None and int(" not in source
    assert "or firstiter_coarse_current_size is not None" not in source
