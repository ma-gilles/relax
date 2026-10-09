"""Global scoring runs one engine at every oversampling order, with or without scale groups.

RELION's ``storeWeightedSums`` accumulates the group-scale ``XA``/``AA`` sums
and the norm-correction residuals in every expectation pass regardless of
oversampling. The adaptive/sparse engine does, and since the direct dense engine
was removed (2026-10-03) it is the only route: oversampling 0 is its single pass
on the current grid.
"""

from __future__ import annotations

import pytest
from helpers.run_options import stand_in
from helpers.tiny_refinement import CallTrace, run_tiny_refinement

from relax.refinement import half_scoring

pytestmark = pytest.mark.unit

_ADAPTIVE_SCORER = {1: "_score_adaptive_k1_dense", 2: "_score_adaptive_kclass_dense"}


def _dense_route(monkeypatch, *, n_classes, oversampling):
    """The numbered dense scorings of a tiny run and the mode scorer called under each."""
    trace = CallTrace(monkeypatch)
    trace.wrap(half_scoring, "_score_half_dense_one_shape", "dense")
    for name in _ADAPTIVE_SCORER.values():
        trace.wrap(half_scoring, name)
    run_tiny_refinement(
        monkeypatch, n_classes=n_classes, final_after_max_iter=False,
        adaptive=stand_in.adaptive(adaptive_oversampling=oversampling),
    )
    return trace


@pytest.mark.parametrize("oversampling", [0, 1])
@pytest.mark.parametrize("n_classes", [1, 2])
def test_there_is_no_direct_dense_route_left(monkeypatch, n_classes, oversampling):
    for name in ("_dense_uses_adaptive_engine", "_score_direct_k1_dense", "_score_direct_kclass_dense", "run_em"):
        assert not hasattr(half_scoring, name)
    trace = _dense_route(monkeypatch, n_classes=n_classes, oversampling=oversampling)
    dense = trace.calls("dense")
    assert len(dense) == 4
    scorer = _ADAPTIVE_SCORER[n_classes]
    assert [call.label for call in trace.calls_seen if call.label != "dense"] == [scorer] * 4
    assert all(call.inside == ("dense",) for call in trace.calls(scorer))


def _engine_sizes(monkeypatch, *, n_classes, oversampling):
    """Per numbered half scoring: the sampling's engine current size and the engine call's keywords."""
    engine_calls = []
    trace = CallTrace(monkeypatch).wrap(half_scoring, "_score_half_dense_one_shape", "dense")
    run_tiny_refinement(
        monkeypatch, n_classes=n_classes, final_after_max_iter=False, engine_calls=engine_calls,
        adaptive=stand_in.adaptive(adaptive_oversampling=oversampling),
    )
    dense = trace.calls("dense")
    assert len(engine_calls) == len(dense) == 4
    return [(call.args[1].image_window_size, engine["kwargs"]) for call, engine in zip(dense, engine_calls, strict=True)]


@pytest.mark.parametrize("n_classes", [1, 2])
def test_oversampling_zero_is_the_single_pass_on_the_current_grid(monkeypatch, n_classes):
    for current_size, kwargs in _engine_sizes(monkeypatch, n_classes=n_classes, oversampling=0):
        assert kwargs["oversampling_order"] == 0
        assert kwargs["coarse_current_size"] == kwargs["fine_current_size"] == current_size


@pytest.mark.parametrize("n_classes", [1, 2])
def test_adaptive_oversampling_keeps_two_passes(monkeypatch, n_classes):
    """RELION keeps two passes under adaptive oversampling even when coarse_size == current_size."""
    for _, kwargs in _engine_sizes(monkeypatch, n_classes=n_classes, oversampling=1):
        assert kwargs["oversampling_order"] == 1
