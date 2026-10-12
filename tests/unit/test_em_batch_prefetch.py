"""Ordered EM prefetch must preserve outputs and release stopped producers."""

import threading

import pytest

from relax.io.batch_fetch import prefetch_depth, prefetched_batches

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_order_and_producer_error(depth):
    def source():
        yield 1
        yield 2
        raise RuntimeError("producer failed")

    values = []
    with pytest.raises(RuntimeError, match="producer failed"):
        with prefetched_batches(source(), depth=depth) as batches:
            values.extend(batches)
    assert values == [1, 2]
    with prefetched_batches(range(20), depth=depth) as batches:
        assert list(batches) == list(range(20))


@pytest.mark.parametrize("consumer_error", [False, True])
def test_exit_releases_full_queue(consumer_error):
    blocked = threading.Event()
    stopped = threading.Event()

    def source():
        try:
            yield 0
            yield 1
            blocked.set()
            yield 2
        finally:
            stopped.set()

    try:
        with prefetched_batches(source(), depth=1) as batches:
            assert next(batches) == 0
            assert blocked.wait(timeout=2)
            if consumer_error:
                raise RuntimeError("consumer failed")
    except RuntimeError as exc:
        assert consumer_error and str(exc) == "consumer failed"
    assert stopped.wait(timeout=2)


def test_default_depth_is_two_and_zero_turns_it_off(monkeypatch):
    monkeypatch.delenv("RELAX_EM_PREFETCH_BATCHES", raising=False)
    assert prefetch_depth() == 2
    monkeypatch.setenv("RELAX_EM_PREFETCH_BATCHES", "0")
    assert prefetch_depth() == 0


@pytest.mark.parametrize("value", ["-1", "invalid", "1.5"])
def test_invalid_depth_rejected(monkeypatch, value):
    monkeypatch.setenv("RELAX_EM_PREFETCH_BATCHES", value)
    with pytest.raises(ValueError, match="non-negative integer"):
        prefetch_depth()


def test_coarse_prefetch_preserves_all_outputs(monkeypatch):
    from test_em_stage_glue_programs import _assert_significance_results_match, _significance_call

    from relax.scoring import significance

    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    args, kwargs = _significance_call(monkeypatch)
    threads = []
    original = args[0].iter_batches

    def capture(*a, **kw):
        for batch in original(*a, **kw):
            threads.append(threading.get_ident())
            yield batch

    monkeypatch.setattr(args[0], "iter_batches", capture)
    monkeypatch.setenv("RELAX_EM_PREFETCH_BATCHES", "0")
    expected = significance._compute_k_class_significance_batched(*args, **kwargs)
    assert threads and set(threads) == {threading.get_ident()}
    threads.clear()
    monkeypatch.setenv("RELAX_EM_PREFETCH_BATCHES", "2")
    actual = significance._compute_k_class_significance_batched(*args, **kwargs)
    assert threads and threading.get_ident() not in threads
    _assert_significance_results_match(actual, expected)
