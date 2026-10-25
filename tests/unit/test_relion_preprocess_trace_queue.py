"""A traced preprocessing status must never escape into a Python queue."""
import jax
import jax.numpy as jnp
import pytest
from helpers import cuda_targets
from helpers.float_compare import assert_matches

from relax.cuda import preprocess as cuda_preprocess


@pytest.mark.parametrize("traced", [False, True])
def test_deferred_check_only_queues_concrete_results(monkeypatch, traced):
    monkeypatch.setattr(cuda_preprocess, "_RELION_PREPROCESS_PENDING_CHECKS", [])
    seen = []
    def native(images, *args):
        seen.append(args[-1])
        return images, images, jnp.zeros((1,), dtype=jnp.int32)
    monkeypatch.setattr(cuda_preprocess, "_relion_preprocess_real_f32_jit", native)
    def call(images):
        return cuda_preprocess.relion_preprocess_real_f32(images, None, None, 1., 1., deferred_finite_check=True)[1]
    images=jnp.ones((1,2,2),dtype=jnp.float32)
    actual=(jax.jit(call) if traced else call)(images)
    assert_matches(actual, images)
    assert seen == [traced]
    assert cuda_targets.pending_relion_preprocess_checks() == (0 if traced else 1)
    assert cuda_preprocess.drain_relion_preprocess_checks() == (0 if traced else 1)


def test_deferred_invalid_status_fails_and_drains(monkeypatch):
    monkeypatch.setattr(cuda_preprocess, "_RELION_PREPROCESS_PENDING_CHECKS", [])
    cuda_preprocess._queue_relion_preprocess_check(jnp.asarray([2],dtype=jnp.int32))
    with pytest.raises(RuntimeError, match="2 image"):
        cuda_preprocess.drain_relion_preprocess_checks()
    assert cuda_targets.pending_relion_preprocess_checks() == 0


def _run_expectation(run_half):
    from types import SimpleNamespace

    from relax.refinement.expectation import run_numbered_halves

    combined = []
    # Each half's inputs are its index; the score is run_half and nothing is finished.
    run_numbered_halves(
        run_half, lambda half, result: None, (0, 1), (0, 1), SimpleNamespace(combine=lambda: combined.append(1)),
        remote_halves=None,
    )
    return combined


def test_an_expectation_that_raises_leaves_no_check_for_a_later_run(monkeypatch):
    monkeypatch.setattr(cuda_preprocess, "_RELION_PREPROCESS_PENDING_CHECKS", [])

    def run_half(k):
        # Half 0 queues an invalid image's count (an unmasked pass-2 preparation), half 1 then fails.
        cuda_preprocess._queue_relion_preprocess_check(jnp.asarray([1, 0], dtype=jnp.int32))
        if k == 1:
            raise ValueError("half 1 failed")

    with pytest.raises(ValueError, match="half 1 failed"):
        _run_expectation(run_half)
    assert cuda_targets.pending_relion_preprocess_checks() == 0
    # The next run in the process reads only its own checks.
    assert _run_expectation(lambda k: cuda_preprocess._queue_relion_preprocess_check(jnp.zeros(2, jnp.int32))) == [1]
    assert cuda_targets.pending_relion_preprocess_checks() == 0


def test_an_expectation_reads_its_checks_when_the_halves_are_done(monkeypatch):
    monkeypatch.setattr(cuda_preprocess, "_RELION_PREPROCESS_PENDING_CHECKS", [])
    with pytest.raises(RuntimeError, match="1 image"):
        _run_expectation(lambda k: cuda_preprocess._queue_relion_preprocess_check(jnp.asarray([k, 0], jnp.int32)))
    assert cuda_targets.pending_relion_preprocess_checks() == 0
