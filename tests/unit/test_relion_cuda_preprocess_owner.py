"""RELION CUDA image-preprocessing detection has one owner, and the fresh K=1 defaults fail closed early without it."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from relax.helpers import preprocessing
from relax.vdam import adaptive_estep

pytestmark = pytest.mark.unit


def _dataset(backend_name, *, depth=0):
    backend = SimpleNamespace(relion_fourier_backend=backend_name)
    source = SimpleNamespace(backend=backend)
    for _ in range(depth):
        source = SimpleNamespace(parent=source)
    return SimpleNamespace(image_source=source)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_detection_follows_subset_parents(depth):
    assert preprocessing.uses_relion_cuda_image_preprocessing(_dataset("relion_cuda", depth=depth)) is True
    assert preprocessing.uses_relion_cuda_image_preprocessing(_dataset("host_numpy", depth=depth)) is False
    assert preprocessing.relion_preprocess_backend(_dataset("relion_cuda", depth=depth)).relion_fourier_backend == "relion_cuda"


def test_datasets_without_a_backend_are_not_cuda():
    assert preprocessing.uses_relion_cuda_image_preprocessing(SimpleNamespace()) is False
    assert preprocessing.uses_relion_cuda_image_preprocessing(SimpleNamespace(image_source=object())) is False


def test_initial_model_patch_point_is_the_owner():
    assert adaptive_estep.uses_relion_cuda_image_preprocessing is preprocessing.uses_relion_cuda_image_preprocessing


def _fake_preprocess(monkeypatch, invalid_counts):
    """Replace the kernel program: call ``i`` reports ``invalid_counts[i]`` invalid images. Returns the call log."""
    from relax.cuda import kernels as em_cuda_kernels

    seen = []

    def fake_jit(images, factors, shifts, radius, width, apply_mask, lane, atomic, check_now):
        seen.append(bool(check_now))
        return images, images, jnp.asarray([invalid_counts[len(seen) - 1]], dtype=jnp.int32)

    monkeypatch.delenv(em_cuda_kernels.RELION_PREPROCESS_DEFERRED_CHECK_ENV, raising=False)
    monkeypatch.setattr(em_cuda_kernels, "_relion_preprocess_real_f32_jit", fake_jit)
    monkeypatch.setattr(em_cuda_kernels, "_RELION_PREPROCESS_PENDING_CHECKS", [])
    monkeypatch.setattr(em_cuda_kernels, "_RELION_PREPROCESS_DEFERRED_SCOPES", [])
    return seen, lambda: em_cuda_kernels.relion_preprocess_real_f32(jnp.zeros((1, 4, 4)), None, None, 1.0, 1.0)


def test_deferred_check_scope_defers_inside_and_checks_on_normal_exit(monkeypatch):
    """Inside the scope the kernel call defers by default (nested scopes included); a clean scope leaves nothing."""
    from relax.cuda import kernels as em_cuda_kernels

    seen, call = _fake_preprocess(monkeypatch, [0] * 5)
    call()
    with em_cuda_kernels.deferred_relion_preprocess_checks("outer") as outer:
        call()
        with em_cuda_kernels.deferred_relion_preprocess_checks("inner") as inner:
            call()
            assert len(inner.pending) == 1 and len(outer.pending) == 1
        call()
        assert len(outer.pending) == 2
    call()
    # check_now: synchronous outside the scope, deferred inside.
    assert seen == [True, False, False, False, True]
    assert outer.pending == [] and inner.pending == []
    assert em_cuda_kernels._RELION_PREPROCESS_DEFERRED_SCOPES == []
    assert em_cuda_kernels.pending_relion_preprocess_checks() == 0


def test_deferred_check_scope_names_the_pass_and_the_batches(monkeypatch):
    """An invalid image raises when the scope ends, naming the scope and each batch that held one."""
    from relax.cuda import kernels as em_cuda_kernels

    _, call = _fake_preprocess(monkeypatch, [0, 2, 0, 1])
    with pytest.raises(RuntimeError) as failure:
        with em_cuda_kernels.deferred_relion_preprocess_checks("pass 1") as checks:
            for batch in range(4):
                checks.at(f"batch {batch} (images {10 * batch}-{10 * batch + 9})")
                call()
    message = str(failure.value)
    assert "deferred check, pass 1" in message and "3 image(s)" in message
    assert "batch 1 (images 10-19): 2 image(s)" in message and "batch 3 (images 30-39): 1 image(s)" in message
    assert "batch 0" not in message and "batch 2" not in message
    assert em_cuda_kernels._RELION_PREPROCESS_DEFERRED_SCOPES == []


def test_deferred_check_scope_drops_its_checks_when_the_loop_raises(monkeypatch):
    """A loop failure propagates unchanged, and its unread checks cannot surface in a later scope or drain."""
    from relax.cuda import kernels as em_cuda_kernels

    _, call = _fake_preprocess(monkeypatch, [3, 0])
    with pytest.raises(KeyError, match="loop failure"):
        with em_cuda_kernels.deferred_relion_preprocess_checks("pass 1"):
            call()  # an invalid image is queued, then the loop fails for another reason
            raise KeyError("loop failure")
    assert em_cuda_kernels._RELION_PREPROCESS_DEFERRED_SCOPES == []
    assert em_cuda_kernels.pending_relion_preprocess_checks() == 0
    assert em_cuda_kernels.drain_relion_preprocess_checks() == 0
    with em_cuda_kernels.deferred_relion_preprocess_checks("next pass"):
        call()


def test_deferred_check_scope_reads_a_full_queue_early(monkeypatch):
    """A scope never holds more than the pending limit: a full queue is read (and can fail) in the loop."""
    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(em_cuda_kernels, "_RELION_PREPROCESS_PENDING_LIMIT", 2)
    _, call = _fake_preprocess(monkeypatch, [0, 0, 0, 1, 0])
    calls = 0
    with pytest.raises(RuntimeError, match="batch 3: 1 image"):
        with em_cuda_kernels.deferred_relion_preprocess_checks("pass 1") as checks:
            for batch in range(5):
                checks.at(f"batch {batch}")
                call()
                calls += 1
                assert len(checks.pending) < 2
    assert calls == 3  # the fourth call filled the queue and raised inside the loop
