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


def test_deferred_check_scope_defers_inside_and_drains_on_normal_exit(monkeypatch):
    """Inside the scope the kernel call defers by default; leaving it drains, and an error skips the drain."""
    from relax.cuda import kernels as em_cuda_kernels

    seen = []

    def fake_jit(images, factors, shifts, radius, width, apply_mask, lane, atomic, check_now):
        seen.append(bool(check_now))
        return images, images, jnp.zeros((1,), dtype=jnp.int32)

    drains = []
    monkeypatch.delenv(em_cuda_kernels.RELION_PREPROCESS_DEFERRED_CHECK_ENV, raising=False)
    monkeypatch.setattr(em_cuda_kernels, "_relion_preprocess_real_f32_jit", fake_jit)
    monkeypatch.setattr(em_cuda_kernels, "_queue_relion_preprocess_check", lambda count: None)
    monkeypatch.setattr(em_cuda_kernels, "drain_relion_preprocess_checks", lambda: drains.append(1))
    call = lambda: em_cuda_kernels.relion_preprocess_real_f32(jnp.zeros((1, 4, 4)), None, None, 1.0, 1.0)  # noqa: E731

    call()
    with em_cuda_kernels.deferred_relion_preprocess_checks():
        call()
        with em_cuda_kernels.deferred_relion_preprocess_checks():
            call()
        call()
    call()
    # check_now: synchronous outside the scope, deferred inside (nested scopes included).
    assert seen == [True, False, False, False, True]
    assert len(drains) == 2  # one per scope left normally

    with pytest.raises(KeyError):
        with em_cuda_kernels.deferred_relion_preprocess_checks():
            raise KeyError("loop failure")
    assert len(drains) == 2 and em_cuda_kernels._RELION_PREPROCESS_DEFERRED_SCOPES == 0
