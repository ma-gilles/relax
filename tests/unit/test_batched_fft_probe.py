"""relax.relion.image_preprocessing.batched_fft_is_per_image and per_image_fft_sub_batch: the batched cuFFT, of
the whole batch or of sub-batches, replaces the per-image one only where a probe shows it bitwise equal."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.relion import image_preprocessing
from relax.relion.image_preprocessing import batched_fft_is_per_image, per_image_fft_sub_batch, process_half_image


class _CapturingDataset:
    """Records the keyword arguments its image backend was called with."""

    def __init__(self):
        self.kwargs = None

    def process_images_half(self, batch, apply_image_mask=False, **kwargs):
        self.kwargs = kwargs
        return jnp.zeros((np.shape(batch)[0], 4), dtype=jnp.complex64)


def _process(monkeypatch, matches, sub_batch=1):
    monkeypatch.setattr(image_preprocessing, "batched_fft_is_per_image", lambda shape: matches)
    probed = []

    def probe(height, width):
        probed.append((height, width))
        return sub_batch

    monkeypatch.setattr(image_preprocessing, "per_image_fft_sub_batch", probe)
    monkeypatch.setattr("relax.relion.optics_aberrations.demodulate_odd_aberrations", lambda dataset, x, idx: x)
    dataset = _CapturingDataset()
    process_half_image(
        dataset, np.zeros((3, 8, 6), np.float32), False, relion_preprocess_kwargs={"relion_fft_per_image": True}
    )
    return dataset.kwargs, probed


def test_the_per_image_fft_stays_unless_the_probe_matches(monkeypatch):
    assert _process(monkeypatch, False) == ({"relion_fft_per_image": True}, [(8, 6)])
    assert _process(monkeypatch, True) == ({"relion_fft_per_image": False}, [])


def test_a_probed_sub_batch_runs_the_per_image_fft_that_many_images_per_plan(monkeypatch):
    kwargs, probed = _process(monkeypatch, False, sub_batch=8)
    assert kwargs == {"relion_fft_per_image": True, "relion_fft_sub_batch": 8}
    assert probed == [(8, 6)]


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="the CPU fallback")
def test_without_a_gpu_the_probe_keeps_the_per_image_fft():
    assert batched_fft_is_per_image((32, 96, 96)) is False


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="the CPU fallback")
def test_without_a_gpu_the_sub_batch_probe_keeps_one_image_per_plan():
    assert per_image_fft_sub_batch(256, 256) == 1


@pytest.mark.gpu
def test_a_probed_sub_batch_transforms_every_image_bitwise_as_the_per_image_fft(gpu_device):
    from recovar.data_io.image_backends import _centered_rfft2_jax_per_image, _centered_rfft2_jax_sub_batches

    probed = {box: per_image_fft_sub_batch(box, box) for box in (128, 256)}
    if all(sub_batch == 1 for sub_batch in probed.values()):
        pytest.skip(f"no probed shape transforms several images per plan on this GPU: {probed}")
    for box, sub_batch in probed.items():
        if sub_batch == 1:
            continue
        for n_images in (sub_batch, 5 * sub_batch + sub_batch // 2 + 1, 64):
            images = jnp.asarray(np.random.default_rng(n_images).standard_normal((n_images, box, box)) * 30.0 + 7.0,
                                 dtype=jnp.float32)
            sub_batched = np.asarray(_centered_rfft2_jax_sub_batches(images, sub_batch))
            per_image = np.asarray(_centered_rfft2_jax_per_image(images))
            assert np.array_equal(sub_batched.view(np.uint32), per_image.view(np.uint32)), (box, sub_batch, n_images)


@pytest.mark.gpu
def test_a_matched_shape_transforms_every_image_bitwise_as_the_per_image_fft(gpu_device):
    from recovar.data_io.image_backends import _centered_rfft2_jax_batched, _centered_rfft2_jax_per_image

    matched = [shape for shape in ((32, 96, 96), (256, 96, 96), (32, 192, 192)) if batched_fft_is_per_image(shape)]
    if not matched:
        pytest.skip("no probed shape uses the batched FFT on this GPU")
    for shape in matched:
        images = jnp.asarray(np.random.default_rng(5).standard_normal(shape) * 30.0 + 7.0, dtype=jnp.float32)
        batched = np.asarray(_centered_rfft2_jax_batched(images))
        per_image = np.asarray(_centered_rfft2_jax_per_image(images))
        assert np.array_equal(batched.view(np.uint32), per_image.view(np.uint32)), shape
