"""relax.relion.image_preprocessing.batched_fft_is_per_image: the batched cuFFT replaces the per-image one only
where a probe shows it bitwise equal."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.relion import image_preprocessing
from relax.relion.image_preprocessing import batched_fft_is_per_image, process_half_image


class _CapturingDataset:
    """Records the keyword arguments its image backend was called with."""

    def __init__(self):
        self.kwargs = None

    def process_images_half(self, batch, apply_image_mask=False, **kwargs):
        self.kwargs = kwargs
        return jnp.zeros((np.shape(batch)[0], 4), dtype=jnp.complex64)


def _process(monkeypatch, matches):
    monkeypatch.setattr(image_preprocessing, "batched_fft_is_per_image", lambda shape: matches)
    monkeypatch.setattr("relax.relion.optics_aberrations.demodulate_odd_aberrations", lambda dataset, x, idx: x)
    dataset = _CapturingDataset()
    process_half_image(
        dataset, np.zeros((3, 8, 8), np.float32), False, relion_preprocess_kwargs={"relion_fft_per_image": True}
    )
    return dataset.kwargs["relion_fft_per_image"]


def test_the_per_image_fft_stays_unless_the_probe_matches(monkeypatch):
    assert _process(monkeypatch, False) is True
    assert _process(monkeypatch, True) is False


@pytest.mark.skipif(jax.default_backend() == "gpu", reason="the CPU fallback")
def test_without_a_gpu_the_probe_keeps_the_per_image_fft():
    assert batched_fft_is_per_image((32, 96, 96)) is False


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
