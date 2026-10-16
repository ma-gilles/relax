from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from recovar.cuda_backproject import _infer_backproject_upsampling, _project_ffi_kwargs

from relax.fourier.fourier_window import (
    centered_half_indices_to_fftw_half_indices,
    make_fourier_window_indices_np,
)

pytestmark = pytest.mark.unit


def test_infer_upsampling_accepts_standard_even_grid():
    assert _infer_backproject_upsampling((128, 128), (256, 256, 256)) == 2


def test_infer_upsampling_accepts_standard_rectangular_grid():
    assert _infer_backproject_upsampling((4, 8), (8, 10, 12)) == 2


def test_infer_upsampling_accepts_final_all_data_relion_pad_without_max_r():
    assert _infer_backproject_upsampling((128, 128), (259, 259, 259), max_r=None) == 2


@pytest.mark.parametrize(
    "image_shape,volume_shape,max_r,expected_ups",
    [
        ((128, 128), (259, 259, 259), 64, 2),
        ((128, 128), (259, 259, 259), 28, 2),
        ((128, 128), (131, 131, 131), 32, 2),
        ((128, 128), (123, 123, 123), 30, 2),
        ((64, 64), (9, 9, 9), 2.5, 1),
    ],
)
def test_infer_upsampling_accepts_relion_odd_pad_size(image_shape, volume_shape, max_r, expected_ups):
    assert _infer_backproject_upsampling(image_shape, volume_shape, max_r=max_r) == expected_ups


def test_infer_upsampling_rejects_unexplained_odd_grid():
    with pytest.raises(ValueError, match="RELION pad_size"):
        _infer_backproject_upsampling((128, 128), (257, 257, 257))


def test_centered_half_indices_to_fftw_half_indices_remaps_rows_only():
    image_shape = (128, 128)
    half_width = 65
    centered = [
        64 * half_width + 0,  # ky=0 -> FFTW row 0
        36 * half_width + 5,  # ky=-28 -> FFTW row 100
        92 * half_width + 7,  # ky=28 -> FFTW row 28
    ]

    mapped = centered_half_indices_to_fftw_half_indices(image_shape, centered)

    assert mapped.tolist() == [
        0 * half_width + 0,
        100 * half_width + 5,
        28 * half_width + 7,
    ]


def _jitted_centered_to_fftw_half_indices(indices, image_shape):
    """The removed jitted map (relax#95), kept here as the reference the host map must equal."""

    @partial(jax.jit, static_argnames=("shape",))
    def remap(values, shape):
        height, width = shape
        half_width = width // 2 + 1
        rows = values // half_width
        cols = values % half_width
        return (((rows + height // 2) % height) * half_width + cols).astype(jnp.int32)

    return remap(jnp.asarray(indices, dtype=jnp.int32), tuple(int(v) for v in image_shape))


@pytest.mark.parametrize("current_size", list(range(38, 129, 2)))
def test_centered_half_indices_to_fftw_half_indices_equals_jitted_map(current_size):
    image_shape = (128, 128)
    for square in (False, True):
        indices, _ = make_fourier_window_indices_np(image_shape, current_size, square=square)
        mapped = centered_half_indices_to_fftw_half_indices(image_shape, indices)
        assert mapped.dtype == jnp.int32
        np.testing.assert_array_equal(
            np.asarray(mapped), np.asarray(_jitted_centered_to_fftw_half_indices(indices, image_shape))
        )


@pytest.mark.parametrize("image_shape", [(127, 127), (129, 129), (130, 130), (126, 128)])
def test_centered_half_indices_to_fftw_half_indices_equals_jitted_map_full_half(image_shape):
    indices = np.arange(int(image_shape[0]) * (int(image_shape[1]) // 2 + 1), dtype=np.int32)
    mapped = centered_half_indices_to_fftw_half_indices(image_shape, jnp.asarray(indices))
    np.testing.assert_array_equal(
        np.asarray(mapped), np.asarray(_jitted_centered_to_fftw_half_indices(indices, image_shape))
    )


def test_relion_texture_uses_one_canonical_boolean_ffi_mode(monkeypatch):
    args = ((56, 56), (115, 115, 115), 1, False, True, 28)

    # The retired diagnostic environment variable must not create a hidden
    # third mode: True always means the production RELION texture convention.
    monkeypatch.setenv("RELAX_RELION_TEXTURE_POSITIVE_NYQUIST", "1")
    texture, _, _ = _project_ffi_kwargs(*args, relion_texture_interp=True)
    assert int(texture["relion_texture_interp"]) == 1

    non_texture, _, _ = _project_ffi_kwargs(*args, relion_texture_interp=False)
    assert int(non_texture["relion_texture_interp"]) == 0
