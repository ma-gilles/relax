"""RELION ``powerClass`` reproductions share one operand owner.

``_relion_powerclass_packed_image`` repacks centred rfft images into RELION's
unshifted ``Faux`` layout with RELION's amplitude convention,
and ``_relion_powerclass_operands`` adds the CUDA shell map and pixel validity.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches, matches

from relax.fine_pass import scoring

pytestmark = pytest.mark.unit

SIZE = 8
HALF = SIZE // 2 + 1


def _images(dtype, n=2):
    rng = np.random.default_rng(1)
    return (rng.standard_normal((n, SIZE * HALF)) + 1j * rng.standard_normal((n, SIZE * HALF))).astype(dtype)


@pytest.mark.parametrize("dtype,expected_real", [(np.complex64, jnp.float32), (np.complex128, jnp.float64)])
def test_packed_image_follows_relion_layout_and_amplitude(dtype, expected_real):
    x = _images(dtype)
    image, real_dtype, height, width, half = scoring._relion_powerclass_packed_image(x, image_shape=(SIZE, SIZE))
    assert (height, width, half) == (SIZE, SIZE, HALF) and real_dtype == expected_real
    expected = np.roll(x.reshape(-1, SIZE, HALF), -(SIZE // 2), axis=1).reshape(x.shape[0], -1) / (SIZE * SIZE)
    assert_matches(np.asarray(image), expected.astype(dtype))
    assert image.dtype == jnp.dtype(dtype)


def test_forced_complex64_matches_native_kernel_precision():
    image, real_dtype, *_ = scoring._relion_powerclass_packed_image(_images(np.complex128), image_shape=(SIZE, SIZE), dtype=jnp.complex64)
    assert image.dtype == jnp.complex64 and real_dtype == jnp.float32


@pytest.mark.parametrize("bad,shape,match", [
    (np.zeros((2, 40), np.complex64), (8, 10), "square images"),
    (np.zeros((2, 39), np.complex64), (8, 8), "flattened centred rfft"),
    (np.zeros((40,), np.complex64), (8, 8), "flattened centred rfft"),
])
def test_packed_image_rejects_wrong_layouts(bad, shape, match):
    with pytest.raises(ValueError, match=match):
        scoring._relion_powerclass_packed_image(bad, image_shape=shape)


def test_resolution_limit_is_static_or_traced():
    assert scoring._relion_powerclass_resolution_limit(None, None, image_width=SIZE) == SIZE // 2 + 1
    assert scoring._relion_powerclass_resolution_limit(6, None, image_width=SIZE) == 4
    traced = scoring._relion_powerclass_resolution_limit(6, np.asarray(4, dtype=np.int32), image_width=SIZE)
    assert isinstance(traced, jnp.ndarray) and int(traced) == 3


def test_operands_shell_and_validity_follow_the_cuda_kernel():
    ops = scoring._relion_powerclass_operands(_images(np.complex64), image_shape=(SIZE, SIZE), current_size=None, runtime_current_size=None)
    rows = np.arange(SIZE)[:, None]
    cols = np.arange(HALF)[None, :]
    signed = np.where(rows < HALF, rows, rows - SIZE)
    shell = np.rint(np.sqrt((cols * cols + signed * signed).astype(np.float32))).astype(np.int32)
    assert ops.shell.dtype == np.int32 and matches(ops.shell, shell)
    valid = ((shell > 0) & (shell < HALF) & ~((cols == 0) & (signed < 0))).reshape(-1)
    assert matches(ops.valid, valid) and ops.resolution_limit == HALF
    assert ops.relion_image.shape == (2, SIZE * HALF)


