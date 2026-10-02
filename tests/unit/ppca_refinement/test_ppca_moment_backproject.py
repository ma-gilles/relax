"""Voxel-major CUDA moment backprojection against recovar's windowed adjoint (float64 reference)."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from relax.helpers.adjoint import batch_adjoint_slice_volume_maybe_windowed
from relax.helpers.fourier_window import make_fourier_window_spec

IMAGE_SHAPE, VOLUME_SHAPE = (32, 32), (32, 32, 32)
HALF = 32 * 32 * 17


def _rotations(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    ).astype(np.float32)


def _recovar(images, indices, rotations, max_r, dtype):
    return batch_adjoint_slice_volume_maybe_windowed(
        images,
        indices,
        rotations,
        jnp.zeros((images.shape[0], HALF), dtype),
        IMAGE_SHAPE,
        VOLUME_SHAPE,
        "linear_interp",
        True,
        True,
        use_window=True,
        max_r=max_r,
    )


def _relative_l2(actual, truth):
    return float(np.linalg.norm(np.asarray(actual) - truth) / np.linalg.norm(truth))


@pytest.mark.gpu
@pytest.mark.parametrize("n_real,n_complex", [(15, 5), (66, 11), (1, 5)])
def test_moment_backprojection_matches_recovar_adjoint(n_real, n_complex, custom_cuda_lib, gpu_device, monkeypatch):
    """Every channel is recovar's adjoint up to float32 summation order; groups of 32 lanes hold all channels.

    Both float32 results are compared with recovar's float64 adjoint of the same images: the
    voxel-major kernel's relative L2 error may be at most twice recovar's own float32 error
    (on one 512-rotation HP4 block the two measured 3.04e-6 and 3.05e-6).
    """
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda.kernels import ppca_moment_backproject_f32

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    spec = make_fourier_window_spec(IMAGE_SHAPE, 28, 32 * 17, square=False, include_recon_window=True)
    rng = np.random.default_rng(5)
    rotations = _rotations(rng, 48)
    indices = np.asarray(spec.recon_indices, np.int32)
    F = indices.size
    real = rng.standard_normal((n_real, 48, F)).astype(np.float32)
    complex_ = (rng.standard_normal((n_complex, 48, F)) + 1j * rng.standard_normal((n_complex, 48, F))).astype(
        np.complex64
    )
    real[0, 3] = 0.0  # all-zero pixels of one channel and one whole rotation still scatter the others
    with jax.default_device(gpu_device):
        out = np.asarray(
            ppca_moment_backproject_f32(
                jnp.asarray(real),
                jnp.asarray(complex_),
                jnp.asarray(indices),
                jnp.asarray(rotations),
                image_shape=IMAGE_SHAPE,
                volume_shape=VOLUME_SHAPE,
                max_r=spec.max_r,
            )
        )
        reference32 = [np.asarray(_recovar(x, indices, rotations, spec.max_r, x.dtype)) for x in (real, complex_)]
        with jax.enable_x64(True):
            reference64 = [
                np.asarray(_recovar(x, indices, rotations.astype(np.float64), spec.max_r, dtype))
                for x, dtype in (
                    (real.astype(np.float64), jnp.float64),
                    (complex_.astype(np.complex128), jnp.complex128),
                )
            ]
    groups = -(-(n_real + 2 * n_complex) // 32)
    assert out.shape == (groups, HALF, 32)
    flat = np.transpose(out, (1, 0, 2)).reshape(HALF, groups * 32)
    got_real = flat[:, :n_real].T
    pairs = flat[:, n_real : n_real + 2 * n_complex].reshape(HALF, n_complex, 2)
    got_complex = (pairs[..., 0] + 1j * pairs[..., 1]).T
    assert np.all(flat[:, n_real + 2 * n_complex :] == 0.0)  # padding lanes stay zero
    for got, ref32, ref64 in zip((got_real, got_complex), reference32, reference64, strict=True):
        recovar_error = _relative_l2(ref32, ref64)
        assert 0 < recovar_error < 1e-5
        assert _relative_l2(got, ref64) <= 2 * recovar_error
