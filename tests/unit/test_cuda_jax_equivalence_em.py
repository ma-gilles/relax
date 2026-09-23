"""CUDA indexed projection (EM library) against the full pipeline projection.

Split from recovar tests/unit/test_cuda_jax_equivalence.py (relax split): ``project_indexed`` lives in the
EM CUDA library.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

pytestmark = [pytest.mark.unit, pytest.mark.gpu]


@pytest.fixture(autouse=True)
def _use_custom_cuda_lib(monkeypatch, custom_cuda_lib):
    import recovar.cuda_backproject as cuda_backproject

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)


# ── Helpers ──────────────────────────────────────────────────────────


def _skip_if_no_cuda():
    from recovar.cuda_backproject import cuda_available

    if not cuda_available():
        pytest.skip("CUDA backproject not available")


def _random_rotations(n, rng):
    """Generate n proper rotation matrices via QR decomposition."""
    z = rng.standard_normal((n, 3, 3))
    q, r = np.linalg.qr(z)
    d = np.sign(np.diagonal(r, axis1=1, axis2=2))
    q = q * d[:, None, :]
    det = np.linalg.det(q)
    q[det < 0] *= -1
    return q.astype(np.float32)


# ── Parametrization ──────────────────────────────────────────────────

_ORDERS = [0, 1]
_HALF_COMBOS = [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
]

# All (order, half_vol, half_img) combos
_ALL_COMBOS = [(order, hv, hi) for order in _ORDERS for hv, hi in _HALF_COMBOS]


@pytest.mark.parametrize("order,half_vol,half_img", _ALL_COMBOS)
@pytest.mark.parametrize("N", [16, 32])
def test_project_indexed_matches_full_project_gather(order, half_vol, half_img, N, gpu_device):
    """Indexed CUDA project must match full CUDA project gathered at the same pixels."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import project
    from relax.cuda.kernels import project_indexed
    import recovar.core.fourier_transform_utils as ftu

    rng = np.random.default_rng(2026)
    n_images = 7
    image_shape = (N, N)
    volume_shape = (N, N, N)
    n_pixels = N * (N // 2 + 1) if half_img else N * N
    # Include boundary and interior pixels, intentionally unsorted, to verify
    # compact output order follows the caller's index order.
    pixel_indices = np.array(
        [0, n_pixels - 1, n_pixels // 2, 3, min(17, n_pixels - 1), max(0, n_pixels - 5)],
        dtype=np.int32,
    )

    rots = jnp.array(_random_rotations(n_images, rng))
    if half_vol:
        vol_real = jnp.array(rng.standard_normal(volume_shape).astype(np.float32))
        vol_full = ftu.get_dft3(vol_real).ravel()
        vol = ftu.full_volume_to_half_volume(vol_full.reshape(volume_shape), volume_shape).ravel()
    else:
        vol = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))

    with jax.default_device(gpu_device):
        vol_gpu = jax.device_put(vol)
        rots_gpu = jax.device_put(rots)
        idx_gpu = jax.device_put(jnp.asarray(pixel_indices, dtype=jnp.int32))
        full = project(
            vol_gpu,
            rots_gpu,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_vol,
            half_image=half_img,
            max_r=None,
        )
        indexed = project_indexed(
            vol_gpu,
            idx_gpu,
            rots_gpu,
            image_shape,
            volume_shape,
            order=order,
            half_volume=half_vol,
            half_image=half_img,
            max_r=None,
        )

    np.testing.assert_allclose(
        np.asarray(indexed),
        np.asarray(full)[:, pixel_indices],
        atol=1e-5,
        rtol=1e-5,
        err_msg=f"project_indexed != gathered project for order={order}, half_vol={half_vol}, half_img={half_img}",
    )


@pytest.mark.parametrize("half_img", [False, True])
def test_project_indexed_matches_full_project_gather_with_max_r(half_img, gpu_device):
    """Indexed projection must preserve the full-projection radius clipping contract."""
    _skip_if_no_cuda()
    from recovar.cuda_backproject import project
    from relax.cuda.kernels import project_indexed

    rng = np.random.default_rng(2027)
    N = 32
    n_images = 4
    image_shape = (N, N)
    volume_shape = (N, N, N)
    n_pixels = N * (N // 2 + 1) if half_img else N * N
    pixel_indices = np.arange(0, n_pixels, max(1, n_pixels // 13), dtype=np.int32)
    rots = jnp.array(_random_rotations(n_images, rng))
    vol = jnp.array((rng.standard_normal(N**3) + 1j * rng.standard_normal(N**3)).astype(np.complex64))

    with jax.default_device(gpu_device):
        vol_gpu = jax.device_put(vol)
        rots_gpu = jax.device_put(rots)
        idx_gpu = jax.device_put(jnp.asarray(pixel_indices, dtype=jnp.int32))
        full = project(
            vol_gpu,
            rots_gpu,
            image_shape,
            volume_shape,
            order=1,
            half_volume=False,
            half_image=half_img,
            max_r=8.0,
        )
        indexed = project_indexed(
            vol_gpu,
            idx_gpu,
            rots_gpu,
            image_shape,
            volume_shape,
            order=1,
            half_volume=False,
            half_image=half_img,
            max_r=8.0,
        )

    np.testing.assert_allclose(np.asarray(indexed), np.asarray(full)[:, pixel_indices], atol=1e-5, rtol=1e-5)
