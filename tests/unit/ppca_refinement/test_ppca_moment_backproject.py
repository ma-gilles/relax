"""Fused CUDA PPCA residuals and moment backprojection against the XLA statistics and recovar's adjoint."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from relax.helpers.adjoint import batch_adjoint_slice_volume_maybe_windowed
from relax.helpers.fourier_window import make_fourier_window_spec
from relax.ppca_refinement.residual_statistics import residual_statistics_from_moment_images

IMAGE_SHAPE, VOLUME_SHAPE = (32, 32), (32, 32, 32)
N_HALF, HALF = 32 * 17, 32 * 32 * 17


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


def _reference(lhs, rhs_parts, projections, indices, rotations, max_r, metric_trace):
    """XLA residual statistics and recovar's windowed adjoint, in the dtype of the operands."""
    F = indices.size
    rhs = jax.lax.complex(rhs_parts[..., :F], rhs_parts[..., F:])
    residual, correction = residual_statistics_from_moment_images(rhs, lhs, projections[:, :, indices])
    if metric_trace:
        P = rhs.shape[0]
        i, j = np.triu_indices(P)
        lhs = jnp.sum(lhs[np.flatnonzero(i == j)], axis=0, keepdims=True)

    def adjoint(images):
        return batch_adjoint_slice_volume_maybe_windowed(
            images,
            indices,
            rotations,
            jnp.zeros((images.shape[0], HALF), images.dtype),
            IMAGE_SHAPE,
            VOLUME_SHAPE,
            "linear_interp",
            True,
            True,
            use_window=True,
            max_r=max_r,
        )

    return np.asarray(adjoint(lhs)), np.asarray(adjoint(residual)), np.asarray(correction)


def _relative_l2(actual, truth):
    return float(np.linalg.norm(np.asarray(actual) - truth) / np.linalg.norm(truth))


@pytest.mark.gpu
@pytest.mark.parametrize("P,metric_trace", [(5, False), (11, False), (5, True), (11, True)])
def test_moment_scatter_matches_xla_statistics_and_recovar_adjoint(
    P, metric_trace, custom_cuda_lib, gpu_device, monkeypatch
):
    """Each output against the float64 reference; error at most twice the float32 reference's own.

    The float32 reference is the XLA residual statistics and recovar's channel-major atomic
    adjoint; on one 512-rotation HP4 block the backprojection errors of the two measured
    3.04e-6 and 3.05e-6 against float64. Metric channels (all packed, or their trace), the
    residual and the noise-power correction are checked; padding lanes stay zero.
    """
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda.kernels import ppca_moment_scatter_f32

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    spec = make_fourier_window_spec(IMAGE_SHAPE, 28, N_HALF, square=False, include_recon_window=True)
    rng = np.random.default_rng(5)
    R, K = 48, P * (P + 1) // 2
    rotations = _rotations(rng, R)
    indices = np.asarray(spec.recon_indices, np.int32)
    F = indices.size
    lhs = rng.standard_normal((K, R, F)).astype(np.float32)
    rhs_parts = rng.standard_normal((P, R, 2 * F)).astype(np.float32)
    projections = (rng.standard_normal((P, R, N_HALF)) + 1j * rng.standard_normal((P, R, N_HALF))).astype(np.complex64)
    lhs[:, 3] = 0.0  # one rotation of all-zero metric images still scatters its residual
    with jax.default_device(gpu_device):
        moments, correction = ppca_moment_scatter_f32(
            jnp.asarray(lhs),
            jnp.asarray(rhs_parts),
            jnp.asarray(projections),
            jnp.asarray(indices),
            jnp.asarray(rotations),
            image_shape=IMAGE_SHAPE,
            volume_shape=VOLUME_SHAPE,
            max_r=spec.max_r,
            metric_trace=metric_trace,
        )
        moments, correction = np.asarray(moments), np.asarray(correction)
        reference32 = _reference(
            jnp.asarray(lhs),
            jnp.asarray(rhs_parts),
            jnp.asarray(projections),
            indices,
            rotations,
            spec.max_r,
            metric_trace,
        )
        with jax.enable_x64(True):
            reference64 = _reference(
                jnp.asarray(lhs, jnp.float64),
                jnp.asarray(rhs_parts, jnp.float64),
                jnp.asarray(projections, jnp.complex128),
                indices,
                rotations.astype(np.float64),
                spec.max_r,
                metric_trace,
            )
    metric = 1 if metric_trace else K
    groups = -(-(metric + 2 * P) // 32)
    assert moments.shape == (groups, HALF, 32) and correction.shape == (F,)
    flat = np.transpose(moments, (1, 0, 2)).reshape(HALF, groups * 32)
    pairs = flat[:, metric : metric + 2 * P].reshape(HALF, P, 2)
    got = (flat[:, :metric].T, (pairs[..., 0] + 1j * pairs[..., 1]).T, correction)
    assert np.all(flat[:, metric + 2 * P :] == 0.0)
    for value, ref32, ref64 in zip(got, reference32, reference64, strict=True):
        reference_error = _relative_l2(ref32, ref64)
        assert 0 < reference_error < 1e-5
        assert _relative_l2(value, ref64) <= 2 * reference_error
