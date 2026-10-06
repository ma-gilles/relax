"""Fused CUDA PPCA residuals and moment backprojection against the XLA statistics and recovar's adjoint."""

import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp
from helpers.float_compare import assert_matches

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
        windowed = projections[:, :, indices]
        moments, correction = ppca_moment_scatter_f32(
            jnp.asarray(lhs),
            jnp.asarray(rhs_parts),
            jnp.asarray(np.concatenate([windowed.real, windowed.imag], axis=-1)),
            jnp.asarray(indices),
            jnp.asarray(rotations),
            image_shape=IMAGE_SHAPE,
            volume_shape=VOLUME_SHAPE,
            max_r=spec.max_r,
            metric_trace=metric_trace,
        )
        moments, correction = np.asarray(moments), np.asarray(correction)
        # A window padded with -1 (the GPU GEMM alignment) ignores whatever the operands hold there.
        junk = 5

        def pad(x, width):
            return np.concatenate([x[..., :width], rng.standard_normal(x.shape[:-1] + (junk,))], axis=-1)

        planar = np.concatenate([windowed.real, windowed.imag], axis=-1)
        padded_moments, padded_correction = ppca_moment_scatter_f32(
            jnp.asarray(pad(lhs, F), jnp.float32),
            jnp.asarray(np.concatenate([pad(rhs_parts[..., :F], F), pad(rhs_parts[..., F:], F)], -1), jnp.float32),
            jnp.asarray(np.concatenate([pad(planar[..., :F], F), pad(planar[..., F:], F)], -1), jnp.float32),
            jnp.asarray(np.concatenate([indices, np.full(junk, -1, np.int32)])),
            jnp.asarray(rotations),
            image_shape=IMAGE_SHAPE,
            volume_shape=VOLUME_SHAPE,
            max_r=spec.max_r,
            metric_trace=metric_trace,
        )
        assert np.all(np.asarray(padded_correction)[F:] == 0)
        assert_matches(np.asarray(padded_correction)[:F], correction)
        assert_matches(np.asarray(padded_moments), moments)
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


def _trilinear_scatter(values, indices, rotations, max_r):
    """float64 numpy scatter of per-(rotation, pixel) ``values`` ``(R, F)`` into the half volume, with the
    linear-interpolation weights ``w`` and with ``w**2``: each windowed half-image pixel and, off the
    self-conjugate columns, its Hermitian partner lands on the 8 trilinear targets of its rotated frequency on
    the centred full grid, and targets with kz < 0 fold onto their partner. Written from the definition, not
    from the kernel's target tables (no doubled interior targets)."""
    H, W = IMAGE_SHAPE[0], IMAGE_SHAPE[1] // 2 + 1
    N0, N1, N2 = VOLUME_SHAPE
    c = np.array([N0 // 2, N1 // 2, N2 // 2], np.float64)
    k0 = (indices // W - H // 2).astype(np.float64)
    k1 = (indices % W).astype(np.float64)
    k1 = np.where(indices % W * 2 == IMAGE_SHAPE[1], -k1, k1)
    keep = k0**2 + k1**2 <= max_r**2
    interior = (indices % W > 0) & (indices % W * 2 != IMAGE_SHAPE[1])
    top = indices // W == 0
    points = [(k0, k1, keep), (np.where(top, k0, -k0), -k1, keep & interior)]
    out = {1: np.zeros(N0 * N1 * (N2 // 2 + 1)), 2: np.zeros(N0 * N1 * (N2 // 2 + 1))}
    for r, rotation in enumerate(np.asarray(rotations, np.float64)):
        for p0, p1, mask in points:
            g = np.stack([p1, p0], -1) @ rotation[:2] + c  # column frequency on row 0 (recovar rot_to_compact)
            base = np.floor(g)
            frac = g - base
            for corner in np.ndindex(2, 2, 2):
                weight = np.prod(np.where(np.array(corner), frac, 1 - frac), axis=-1)
                j = (base + np.array(corner)).astype(np.int64)
                kz = j[:, 2] - N2 // 2
                fold = kz < 0
                j0 = np.where(fold, (N0 - j[:, 0]) % N0, j[:, 0])
                j1 = np.where(fold, (N1 - j[:, 1]) % N1, j[:, 1])
                target = (j0 * N1 + j1) * (N2 // 2 + 1) + np.abs(kz)
                for power in (1, 2):
                    np.add.at(out[power], target[mask], (weight**power * values[r])[mask])
    return out[1], out[2]


@pytest.mark.gpu
@pytest.mark.parametrize("metric_trace", [False, True])
def test_exact_trace_channel_scatters_squared_weights(metric_trace, custom_cuda_lib, gpu_device, monkeypatch):
    """The exact-trace channel is ``sum w**2 tr L`` from a float64 numpy scatter written from the definition
    (whose ``w`` channel matches recovar's adjoint, which validates its geometry); the other channels equal
    those of the same call without it."""
    import recovar.cuda_backproject as cuda_backproject

    from relax.cuda.kernels import ppca_moment_scatter_f32

    monkeypatch.setenv("RECOVAR_CUDA_LIB", str(custom_cuda_lib))
    monkeypatch.delenv("RECOVAR_DISABLE_CUDA", raising=False)
    monkeypatch.setattr(cuda_backproject, "_cuda_ok", None)
    spec = make_fourier_window_spec(IMAGE_SHAPE, 28, N_HALF, square=False, include_recon_window=True)
    rng = np.random.default_rng(7)
    P, R = 3, 24
    K = P * (P + 1) // 2
    rotations = _rotations(rng, R)
    indices = np.asarray(spec.recon_indices, np.int32)
    F = indices.size
    lhs = rng.standard_normal((K, R, F)).astype(np.float32)
    rhs_parts = rng.standard_normal((P, R, 2 * F)).astype(np.float32)
    planar = rng.standard_normal((P, R, 2 * F)).astype(np.float32)
    i, j = np.triu_indices(P)
    trace = lhs[np.flatnonzero(i == j)].sum(axis=0).astype(np.float64)
    with jax.default_device(gpu_device):

        def scatter(exact):
            return [
                np.asarray(x)
                for x in ppca_moment_scatter_f32(
                    jnp.asarray(lhs),
                    jnp.asarray(rhs_parts),
                    jnp.asarray(planar),
                    jnp.asarray(indices),
                    jnp.asarray(rotations),
                    image_shape=IMAGE_SHAPE,
                    volume_shape=VOLUME_SHAPE,
                    max_r=spec.max_r,
                    metric_trace=metric_trace,
                    exact_trace=exact,
                )
            ]

        (plain, plain_correction), (moments, correction) = scatter(False), scatter(True)
        recovar_trace = np.asarray(
            batch_adjoint_slice_volume_maybe_windowed(
                jnp.asarray(trace[None], jnp.float32),
                indices,
                rotations,
                jnp.zeros((1, HALF), jnp.float32),
                IMAGE_SHAPE,
                VOLUME_SHAPE,
                "linear_interp",
                True,
                True,
                use_window=True,
                max_r=spec.max_r,
            )
        )[0]
    linear, squared = _trilinear_scatter(trace, indices, rotations, spec.max_r)
    assert _relative_l2(recovar_trace, linear) < 1e-5  # the numpy geometry is recovar's adjoint
    metric = 1 if metric_trace else K
    flat = np.transpose(moments, (1, 0, 2)).reshape(HALF, -1)
    flat_plain = np.transpose(plain, (1, 0, 2)).reshape(HALF, -1)
    assert flat.shape[1] == 32 * -(-(metric + 1 + 2 * P) // 32)
    assert _relative_l2(flat[:, metric], squared) < 1e-5
    # The squared weights give another volume (no rotation here is grid-aligned).
    assert _relative_l2(squared, linear) > 0.1
    assert_matches(flat[:, :metric], flat_plain[:, :metric])
    assert_matches(flat[:, metric + 1 : metric + 1 + 2 * P], flat_plain[:, metric : metric + 2 * P])
    assert np.all(flat[:, metric + 1 + 2 * P :] == 0.0)
    assert_matches(correction, plain_correction)
