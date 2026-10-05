"""Focused tests for RELION's accelerated real-space preprocessing FFI."""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

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


def _zero_fill_shift(images, factors, shifts):
    normalized = images.astype(np.float32) * factors[:, None, None]
    out = np.zeros_like(normalized)
    height, width = images.shape[-2:]
    for row, (dx, dy) in enumerate(shifts.tolist()):
        src_x0 = max(0, -dx)
        src_x1 = width - max(0, dx)
        src_y0 = max(0, -dy)
        src_y1 = height - max(0, dy)
        if src_x0 >= src_x1 or src_y0 >= src_y1:
            continue
        dst_x0 = max(0, dx)
        dst_y0 = max(0, dy)
        out[row, dst_y0 : dst_y0 + src_y1 - src_y0, dst_x0 : dst_x0 + src_x1 - src_x0] = normalized[
            row, src_y0:src_y1, src_x0:src_x1
        ]
    return out


def test_relion_cuda_normalize_shift_is_bit_exact(gpu_device):
    from relax.cuda.kernels import relion_preprocess_real_f32

    rng = np.random.default_rng(20260714)
    images = rng.standard_normal((3, 32, 32)).astype(np.float32)
    factors = np.asarray([0.9988004, 1.0, 1.03125], dtype=np.float32)
    shifts = np.asarray([[-1, -1], [0, 0], [3, -2]], dtype=np.int32)
    expected = _zero_fill_shift(images, factors, shifts)

    with jax.default_device(gpu_device):
        normalized_shifted, unmasked = relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.asarray(factors),
            jnp.asarray(shifts),
            radius=10.0,
            cosine_width=3.0,
            apply_mask=False,
        )

    assert_matches(np.asarray(normalized_shifted), expected)
    assert_matches(np.asarray(unmasked), expected)


def test_relion_cuda_softmask_has_constant_exterior_and_finite_output(gpu_device):
    from relax.cuda.kernels import relion_preprocess_real_f32

    rng = np.random.default_rng(17)
    images = rng.standard_normal((2, 64, 64)).astype(np.float32)
    factors = np.asarray([0.97, 1.04], dtype=np.float32)
    shifts = np.asarray([[2, 1], [-3, 0]], dtype=np.int32)

    with jax.default_device(gpu_device):
        normalized_shifted, masked = relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.asarray(factors),
            jnp.asarray(shifts),
            radius=20.0,
            cosine_width=5.0,
            apply_mask=True,
        )

    normalized_shifted = np.asarray(normalized_shifted)
    masked = np.asarray(masked)
    assert_matches(normalized_shifted, _zero_fill_shift(images, factors, shifts))
    assert np.all(np.isfinite(masked))
    yy, xx = np.meshgrid(np.arange(64) - 32, np.arange(64) - 32, indexing="ij")
    exterior = np.sqrt(xx * xx + yy * yy) > 25.0
    for row in range(masked.shape[0]):
        assert_matches(masked[row][exterior], np.full(np.count_nonzero(exterior), masked[row, 0, 0]))


@pytest.mark.parametrize("native_lane_reduction", [False, True])
def test_relion_cuda_softmask_repeats_bit_exactly(gpu_device, native_lane_reduction):
    from relax.cuda.kernels import relion_preprocess_real_f32

    rng = np.random.default_rng(20260731)
    images = rng.standard_normal((3, 128, 128)).astype(np.float32)
    factors = np.asarray([0.9988004, 1.0, 1.03125], dtype=np.float32)
    shifts = np.asarray([[-1, -1], [0, 0], [3, -2]], dtype=np.int32)

    masked_repeats = []
    with jax.default_device(gpu_device):
        for _ in range(4):
            _normalized_shifted, masked = relion_preprocess_real_f32(
                jnp.asarray(images),
                jnp.asarray(factors),
                jnp.asarray(shifts),
                radius=23.529411,
                cosine_width=5.0,
                apply_mask=True,
                native_lane_reduction=native_lane_reduction,
            )
            masked_repeats.append(np.asarray(masked))

    for repeat in masked_repeats[1:]:
        assert_matches(repeat, masked_repeats[0])


@pytest.mark.parametrize("radius,cosine_width", [(1.0e-6, 1.0), (15.999, 1.0e-4)])
def test_relion_cuda_softmask_boundary_radii_remain_finite(radius, cosine_width, gpu_device):
    from relax.cuda.kernels import relion_preprocess_real_f32

    rng = np.random.default_rng(23)
    images = rng.standard_normal((2, 32, 32)).astype(np.float32)
    with jax.default_device(gpu_device):
        _normalized_shifted, masked = relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.ones((2,), dtype=jnp.float32),
            jnp.asarray([[1, -1], [-2, 2]], dtype=jnp.int32),
            radius=radius,
            cosine_width=cosine_width,
            apply_mask=True,
        )

    assert np.all(np.isfinite(np.asarray(masked)))


def test_relion_cuda_softmask_fails_closed_with_zero_background_area(gpu_device):
    from relax.cuda.kernels import relion_preprocess_real_f32

    with jax.default_device(gpu_device), pytest.raises(jax.errors.JaxRuntimeError, match="CUDA: invalid argument"):
        _normalized_shifted, masked = relion_preprocess_real_f32(
            jnp.ones((1, 8, 8), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.int32),
            radius=100.0,
            cosine_width=1.0,
            apply_mask=True,
        )
        masked.block_until_ready()


@pytest.mark.parametrize(
    "images,factors,shifts,error",
    [
        (np.zeros((1, 8, 8), dtype=np.float64), np.ones(1, dtype=np.float32), np.zeros((1, 2), dtype=np.int32), "images must be float32"),
        (np.zeros((1, 8, 8), dtype=np.float32), np.ones(1, dtype=np.float64), np.zeros((1, 2), dtype=np.int32), "normalization_factors must be float32"),
        (np.zeros((1, 8, 8), dtype=np.float32), np.ones(1, dtype=np.float32), np.zeros((1, 2), dtype=np.int64), "integer_shifts must be int32"),
    ],
)
def test_relion_cuda_preprocess_rejects_wrong_dtypes(images, factors, shifts, error, gpu_device):
    from relax.cuda.kernels import relion_preprocess_real_f32

    with jax.default_device(gpu_device), pytest.raises(TypeError, match=error):
        relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.asarray(factors),
            jnp.asarray(shifts),
            radius=2.0,
            cosine_width=1.0,
        )


def test_relion_cuda_preprocess_fails_closed_without_gpu(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="requires a JAX GPU backend"):
        em_cuda_kernels.relion_preprocess_real_f32.__wrapped__(
            jnp.zeros((1, 8, 8), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.int32),
            radius=2.0,
            cosine_width=1.0,
        )


def test_relion_cuda_preprocess_fails_closed_when_custom_cuda_disabled(monkeypatch):
    import recovar.cuda_backproject as cuda_backproject
    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setattr(cuda_backproject.jax, "default_backend", lambda: "gpu")
    monkeypatch.setenv("RECOVAR_DISABLE_CUDA", "1")
    with pytest.raises(RuntimeError, match="custom CUDA is disabled"):
        em_cuda_kernels.relion_preprocess_real_f32.__wrapped__(
            jnp.zeros((1, 8, 8), dtype=jnp.float32),
            jnp.ones((1,), dtype=jnp.float32),
            jnp.zeros((1, 2), dtype=jnp.int32),
            radius=2.0,
            cosine_width=1.0,
        )


@pytest.mark.parametrize("native_lane_reduction", [False, True])
def test_relion_cuda_softmask_batch_matches_single_image_bitwise(gpu_device, native_lane_reduction):
    """Batched soft-masking must equal the per-image result bit for bit.

    The launcher runs one background/fill launch for the whole batch and keeps
    the device-side per-image CUB sums, so an image's masked pixels may not
    depend on which other images share its batch.
    """

    from relax.cuda.kernels import relion_preprocess_real_f32

    rng = np.random.default_rng(20260913)
    images = rng.standard_normal((5, 64, 64)).astype(np.float32)
    factors = rng.uniform(0.9, 1.1, size=5).astype(np.float32)
    shifts = rng.integers(-3, 4, size=(5, 2)).astype(np.int32)

    with jax.default_device(gpu_device):
        batched = relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.asarray(factors),
            jnp.asarray(shifts),
            radius=20.0,
            cosine_width=5.0,
            apply_mask=True,
            native_lane_reduction=native_lane_reduction,
        )
        batched = tuple(np.asarray(value) for value in batched)
        for row in range(images.shape[0]):
            single = relion_preprocess_real_f32(
                jnp.asarray(images[row : row + 1]),
                jnp.asarray(factors[row : row + 1]),
                jnp.asarray(shifts[row : row + 1]),
                radius=20.0,
                cosine_width=5.0,
                apply_mask=True,
                native_lane_reduction=native_lane_reduction,
            )
            for batched_value, single_value in zip(batched, single):
                assert_matches(batched_value[row], np.asarray(single_value)[0])


def test_relion_cuda_non_finite_image_without_a_mask_is_queued_not_read(gpu_device):
    """Without a mask the call reads nothing back (it never did): the count is queued and raises at the drain."""

    from relax.cuda import kernels as em_cuda_kernels
    from relax.cuda.kernels import relion_preprocess_real_f32

    em_cuda_kernels.drain_relion_preprocess_checks()
    rng = np.random.default_rng(3)
    images = rng.standard_normal((3, 32, 32)).astype(np.float32)
    images[1, 16, 16] = np.nan
    with jax.default_device(gpu_device):
        relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.ones(3, dtype=jnp.float32),
            jnp.zeros((3, 2), dtype=jnp.int32),
            radius=10.0,
            cosine_width=3.0,
            apply_mask=False,
        )
        assert em_cuda_kernels.pending_relion_preprocess_checks() == 1
        with pytest.raises(RuntimeError, match=r"deferred check.*1 image\(s\) had a non-finite pixel"):
            em_cuda_kernels.drain_relion_preprocess_checks()
    assert em_cuda_kernels.pending_relion_preprocess_checks() == 0


@pytest.mark.parametrize("pixel", [(5, 7), (16, 16)], ids=["mask_exterior", "inside_mask"])
def test_relion_cuda_non_finite_image_fails_closed_and_is_named(gpu_device, pixel, apply_mask=True):
    """A non-finite pixel anywhere in an image aborts the call and names the image (relax#16).

    The soft-mask background sums read only the pixels outside the mask, so a NaN inside it used to pass.
    """

    from relax.cuda import kernels as em_cuda_kernels
    from relax.cuda.kernels import relion_preprocess_real_f32

    rng = np.random.default_rng(3)
    images = rng.standard_normal((3, 32, 32)).astype(np.float32)
    images[1][pixel] = np.nan

    def call():
        return relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.ones(3, dtype=jnp.float32),
            jnp.zeros((3, 2), dtype=jnp.int32),
            radius=10.0,
            cosine_width=3.0,
            apply_mask=apply_mask,
        )

    with jax.default_device(gpu_device):
        em_cuda_kernels.note_relion_preprocess_batch(None)
        with pytest.raises(RuntimeError, match=r"1 image\(s\) had a non-finite pixel.*first: position 1 of the batch"):
            call()
        # A batch noted by the caller names the dataset image; the note serves one call.
        em_cuda_kernels.note_relion_preprocess_batch(np.asarray([40, 41, 42]))
        with pytest.raises(RuntimeError, match=r"first: dataset image 41 \(position 1 of the batch\)"):
            call()
        with pytest.raises(RuntimeError, match="first: position 1 of the batch"):
            call()
        # A note of another size is not this call's batch.
        em_cuda_kernels.note_relion_preprocess_batch(np.asarray([40, 41]))
        with pytest.raises(RuntimeError, match="first: position 1 of the batch"):
            call()
        # Inside a compiled program the kernel's own check fails the call.
        with pytest.raises(jax.errors.JaxRuntimeError, match="CUDA: invalid argument"):
            jax.jit(lambda: call()[1])().block_until_ready()


def test_relion_cuda_deferred_check_names_an_image_with_a_pixel_inside_the_mask(gpu_device):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.cuda.kernels import relion_preprocess_real_f32

    rng = np.random.default_rng(5)
    images = rng.standard_normal((4, 32, 32)).astype(np.float32)
    images[2, 16, 16] = np.inf
    images[3, 15, 17] = np.nan
    with jax.default_device(gpu_device), pytest.raises(RuntimeError) as failure:
        with em_cuda_kernels.deferred_relion_preprocess_checks("test pass") as checks:
            for batch in range(2):
                checks.at(f"batch {batch}")
                em_cuda_kernels.note_relion_preprocess_batch(np.arange(4) + 10 * batch)
                relion_preprocess_real_f32(
                    jnp.asarray(images if batch else images[[0, 1, 0, 1]]),
                    jnp.ones(4, dtype=jnp.float32),
                    jnp.zeros((4, 2), dtype=jnp.int32),
                    radius=10.0,
                    cosine_width=3.0,
                    apply_mask=True,
                )
    message = str(failure.value)
    assert "deferred check, test pass" in message and "2 image(s) had a non-finite pixel" in message
    assert "batch 1: 2 image(s), first dataset image 12 (position 2 of the batch)" in message
    assert "batch 0" not in message


def test_relion_cuda_softmask_deferred_check_queues_and_fails_closed_on_drain(gpu_device):
    """With the deferred check, a non-finite image is reported at the drain, not per call.

    The finite rows must still equal their single-image results bit for bit
    and the invalid row carries NaN in its masked exterior until the drain.
    """

    from relax.cuda import kernels as em_cuda_kernels
    from relax.cuda.kernels import relion_preprocess_real_f32

    em_cuda_kernels.drain_relion_preprocess_checks()
    rng = np.random.default_rng(3)
    images = rng.standard_normal((3, 32, 32)).astype(np.float32)
    images[1, 5, 7] = np.nan
    factors = np.ones(3, dtype=np.float32)
    shifts = np.zeros((3, 2), dtype=np.int32)

    with jax.default_device(gpu_device):
        _normalized_shifted, masked = relion_preprocess_real_f32(
            jnp.asarray(images),
            jnp.asarray(factors),
            jnp.asarray(shifts),
            radius=10.0,
            cosine_width=3.0,
            apply_mask=True,
            deferred_finite_check=True,
        )
        masked = np.asarray(masked)
        assert em_cuda_kernels.pending_relion_preprocess_checks() == 1
        for row in (0, 2):
            _single_shifted, single = relion_preprocess_real_f32(
                jnp.asarray(images[row : row + 1]),
                jnp.asarray(factors[row : row + 1]),
                jnp.asarray(shifts[row : row + 1]),
                radius=10.0,
                cosine_width=3.0,
                apply_mask=True,
                deferred_finite_check=False,
            )
            assert_matches(masked[row], np.asarray(single)[0])

    yy, xx = np.meshgrid(np.arange(32) - 16, np.arange(32) - 16, indexing="ij")
    exterior = np.sqrt(xx * xx + yy * yy) > 13.0
    assert np.all(np.isnan(masked[1][exterior]))
    with pytest.raises(RuntimeError, match="deferred check.*1 image"):
        em_cuda_kernels.drain_relion_preprocess_checks()
    assert em_cuda_kernels.pending_relion_preprocess_checks() == 0


def test_relion_cuda_softmask_deferred_check_drains_clean_batches(gpu_device):
    from relax.cuda import kernels as em_cuda_kernels
    from relax.cuda.kernels import relion_preprocess_real_f32

    em_cuda_kernels.drain_relion_preprocess_checks()
    rng = np.random.default_rng(11)
    with jax.default_device(gpu_device):
        for _ in range(3):
            relion_preprocess_real_f32(
                jnp.asarray(rng.standard_normal((2, 32, 32)).astype(np.float32)),
                jnp.ones(2, dtype=jnp.float32),
                jnp.zeros((2, 2), dtype=jnp.int32),
                radius=10.0,
                cosine_width=3.0,
                apply_mask=True,
                deferred_finite_check=True,
            )
        # apply_mask=False queues its count too: a non-finite pixel is invalid without a mask.
        relion_preprocess_real_f32(
            jnp.asarray(rng.standard_normal((2, 32, 32)).astype(np.float32)),
            jnp.ones(2, dtype=jnp.float32),
            jnp.zeros((2, 2), dtype=jnp.int32),
            radius=10.0,
            cosine_width=3.0,
            apply_mask=False,
            deferred_finite_check=True,
        )
    assert em_cuda_kernels.pending_relion_preprocess_checks() == 4
    assert em_cuda_kernels.drain_relion_preprocess_checks() == 4
    assert em_cuda_kernels.drain_relion_preprocess_checks() == 0


def test_relion_preprocess_deferred_check_flag_is_strict(monkeypatch):
    from relax.cuda import kernels as em_cuda_kernels

    monkeypatch.setenv(em_cuda_kernels.RELION_PREPROCESS_DEFERRED_CHECK_ENV, "1")
    assert em_cuda_kernels.relion_preprocess_deferred_check_requested() is True
    monkeypatch.delenv(em_cuda_kernels.RELION_PREPROCESS_DEFERRED_CHECK_ENV)
    assert em_cuda_kernels.relion_preprocess_deferred_check_requested() is False
    monkeypatch.setenv(em_cuda_kernels.RELION_PREPROCESS_DEFERRED_CHECK_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        em_cuda_kernels.relion_preprocess_deferred_check_requested()
