"""Reduction-order tests for RELION CUDA-style fine Gaussian scores."""


import numpy as np
import pytest
from helpers.float_compare import assert_matches, matches

pytest.importorskip("jax")
import jax.numpy as jnp
from helpers.sparse_pass2_test_support import (
    _relion_cuda_fine_tree_sum,
)

from relax.helpers.fourier_window import make_fourier_window_spec
from relax.local.local_big_jit import _validate_relion_exact_fine_diff2_preconditions
from relax.local.local_bucket_stages import _relion_exact_fine_full_to_compact_lookup
from relax.sparse_pass2.sparse_pass2_scoring import _RELION_CUDA_FINE_REF3D_BLOCK_SIZE, _RELION_CUDA_POWERCLASS_BLOCK_SIZE, _relion_cuda_fine_diff2_sum, _relion_cuda_fine_diff2_to_scores, _relion_cuda_fine_full_to_compact_lookup, _relion_cuda_fine_pixel_weights, _relion_cuda_powerclass_highres_norm_units, _relion_cuda_powerclass_highres_xi2_half, _score_pass2_bucket_relion_gpu_diff2, _score_pass2_bucket_relion_gpu_diff2_raw

pytestmark = pytest.mark.unit


def _numpy_cuda_fine_tree(values):
    values = np.asarray(values, dtype=np.float32)
    lanes = np.zeros(values.shape[:-1] + (_RELION_CUDA_FINE_REF3D_BLOCK_SIZE,), dtype=np.float32)
    for pixel in range(values.shape[-1]):
        lane = pixel % _RELION_CUDA_FINE_REF3D_BLOCK_SIZE
        lanes[..., lane] = np.float32(lanes[..., lane] + values[..., pixel])
    width = _RELION_CUDA_FINE_REF3D_BLOCK_SIZE // 2
    while width:
        lanes[..., :width] = np.float32(lanes[..., :width] + lanes[..., width : 2 * width])
        width //= 2
    return lanes[..., 0]


def _numpy_cuda_fine_diff2(reference, shifted, weight):
    reference = np.asarray(reference)
    shifted = np.asarray(shifted)
    weight = np.asarray(weight, dtype=np.float32)
    diff_real = np.float32(reference.real - shifted.real)
    diff_imag = np.float32(reference.imag - shifted.imag)
    square_sum = np.float32(
        np.float32(diff_real * diff_real) + np.float32(diff_imag * diff_imag)
    )
    terms = np.float32(np.float32(square_sum * np.float32(0.5)) * weight)
    return _numpy_cuda_fine_tree(terms)


def _numpy_cuda_powerclass_highres_half(centered_image, current_size):
    centered_image = np.asarray(centered_image, dtype=np.complex64)
    batch, height, half_width = centered_image.shape
    relion_image = np.roll(centered_image, -(height // 2), axis=1)
    relion_image = np.complex64(relion_image / np.float32(height * height))
    lanes = np.zeros(
        (batch, (height * half_width + _RELION_CUDA_POWERCLASS_BLOCK_SIZE - 1)
         // _RELION_CUDA_POWERCLASS_BLOCK_SIZE, _RELION_CUDA_POWERCLASS_BLOCK_SIZE),
        dtype=np.float32,
    )
    for voxel in range(height * half_width):
        x = voxel % half_width
        row = voxel // half_width
        y = row if row < half_width else row - height
        shell = int(np.rint(np.sqrt(np.float32(x * x + y * y))))
        if shell <= 0 or shell >= half_width or (x == 0 and y < 0):
            continue
        if shell >= current_size // 2 + 1:
            value = relion_image.reshape(batch, -1)[:, voxel]
            power = np.float32(
                np.float32(value.real * value.real)
                + np.float32(value.imag * value.imag)
            )
            lanes[:, voxel // _RELION_CUDA_POWERCLASS_BLOCK_SIZE, voxel % _RELION_CUDA_POWERCLASS_BLOCK_SIZE] = power
    width = _RELION_CUDA_POWERCLASS_BLOCK_SIZE // 2
    while width:
        lanes[..., :width] = np.float32(lanes[..., :width] + lanes[..., width : 2 * width])
        width //= 2
    total = np.zeros((batch,), dtype=np.float32)
    for block_sum in lanes[..., 0].T:
        total = np.float32(total + block_sum)
    return np.float32(total * np.float32(0.5))


def _numpy_cuda_powerclass_highres_half_double(centered_image, current_size):
    centered_image = np.asarray(centered_image, dtype=np.complex128)
    batch, height, half_width = centered_image.shape
    relion_image = np.roll(centered_image, -(height // 2), axis=1)
    relion_image = np.complex128(relion_image / np.float64(height * height))
    lanes = np.zeros(
        (
            batch,
            (height * half_width + _RELION_CUDA_POWERCLASS_BLOCK_SIZE - 1)
            // _RELION_CUDA_POWERCLASS_BLOCK_SIZE,
            _RELION_CUDA_POWERCLASS_BLOCK_SIZE,
        ),
        dtype=np.float64,
    )
    for voxel in range(height * half_width):
        x = voxel % half_width
        row = voxel // half_width
        y = row if row < half_width else row - height
        shell = int(np.rint(np.sqrt(np.float64(x * x + y * y))))
        if shell <= 0 or shell >= half_width or (x == 0 and y < 0):
            continue
        if shell >= current_size // 2 + 1:
            value = relion_image.reshape(batch, -1)[:, voxel]
            power = np.float64(
                np.float64(value.real * value.real)
                + np.float64(value.imag * value.imag)
            )
            lanes[
                :,
                voxel // _RELION_CUDA_POWERCLASS_BLOCK_SIZE,
                voxel % _RELION_CUDA_POWERCLASS_BLOCK_SIZE,
            ] = power
    width = _RELION_CUDA_POWERCLASS_BLOCK_SIZE // 2
    while width:
        lanes[..., :width] = np.float64(
            lanes[..., :width] + lanes[..., width : 2 * width]
        )
        width //= 2
    total = np.zeros((batch,), dtype=np.float64)
    for block_sum in lanes[..., 0].T:
        total = np.float64(total + block_sum)
    return np.float64(total * np.float64(0.5))


def test_relion_cuda_fine_tree_matches_256_lane_pass_and_tree():
    # Alternating scales make sequential per-lane accumulation distinguishable
    # from a flat reduction while retaining deterministic float32 operands.
    index = np.arange(773, dtype=np.float32)
    values = np.stack(
        [
            np.where((index.astype(np.int32) & 1) == 0, index * 1.0e4, -index * 1.0e-3),
            np.sin(index).astype(np.float32) * np.float32(1.0e3),
        ]
    ).astype(np.float32)

    expected = _numpy_cuda_fine_tree(values)
    actual = np.asarray(_relion_cuda_fine_tree_sum(jnp.asarray(values)))

    assert_matches(actual, expected)
    assert actual.dtype == np.float32


def test_relion_cuda_fine_diff2_preserves_acc_double_precision_operands():
    rng = np.random.default_rng(365)
    reference64 = (
        rng.normal(size=(2, 259)) + 1j * rng.normal(size=(2, 259))
    ).astype(np.complex64)
    shifted64 = (
        rng.normal(size=(3, 259)) + 1j * rng.normal(size=(3, 259))
    ).astype(np.complex64)
    weight = rng.uniform(size=(1, 259)).astype(np.float32)

    expected = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference64[:, None, :]),
            jnp.asarray(shifted64[None, :, :]),
            jnp.asarray(weight),
        )
    )
    actual = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference64.astype(np.complex128)[:, None, :]),
            jnp.asarray(shifted64.astype(np.complex128)[None, :, :]),
            jnp.asarray(weight, dtype=jnp.float64),
        )
    )

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected.astype(np.float64), rtol=2e-7, atol=1e-7)


def test_relion_cuda_fine_pixel_weight_preserves_acc_double_precision():
    corr = np.asarray([75351.31086994553], dtype=np.float64)
    half_weight = np.asarray([53814.33132654639], dtype=np.float64)
    expected = corr * half_weight
    narrowed = np.float32(corr) * np.float32(half_weight)

    actual = np.asarray(
        _relion_cuda_fine_pixel_weights(jnp.asarray(corr), jnp.asarray(half_weight))
    )

    assert_matches(actual, expected)
    assert actual.dtype == np.float64
    assert not matches(actual, narrowed.astype(np.float64))


def test_relion_cuda_powerclass_highres_matches_128_lane_block_trees():
    rng = np.random.default_rng(2297)
    height = 32
    centered = (
        rng.normal(size=(2, height, height // 2 + 1))
        + 1j * rng.normal(size=(2, height, height // 2 + 1))
    ).astype(np.complex64) * np.float32(height * height)
    current_size = 14
    expected = _numpy_cuda_powerclass_highres_half(centered, current_size)
    actual = np.asarray(
        _relion_cuda_powerclass_highres_xi2_half(
            jnp.asarray(centered.reshape(2, -1)),
            image_shape=(height, height),
            current_size=current_size,
        )
    )

    assert_matches(actual, expected)
    assert actual.dtype == np.float32


def test_relion_cuda_powerclass_highres_preserves_acc_double_precision():
    rng = np.random.default_rng(2298)
    height = 32
    centered = (
        rng.normal(size=(2, height, height // 2 + 1))
        + 1j * rng.normal(size=(2, height, height // 2 + 1))
    ).astype(np.complex128) * np.float64(height * height)
    centered += np.complex128(2.0**-35 + 1j * 2.0**-36)
    current_size = 14
    expected = _numpy_cuda_powerclass_highres_half_double(centered, current_size)
    actual = np.asarray(
        _relion_cuda_powerclass_highres_xi2_half(
            jnp.asarray(centered.reshape(2, -1)),
            image_shape=(height, height),
            current_size=current_size,
        )
    )

    assert_matches(actual, expected)
    assert actual.dtype == np.float64
    assert not matches(actual, expected.astype(np.float32).astype(np.float64))


def test_relion_cuda_powerclass_norm_units_match_randomized_reference():
    rng = np.random.default_rng(4021)
    height = 32
    centered = (
        rng.normal(size=(2, height, height // 2 + 1))
        + 1j * rng.normal(size=(2, height, height // 2 + 1))
    ).astype(np.complex64) * np.float32(height * height)
    current_size = 14
    expected_half = _numpy_cuda_powerclass_highres_half(centered, current_size)
    expected = np.float32(
        np.float32(expected_half * np.float32(2.0))
        * np.float32((height * height) ** 2)
    )

    actual = np.asarray(
        _relion_cuda_powerclass_highres_norm_units(
            jnp.asarray(centered.reshape(2, -1)),
            image_shape=(height, height),
            current_size=current_size,
        )
    )
    assert_matches(actual, expected)
    assert actual.dtype == np.float32


@pytest.mark.parametrize("height", [30, 32])
def test_relion_cuda_powerclass_norm_units_preserve_divide_before_square(height):
    # One high-shell pixel removes every reduction-order ambiguity. At 30x30,
    # rounding 1/900 before squaring/rescaling gives the next float32 above 1.
    # At 32x32, division by 1024 is exact: both orders legitimately give 1.
    centered = np.zeros((1, height, height // 2 + 1), dtype=np.complex64)
    centered[0, height // 2, 9] = 1.0  # ky=0, kx=9; above current_size/2=7.
    expected = np.float32(1.0)
    if height == 30:
        expected = np.nextafter(expected, np.float32(2.0))

    actual = np.asarray(
        _relion_cuda_powerclass_highres_norm_units(
            jnp.asarray(centered.reshape(1, -1)),
            image_shape=(height, height),
            current_size=14,
        )
    )

    assert_matches(actual, np.asarray([expected], dtype=np.float32))
    assert actual.dtype == np.float32


def test_relion_cuda_fine_conversion_uses_common_min_and_source_operation_order():
    # Captured case-20 particle 469 values: RELION's candidates differ by one
    # ULP in positive diff2. The large common min must be inserted at the same
    # point as cuda_kernel_exponentiate_weights_fine.
    diff2 = np.asarray([[1214.7265625, 1214.7264404296875]], dtype=np.float32)
    rotation_prior = np.asarray([[-7.326465606689453, -7.326465606689453]], dtype=np.float64)
    translation_prior = np.asarray([[-7.363075256347656, -7.363075256347656]], dtype=np.float64)
    common_min = np.asarray([1214.0980224609375], dtype=np.float32)
    mask = np.ones_like(diff2, dtype=bool)

    actual = np.asarray(
        _relion_cuda_fine_diff2_to_scores(
            jnp.asarray(diff2),
            jnp.asarray(rotation_prior),
            jnp.asarray(translation_prior),
            jnp.asarray(mask),
            min_diff2=jnp.asarray(common_min),
        )
    )
    expected = np.float32(np.float32(np.float32(rotation_prior) + np.float32(translation_prior)) + common_min[:, None])
    expected = np.float32(expected - diff2)
    naive = np.float32(np.float32(-diff2 + np.float32(rotation_prior)) + np.float32(translation_prior))

    assert_matches(actual, expected)
    assert actual.dtype == np.float32
    assert actual[0, 1] > actual[0, 0]
    assert not matches(actual, naive)


def test_relion_cuda_fine_conversion_rejects_diff2_below_external_minimum():
    diff2 = jnp.asarray([[100.0, 100.00001, 100.00002]], dtype=jnp.float32)
    mask = jnp.ones_like(diff2, dtype=bool)
    scores = np.asarray(
        _relion_cuda_fine_diff2_to_scores(
            diff2,
            jnp.zeros_like(diff2),
            jnp.zeros_like(diff2),
            mask,
            min_diff2=jnp.asarray([100.00001], dtype=jnp.float32),
        )
    )

    assert np.isneginf(scores[0, 0])
    external_min = np.float32(100.00001)
    assert_matches(
        scores[0, 1:],
        np.float32(external_min - np.asarray(diff2, dtype=np.float32)[0, 1:]),
    )


def test_relion_cuda_fine_raw_scorer_hlo_avoids_hypothesis_pixel_temporary():
    batch, translations, rotations, pixels = 2, 7, 5, 517
    args = (
        jnp.zeros((batch, translations, pixels), dtype=jnp.complex64),
        jnp.ones((batch, pixels), dtype=jnp.float32),
        jnp.zeros((batch, rotations, pixels), dtype=jnp.complex64),
        jnp.ones((pixels,), dtype=jnp.float32),
    )
    lowered = _score_pass2_bucket_relion_gpu_diff2_raw.lower(*args)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))

    assert "stablehlo.dot_general" not in stablehlo
    assert stablehlo.count("stablehlo.while") == 1

    memory = lowered.compile().memory_analysis()
    full_hypothesis_pixel_bytes = batch * rotations * translations * pixels * 4
    assert memory.temp_size_in_bytes < full_hypothesis_pixel_bytes


def test_relion_cuda_fine_diff2_matches_reference_without_pixel_tensor():
    rng = np.random.default_rng(18)
    reference = (
        rng.normal(size=(2, 3, 1, 517)) + 1j * rng.normal(size=(2, 3, 1, 517))
    ).astype(np.complex64)
    shifted = (
        rng.normal(size=(2, 1, 4, 517)) + 1j * rng.normal(size=(2, 1, 4, 517))
    ).astype(np.complex64)
    weight = rng.uniform(0.0, 3.0, size=(2, 1, 1, 517)).astype(np.float32)

    expected = np.empty((2, 3, 4), dtype=np.float32)
    for batch in range(2):
        for rotation in range(3):
            for translation in range(4):
                expected[batch, rotation, translation] = _numpy_cuda_fine_diff2(
                    reference[batch, rotation, 0],
                    shifted[batch, 0, translation],
                    weight[batch, 0, 0],
                )
    actual = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference), jnp.asarray(shifted), jnp.asarray(weight)
        )
    )

    # XLA may contract the pointwise square/add/multiply into FMAs whereas
    # NumPy evaluates each operation separately. The lane pass and reduction
    # are tested above; pointwise contraction moves results by up to one ULP,
    # inside the default float32 band.
    assert_matches(actual, expected)
    assert actual.shape == (2, 3, 4)
    assert actual.dtype == np.float32


def test_relion_cuda_fine_diff2_preserves_full_grid_zero_gap_lane_topology():
    full_size = 130
    compact_indices = np.asarray([0, 1, 129], dtype=np.int32)
    lookup = np.full(full_size, -1, dtype=np.int32)
    lookup[compact_indices] = np.arange(compact_indices.size, dtype=np.int32)
    # The two small terms combine before the large term only when their full
    # pixel lanes (1 and 129) are retained. Compacting them into lanes 1 and 2
    # loses both to float32 rounding against the large lane-0 term.
    terms = np.asarray([1.0e8, 4.0, 4.0], dtype=np.float32)
    reference = np.sqrt(np.float32(2.0) * terms).astype(np.complex64)
    shifted = np.zeros(compact_indices.size, dtype=np.complex64)
    weight = np.ones(compact_indices.size, dtype=np.float32)
    full_reference = np.zeros(full_size, dtype=np.complex64)
    full_shifted = np.zeros(full_size, dtype=np.complex64)
    full_weight = np.zeros(full_size, dtype=np.float32)
    full_reference[compact_indices] = reference
    full_shifted[compact_indices] = shifted
    full_weight[compact_indices] = weight

    expected = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(full_reference),
            jnp.asarray(full_shifted),
            jnp.asarray(full_weight),
        )
    )
    actual = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference),
            jnp.asarray(shifted),
            jnp.asarray(weight),
            jnp.asarray(lookup),
        )
    )
    wrong_compact_consecutive = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray(reference),
            jnp.asarray(shifted),
            jnp.asarray(weight),
        )
    )

    assert_matches(actual, expected)
    assert actual != wrong_compact_consecutive


def test_case20_current_grid_lookup_has_relion_56_by_29_topology():
    from relax.helpers.fourier_window import make_fourier_window_indices_np

    compact_indices, count = make_fourier_window_indices_np((256, 256), 56)
    lookup = _relion_cuda_fine_full_to_compact_lookup(
        (256, 256), 56, compact_indices
    )

    assert count == 1275
    assert lookup.shape == (56 * 29,)
    assert np.count_nonzero(lookup >= 0) == count
    assert_matches(np.sort(lookup[lookup >= 0]), np.arange(count, dtype=np.int32))


def test_full_size_lookup_is_a_bijection_of_the_complete_half_spectrum():
    image_shape = (8, 8)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    window_spec = make_fourier_window_spec(image_shape, image_shape[0], n_half)

    assert window_spec.use_window is False
    lookup = _relion_exact_fine_full_to_compact_lookup(
        image_shape,
        image_shape[0],
        n_half,
        window_spec,
    )

    assert lookup.shape == (n_half,)
    assert_matches(np.sort(lookup), np.arange(n_half, dtype=np.int32))

    # At full size ``use_window`` is false, but the identity lookup above is a
    # valid exact-fine representation. The big-JIT gate must therefore depend
    # only on the exact operand/preprocessing contract.
    _validate_relion_exact_fine_diff2_preconditions(
        relion_exact_fine_diff2=True,
        relion_exact_bpref_operands=True,
        use_relion_cuda_preprocess=True,
    )


def test_exact_fine_big_jit_still_rejects_missing_exact_preprocessing():
    with pytest.raises(ValueError, match="exact operands and CUDA preprocessing"):
        _validate_relion_exact_fine_diff2_preconditions(
            relion_exact_fine_diff2=True,
            relion_exact_bpref_operands=True,
            use_relion_cuda_preprocess=False,
        )


def test_relion_cuda_fine_diff2_handles_zero_pixels_and_nonfinite_scores():
    zero = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.zeros((2, 1, 0), dtype=jnp.complex64),
            jnp.zeros((1, 3, 0), dtype=jnp.complex64),
            jnp.zeros((1, 1, 0), dtype=jnp.float32),
        )
    )
    assert_matches(zero, np.zeros((2, 3), dtype=np.float32))

    # Sentinel gaps gather row zero for bounds safety; masking must use where,
    # because NaN * False would still contaminate RELION's zero-corr slots.
    gap_only = np.asarray(
        _relion_cuda_fine_diff2_sum(
            jnp.asarray([np.nan + 1j * np.nan], dtype=jnp.complex64),
            jnp.zeros(1, dtype=jnp.complex64),
            jnp.ones(1, dtype=jnp.float32),
            jnp.asarray([-1, -1, -1], dtype=jnp.int32),
        )
    )
    assert_matches(gap_only, np.asarray(0.0, dtype=np.float32))

    shifted = jnp.zeros((1, 1, 4), dtype=jnp.complex64)
    projection = jnp.zeros((1, 1, 4), dtype=jnp.complex64)
    scores = np.asarray(
        _score_pass2_bucket_relion_gpu_diff2(
            shifted,
            jnp.asarray([[np.nan, 1.0, 1.0, 1.0]], dtype=jnp.float32),
            projection,
            jnp.ones(4, dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.ones((1, 1, 1), dtype=bool),
        )
    )
    assert np.isneginf(scores[0, 0, 0])
