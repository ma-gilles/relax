import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches
from helpers.mstep_reference import numpy_relion_f32_mstep_sums

from relax.fine_pass.local_backprojection import (
    compute_local_mstep_sums,
    compute_local_weighted_sums,
    compute_relion_sequential_mstep_sums,
)


def test_relion_f32_sequential_mstep_sums_match_numpy_translation_loop_exactly():
    probs = np.array([[[1.0, 1.0, 1.0], [0.5, 0.25, 0.125]]], dtype=np.float64)
    shifted = np.array(
        [[[1.0e8 + 2.0j, -8.0j], [1.0 + 4.0j, 2.0j], [-1.0e8 - 2.0j, 4.0j]]],
        dtype=np.complex64,
    )
    ctf2_over_nv = np.array([[1.0e8, 8.0]], dtype=np.float64)

    expected_y, expected_ctf = numpy_relion_f32_mstep_sums(probs, shifted, ctf2_over_nv)
    actual_y, actual_ctf = compute_relion_sequential_mstep_sums(
        probs.astype(np.float32), shifted, ctf2_over_nv.astype(np.float32)
    )

    assert actual_y.dtype == np.dtype(np.complex64)
    assert actual_ctf.dtype == np.dtype(np.float32)
    assert_matches(np.asarray(actual_y), expected_y)
    assert_matches(np.asarray(actual_ctf), expected_ctf)
    # This cancellation pattern distinguishes left-to-right float32 carrying
    # from a higher-precision or reassociated reduction.
    assert np.asarray(actual_y)[0, 0, 0] == np.complex64(0.0 + 4.0j)


def test_local_weighted_sums_requests_highest_dot_precision():
    probs = jnp.ones((1, 2, 3), dtype=jnp.float32)
    shifted = jnp.ones((1, 3, 4), dtype=jnp.complex64)

    jaxpr = str(jax.make_jaxpr(compute_local_weighted_sums)(probs, shifted))

    assert "Precision.HIGHEST" in jaxpr


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="GPU HLO regression")
def test_local_weighted_sums_requests_highest_precision_in_complex64_gpu_hlo():
    probs = jnp.ones((1, 2, 3), dtype=jnp.float32)
    shifted = jnp.ones((1, 3, 4), dtype=jnp.complex64)

    assert compute_local_weighted_sums(probs, shifted).dtype == jnp.complex64
    hlo = compute_local_weighted_sums.lower(probs, shifted).compiler_ir("hlo").as_hlo_text()

    assert "operand_precision={highest,highest}" in hlo


def test_local_weighted_sums_match_explicit_highest_precision_matmul():
    probs = jnp.asarray(
        [[[1.0, 1.0, 1.0], [0.5, 0.25, 0.125]]],
        dtype=jnp.float32,
    )
    shifted = jnp.asarray(
        [[[1.0e8 + 2.0j], [1.0 + 4.0j], [-1.0e8 - 2.0j]]],
        dtype=jnp.complex64,
    )

    actual = compute_local_weighted_sums(probs, shifted)
    expected = jnp.matmul(probs, shifted, precision=jax.lax.Precision.HIGHEST)

    assert_matches(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(
    "real_dtype,complex_dtype",
    [(np.float32, np.complex64), (np.float64, np.complex128)],
)
def test_local_mstep_sums_env_gate_preserves_xfloat_precision(
    monkeypatch, real_dtype, complex_dtype
):
    monkeypatch.setenv("RELAX_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION", "1")
    probs = np.array([[[1.0, 1.0, 1.0]]], dtype=real_dtype)
    shifted = np.array(
        [[[1.0e8 + 0j], [1.0 + 0j], [-1.0e8 + 0j]]], dtype=complex_dtype
    )
    ctf2_over_nv = np.array([[2.0]], dtype=real_dtype)

    xhalf_y, xhalf_ctf = compute_local_mstep_sums(probs, shifted, ctf2_over_nv, relion_x_half=True)
    normal_y, normal_ctf = compute_local_mstep_sums(probs, shifted, ctf2_over_nv, relion_x_half=False)

    assert xhalf_y.dtype == np.dtype(complex_dtype)
    assert xhalf_ctf.dtype == np.dtype(real_dtype)
    assert normal_y.dtype == np.dtype(complex_dtype)
    assert normal_ctf.dtype == np.dtype(real_dtype)
    expected_value = 0.0 if real_dtype == np.float32 else 1.0
    assert np.asarray(xhalf_y)[0, 0, 0] == expected_value
    assert_matches(np.asarray(normal_ctf), np.array([[[6.0]]], dtype=real_dtype))


def test_local_mstep_sums_preserve_promoted_precision_for_mixed_operands(monkeypatch):
    monkeypatch.setenv("RELAX_RELION_X_HALF_SEQUENTIAL_TRANSLATION_REDUCTION", "1")
    probs = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float64)
    shifted = np.array([[[1.0e8 + 0j], [1.0 + 0j], [-1.0e8 + 0j]]], dtype=np.complex64)
    ctf2_over_nv = np.array([[2.0]], dtype=np.float64)

    xhalf_y, xhalf_ctf = compute_local_mstep_sums(probs, shifted, ctf2_over_nv, relion_x_half=True)
    normal_y, normal_ctf = compute_local_mstep_sums(probs, shifted, ctf2_over_nv, relion_x_half=False)

    assert xhalf_y.dtype == np.dtype(np.complex128)
    assert xhalf_ctf.dtype == np.dtype(np.float64)
    assert normal_y.dtype == np.dtype(np.complex128)
    assert normal_ctf.dtype == np.dtype(np.float64)
    assert np.asarray(xhalf_y)[0, 0, 0] == 1.0
    assert np.asarray(normal_y)[0, 0, 0] == 1.0
    assert_matches(np.asarray(normal_ctf), np.array([[[6.0]]], dtype=np.float64))
    assert_matches(xhalf_ctf, normal_ctf)
