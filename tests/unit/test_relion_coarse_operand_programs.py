"""Equivalence tests for the jitted RELION coarse operand assemblies (P3-I).

The coarse operand assembly runs as one traced program per batch shape instead
of dispatching one compiled program per primitive. The eager function stays
the reference, so every test here compares the program's output against the
eager function it is ``jax.jit`` of: equal dtype and shape, values within the
default float band of ``helpers.float_compare``.

The two assemblies are the exact-source operands and the normalized-CC
(``--firstiter_cc``) operands (the generic sincosf assembly went with the generic
coarse scorer on 2026-10-02). The
padded last coarse batch and the inactive-support mask are covered because both
reach the production path that P3-B's ``RELAX_COARSE_PAD_FINAL_IMAGE_BATCH``
and T18b's fix ``1431919a7`` created.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers.float_compare import assert_matches

from relax.relion.relion_coarse_operands import (
    _relion_cc_coarse_operand_program,
    _relion_cc_coarse_operands,
    _relion_cc_inverse_power_from_processed,
    _relion_exact_coarse_operand_program,
    _relion_exact_coarse_operands,
    _repeat_pad_batch_axis,
    assemble_relion_cc_coarse_operands,
)

IMAGE_SIZE = 16
N_HALF = IMAGE_SIZE * (IMAGE_SIZE // 2 + 1)
N_SCORE = 37
ACTUAL_BATCH = 6
PADDED_BATCH = 8


def _assert_same_operand(actual, expected, what):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.dtype == expected.dtype, f"{what}: {actual.dtype} != {expected.dtype}"
    assert actual.shape == expected.shape, f"{what}: {actual.shape} != {expected.shape}"
    assert_matches(actual, expected, err_msg=what)


def _score_indices(rng):
    return jnp.asarray(
        np.sort(rng.choice(N_HALF, size=N_SCORE, replace=False)).astype(np.int32)
    )


def _active_mask(rng):
    mask = rng.random(N_SCORE) > 0.25
    # The support is never empty and never complete on a real current size.
    mask[0] = True
    mask[-1] = False
    return jnp.asarray(mask)


@pytest.mark.parametrize("use_float64_scoring", [False, True])
@pytest.mark.parametrize("scale_corrections_enabled", [False, True])
def test_exact_program_matches_the_eager_assembly(
    use_float64_scoring, scale_corrections_enabled
):
    rng = np.random.default_rng(20260921)
    real_dtype = jnp.float64 if use_float64_scoring else jnp.float32
    complex_dtype = jnp.complex128 if use_float64_scoring else jnp.complex64
    ctf = jnp.asarray(rng.uniform(-1.5, 1.5, (ACTUAL_BATCH, N_SCORE)), dtype=jnp.float64)
    # RELION's CTF crosses zero; the pixel correction has a magnitude guard there.
    ctf = ctf.at[:, ::9].set(0.0)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=real_dtype)
    processed = jnp.asarray(
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF)),
        dtype=complex_dtype,
    )
    indices = _score_indices(rng)
    mask = _active_mask(rng)
    noise = jnp.asarray(rng.uniform(1e3, 1e6, N_HALF), dtype=jnp.float64)
    half_weights = jnp.asarray(rng.uniform(0.5, 1.5, N_HALF), dtype=jnp.float64)

    kwargs = dict(
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        use_float64_scoring=use_float64_scoring,
        scale_corrections_enabled=scale_corrections_enabled,
    )
    eager = _relion_exact_coarse_operands(
        ctf, scale, processed, indices, mask, noise, half_weights, **kwargs
    )
    program = _relion_exact_coarse_operand_program(
        ctf, scale, processed, indices, mask, noise, half_weights, **kwargs
    )
    _assert_same_operand(program[0], eager[0], "exact unshifted_corrected")
    _assert_same_operand(program[1], eager[1], "exact pixel_weight")
    assert eager[0].dtype == complex_dtype
    assert eager[1].dtype == real_dtype


def test_exact_program_holds_on_a_repeat_padded_last_batch():
    """The padded coarse batch must give the live rows the unpadded answer.

    ``RELAX_COARSE_PAD_FINAL_IMAGE_BATCH`` repeats image row zero to fill the
    last batch of a half set. Padding may not move a science row, and the
    program must agree with the eager assembly on the padded extent as well.
    """

    rng = np.random.default_rng(20260922)
    ctf = np.asarray(rng.uniform(0.3, 1.5, (ACTUAL_BATCH, N_SCORE)), dtype=np.float64)
    scale = np.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=np.float32)
    processed = (
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF))
    ).astype(np.complex64)
    indices = _score_indices(rng)
    mask = _active_mask(rng)
    noise = jnp.asarray(rng.uniform(1e3, 1e6, N_HALF), dtype=jnp.float64)
    half_weights = jnp.asarray(rng.uniform(0.5, 1.5, N_HALF), dtype=jnp.float64)
    kwargs = dict(
        image_shape=(IMAGE_SIZE, IMAGE_SIZE),
        use_float64_scoring=False,
        scale_corrections_enabled=True,
    )

    padded_ctf = jnp.asarray(_repeat_pad_batch_axis(ctf, PADDED_BATCH))
    padded_scale = jnp.asarray(_repeat_pad_batch_axis(scale, PADDED_BATCH))
    padded_processed = jnp.asarray(_repeat_pad_batch_axis(processed, PADDED_BATCH))
    assert padded_ctf.shape == (PADDED_BATCH, N_SCORE)

    unpadded = _relion_exact_coarse_operands(
        jnp.asarray(ctf), jnp.asarray(scale), jnp.asarray(processed),
        indices, mask, noise, half_weights, **kwargs
    )
    padded_eager = _relion_exact_coarse_operands(
        padded_ctf, padded_scale, padded_processed,
        indices, mask, noise, half_weights, **kwargs
    )
    padded_program = _relion_exact_coarse_operand_program(
        padded_ctf, padded_scale, padded_processed,
        indices, mask, noise, half_weights, **kwargs
    )
    for name, i in (("unshifted_corrected", 0), ("pixel_weight", 1)):
        _assert_same_operand(padded_program[i], padded_eager[i], f"padded exact {name}")
        _assert_same_operand(
            padded_program[i][:ACTUAL_BATCH], unpadded[i], f"padded exact live rows {name}"
        )
        _assert_same_operand(
            padded_program[i][ACTUAL_BATCH:],
            np.repeat(np.asarray(unpadded[i][:1]), PADDED_BATCH - ACTUAL_BATCH, axis=0),
            f"padded exact repeated rows {name}",
        )


@pytest.mark.parametrize("with_window", [False, True])
@pytest.mark.parametrize("with_phase_factors", [False, True])
@pytest.mark.parametrize("scale_corrections_enabled", [False, True])
def test_cc_program_matches_the_eager_assembly(
    with_window, with_phase_factors, scale_corrections_enabled
):
    rng = np.random.default_rng(20260923)
    processed = jnp.asarray(
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF)),
        dtype=jnp.complex64,
    )
    ctf = jnp.asarray(rng.uniform(-1.5, 1.5, (ACTUAL_BATCH, N_HALF)), dtype=jnp.float64)
    ctf = ctf.at[:, ::11].set(0.0)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=jnp.float32)
    window = _score_indices(rng) if with_window else None
    inverse_power = _relion_cc_inverse_power_from_processed(processed, window)
    phase = (
        jnp.asarray(
            np.exp(1j * rng.uniform(-np.pi, np.pi, (ACTUAL_BATCH, N_HALF))),
            dtype=jnp.complex64,
        )
        if with_phase_factors
        else None
    )

    eager = _relion_cc_coarse_operands(
        processed, ctf, inverse_power, scale, phase, window,
        scale_corrections_enabled=scale_corrections_enabled,
    )
    program = _relion_cc_coarse_operand_program(
        processed, ctf, inverse_power, scale, phase, window,
        scale_corrections_enabled=scale_corrections_enabled,
    )
    for name in eager._fields:
        _assert_same_operand(getattr(program, name), getattr(eager, name), f"cc {name}")
    if window is None:
        _assert_same_operand(eager.windowed_unshifted, eager.unshifted_corrected, "cc unwindowed")
    else:
        assert eager.windowed_unshifted.shape == (ACTUAL_BATCH, N_SCORE)
        assert eager.windowed_corr_img.shape == (ACTUAL_BATCH, N_SCORE)


def test_cc_entry_point_runs_the_program():
    """The public entry point returns the program's operands, equal to the eager function's."""

    rng = np.random.default_rng(20260924)
    processed = jnp.asarray(
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF)),
        dtype=jnp.complex64,
    )
    ctf = jnp.asarray(rng.uniform(0.3, 1.5, (ACTUAL_BATCH, N_HALF)), dtype=jnp.float64)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=jnp.float32)
    inverse_power = _relion_cc_inverse_power_from_processed(processed, None)

    eager = _relion_cc_coarse_operands(
        processed, ctf, inverse_power, scale, None, None, scale_corrections_enabled=True
    )
    entry = assemble_relion_cc_coarse_operands(
        processed, ctf, inverse_power, scale, scale_corrections_enabled=True
    )
    for name, value in zip(entry._fields, eager):
        _assert_same_operand(getattr(entry, name), value, f"cc entry {name}")


def test_each_assembly_is_one_program_per_batch_shape():
    """The point of the program: one traced program, not one program per primitive.

    The eager path dispatches one compiled program per primitive it runs, so
    the number of equations in the assembly's jaxpr is the number of eager
    dispatches it costs per image batch (before JAX's index renormalization
    multiplies each fancy index by five). Under ``jax.jit`` the same chain is a
    single ``pjit`` equation.
    """

    rng = np.random.default_rng(20260925)
    ctf = jnp.asarray(rng.uniform(0.3, 1.5, (ACTUAL_BATCH, N_HALF)), dtype=jnp.float64)
    scale = jnp.asarray(rng.uniform(0.8, 1.2, ACTUAL_BATCH), dtype=jnp.float32)
    processed = jnp.asarray(
        rng.normal(size=(ACTUAL_BATCH, N_HALF)) + 1j * rng.normal(size=(ACTUAL_BATCH, N_HALF)),
        dtype=jnp.complex64,
    )
    inverse_power = _relion_cc_inverse_power_from_processed(processed, None)
    cc_args = (processed, ctf, inverse_power, scale, None, None)
    cc_eager = jax.make_jaxpr(
        lambda *a: _relion_cc_coarse_operands(*a, scale_corrections_enabled=True)
    )(*cc_args)
    cc_program = jax.make_jaxpr(
        lambda *a: _relion_cc_coarse_operand_program(*a, scale_corrections_enabled=True)
    )(*cc_args)
    assert len(cc_eager.eqns) >= 8, len(cc_eager.eqns)
    assert [str(eqn.primitive) for eqn in cc_program.eqns] in (["pjit"], ["jit"])
