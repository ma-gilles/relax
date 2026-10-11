"""Fast guardrail checks for dense/local EM refactor helpers.

The full dense/local guardrail command lives in ``scripts/run_em_fast_guard.sh``.
This file keeps the helper-specific assertions small enough to run alongside
existing dense and local exact EM equivalence tests.
"""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from relax.fourier.fourier_window import make_fourier_window_spec
from relax.fourier.image_shifts import half_image_phase_factors, tiled_half_image_phase_factors

pytestmark = pytest.mark.unit


def test_fourier_window_spec_gathers_last_axis_for_batched_values():
    image_shape = (8, 8)
    n_half = image_shape[0] * (image_shape[1] // 2 + 1)
    spec = make_fourier_window_spec(
        image_shape,
        current_size=6,
        n_half=n_half,
        include_recon_window=True,
    )
    values = jnp.arange(2 * 3 * n_half, dtype=jnp.float32).reshape(2, 3, n_half)

    assert_matches(
        np.asarray(spec.score_values(values)),
        np.asarray(values)[..., np.asarray(spec.score_indices_np)],
    )
    assert_matches(
        np.asarray(spec.recon_values(values)),
        np.asarray(values)[..., np.asarray(spec.recon_indices_np)],
    )


def test_half_image_pre_shift_phase_helpers_match_explicit_tiling():
    image_shape = (8, 8)
    shifts = jnp.asarray([[0.5, -1.25], [2.0, 0.0]], dtype=jnp.float32)
    phase = half_image_phase_factors(image_shape, shifts)
    tiled = tiled_half_image_phase_factors(image_shape, shifts, n_trans=3)

    assert phase.shape == (2, image_shape[0] * (image_shape[1] // 2 + 1))
    assert_matches(np.asarray(tiled), np.asarray(jnp.repeat(phase, 3, axis=0)))


def test_half_image_pre_shift_phase_requests_highest_precision():
    shifts = jnp.asarray([[0.5, -1.25], [2.0, 0.0]], dtype=jnp.float32)
    phase_jaxpr = jax.make_jaxpr(lambda value: half_image_phase_factors((8, 8), value))(shifts)

    jaxpr_text = str(phase_jaxpr)
    assert "dot_general[" in jaxpr_text
    assert "precision=(Precision.HIGHEST, Precision.HIGHEST)" in jaxpr_text
