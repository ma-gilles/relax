"""Nonfinite-score guardrails for exact-local EM's big-JIT normalizer."""

import numpy as np
import pytest
from helpers.float_compare import assert_matches

pytest.importorskip("jax")
import jax.numpy as jnp

from relax.fine_pass.local_backprojection import compute_local_ctf_sums
from relax.projection.projection import compute_noise_block

pytestmark = pytest.mark.unit


def _mstep_outputs(result):
    arrays = [np.asarray(x) for x in result]
    return {
        "log_z": arrays[0],
        "best_log_score": arrays[1],
        "best_argmax": arrays[2],
        "max_posterior": arrays[3],
        "reconstruction_sample_mask": arrays[4],
        "reconstruction_rotation_mask": arrays[5],
        "n_significant_samples": arrays[6],
        "reconstruction_probs": arrays[7],
        "probs_sum_t": arrays[8],
        "reconstruction_probs_sum_t": arrays[9],
        "summed": arrays[10],
        "ctf_probs": arrays[11],
    }


def _base_inputs():
    batch_size, n_rot, n_trans, n_half = 2, 1, 2, 3
    shifted = jnp.asarray(
        [
            [[0.2 + 0.1j, -0.1 + 0.3j, 0.4 - 0.2j], [0.1 - 0.2j, 0.3 + 0.2j, -0.2 + 0.1j]],
            [[np.inf + 0.0j, np.inf + 0.0j, np.inf + 0.0j], [np.inf + 0.0j, np.inf + 0.0j, np.inf + 0.0j]],
        ],
        dtype=jnp.complex64,
    )
    return dict(
        shifted_score_split=shifted,
        ctf2_over_nv_score=jnp.ones((batch_size, n_half), dtype=jnp.float32),
        proj_weighted=jnp.ones((batch_size, n_rot, n_half), dtype=jnp.complex64),
        half_weights=jnp.ones((n_half,), dtype=jnp.float32),
        rotation_log_prior=jnp.zeros((batch_size, n_rot), dtype=jnp.float32),
        translation_log_prior=jnp.zeros((batch_size, n_trans), dtype=jnp.float32),
        rotation_mask=jnp.ones((batch_size, n_rot), dtype=bool),
        sample_mask=jnp.ones((batch_size, n_rot, n_trans), dtype=bool),
        valid_image_mask=jnp.ones((batch_size,), dtype=bool),
        normalization_log_z=jnp.zeros((batch_size,), dtype=jnp.float32),
        shifted_recon_split=jnp.ones((batch_size, n_trans, n_half), dtype=jnp.complex64),
        ctf2_over_nv_recon=jnp.ones((batch_size, n_half), dtype=jnp.float32),
    )


def test_compute_noise_block_zero_weight_nonfinite_projection_is_zero():
    """Zero posterior support must not become NaN via 0 * inf products."""

    proj_half = jnp.asarray(
        [
            [complex(np.inf, 1.0), complex(1.0, np.inf), complex(np.inf, np.inf)],
            [complex(np.inf, 2.0), complex(2.0, np.inf), complex(np.inf, np.inf)],
        ],
        dtype=jnp.complex64,
    )
    proj_abs2_half = jnp.abs(proj_half) ** 2
    summed_masked = jnp.zeros_like(proj_half)
    ctf_probs = jnp.zeros((2, 3), dtype=jnp.float32)
    noise_variance_half = jnp.asarray([1.0, np.inf, np.inf], dtype=jnp.float32)
    shell_indices = jnp.asarray([0, 1, 2], dtype=jnp.int32)

    noise_shells, a2_shells, xa_shells = compute_noise_block(
        proj_half,
        proj_abs2_half,
        summed_masked,
        ctf_probs,
        noise_variance_half,
        shell_indices,
        3,
        True,
    )

    assert_matches(np.asarray(noise_shells), np.zeros(3, dtype=np.float32))
    assert_matches(np.asarray(a2_shells), np.zeros(3, dtype=np.float32))
    assert_matches(np.asarray(xa_shells), np.zeros(3, dtype=np.float32))


def test_compute_local_ctf_sums_zero_mass_nonfinite_ctf_is_zero():
    """Zero posterior CTF rows must not become NaN when CTF/noise is nonfinite."""

    probs = jnp.asarray(
        [
            [[0.25, 0.75]],
            [[0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    ctf2_over_nv = jnp.asarray([[1.0, 2.0, 3.0], [jnp.inf, jnp.inf, jnp.inf]], dtype=jnp.float32)

    ctf_sums = np.asarray(compute_local_ctf_sums(probs, ctf2_over_nv))

    assert np.all(np.isfinite(ctf_sums))
    assert_matches(ctf_sums[1], np.zeros_like(ctf_sums[1]))


